using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace RulZImageGrid
{
    public static class Renderer
    {
        /// <summary>
        /// Renders the grid to a bitmap. Loads each source image at exactly the size it needs and draws it into the
        /// output straight away, so only one decoded image is in memory at a time rather than all of them. With
        /// <paramref name="useThumbs"/> it uses the in-memory preview of any image whose preview is big enough, or
        /// whose file can no longer be read (for the export dialog's preview, so it doesn't re-read every file).
        /// </summary>
        public static BitmapSource Render(GridSettings s, IList<ImageItem> items, double scale, bool forceOpaque,
            Action<string> progress, bool useThumbs = false, CancellationToken cancel = default)
        {
            var layout = GridLayout.Compute(s, items.Select(i => new Size(i.PixelW, i.PixelH)).ToList());
            int w = Math.Max(1, (int)Math.Round(layout.Width * scale));
            int h = Math.Max(1, (int)Math.Round(layout.Height * scale));
            var rtb = new RenderTargetBitmap(w, h, 96, 96, PixelFormats.Pbgra32);
            var zoom = new ScaleTransform(scale, scale);
            zoom.Freeze();
            var textBrush = Frozen(s.ResolveTextColor());

            var back = new DrawingVisual();
            using (var dc = back.RenderOpen())
            {
                dc.PushTransform(zoom);
                Color bg = s.Transparent ? (forceOpaque ? Colors.White : Colors.Transparent) : s.Background;
                dc.DrawRectangle(Frozen(bg), null, new Rect(0, 0, layout.Width, layout.Height));
            }
            rtb.Render(back);

            // Each tile is drawn on top of what's there already (RenderTargetBitmap.Render adds to the bitmap), and
            // its decoded image can be freed before the next one loads.
            for (int i = 0; i < items.Count; i++)
            {
                cancel.ThrowIfCancellationRequested();
                progress?.Invoke($"Loading image {i + 1} of {items.Count}…");
                var it = items[i];
                var r = layout.ImageRect(i, it.PixelW, it.PixelH);
                int target = Math.Max(1, (int)Math.Ceiling(r.Width * scale));
                var bmp = useThumbs && it.Thumb != null && it.Thumb.PixelWidth >= target
                    ? it.Thumb
                    : LoadForRender(it, target < it.PixelW ? target : 0, useThumbs);

                var tile = new DrawingVisual();
                RenderOptions.SetBitmapScalingMode(tile, BitmapScalingMode.HighQuality);
                using (var dc = tile.RenderOpen())
                {
                    dc.PushTransform(zoom);
                    dc.DrawImage(bmp, r);
                    if (s.ShowCaptions) TextMetrics.DrawCaption(dc, layout, i, it.Name, s.CaptionFont, textBrush);
                }
                rtb.Render(tile);
            }

            var labels = new DrawingVisual();
            using (var dc = labels.RenderOpen())
            {
                dc.PushTransform(zoom);
                if (s.ShowColLabels)
                    for (int c = 0; c < layout.Cols; c++)
                    {
                        var text = s.ColLabel(c);
                        if (text.Length == 0) continue;
                        var hr = layout.ColHeaderRect(c);
                        var ft = TextMetrics.Make(text, s.LabelFont, true, textBrush);
                        ft.MaxTextWidth = hr.Width;
                        ft.MaxLineCount = 1;
                        ft.Trimming = TextTrimming.CharacterEllipsis;
                        ft.TextAlignment = TextAlignment.Center;
                        dc.DrawText(ft, hr.TopLeft);
                    }

                if (s.ShowRowLabels)
                    for (int r = 0; r < layout.Rows; r++)
                    {
                        var text = s.RowLabel(r);
                        if (text.Length == 0) continue;
                        var hr = layout.RowHeaderRect(r);
                        var ft = TextMetrics.Make(text, s.LabelFont, true, textBrush);
                        ft.MaxTextWidth = hr.Width;
                        ft.MaxLineCount = 1;
                        ft.Trimming = TextTrimming.CharacterEllipsis;
                        ft.TextAlignment = TextAlignment.Right;
                        dc.DrawText(ft, new Point(hr.X, hr.Y + (hr.Height - ft.Height) / 2));
                    }
            }
            rtb.Render(labels);
            rtb.Freeze();
            return rtb;
        }

        /// <summary>
        /// Loads one source image for a render. A file moved, deleted or broken since it was added falls back to the
        /// in-memory preview when drawing a preview, and otherwise fails with the file's name in the message.
        /// </summary>
        static BitmapSource LoadForRender(ImageItem it, int width, bool previewOk)
        {
            try { return ImageLoader.Load(it.Path, it.Orientation, width); }
            catch (Exception ex) when (!(ex is OutOfMemoryException))
            {
                if (previewOk && it.Thumb != null) return it.Thumb;
                throw new IOException($"\"{Path.GetFileName(it.Path)}\" couldn't be read: {ex.Message}", ex);
            }
        }

        public static void Save(BitmapSource bmp, string path, bool jpg, int quality, bool keepAlpha)
        {
            BitmapEncoder enc;
            if (jpg)
            {
                enc = new JpegBitmapEncoder { QualityLevel = Math.Max(1, Math.Min(100, quality)) };
                enc.Frames.Add(BitmapFrame.Create(new FormatConvertedBitmap(bmp, PixelFormats.Bgr24, null, 0)));
            }
            else
            {
                enc = new PngBitmapEncoder();
                var src = keepAlpha ? (BitmapSource)new FormatConvertedBitmap(bmp, PixelFormats.Bgra32, null, 0)
                                    : new FormatConvertedBitmap(bmp, PixelFormats.Bgr24, null, 0);
                enc.Frames.Add(BitmapFrame.Create(src));
            }
            // Write to a temporary file in the same folder first, so a failed save never leaves a half-written image
            // and an existing file is only replaced once the new one is complete. The short random name can't clash
            // with the user's own files (CreateNew refuses to overwrite) and is never longer than the target's name
            // could be.
            string tmp = SiblingName(path, ".tmp");
            try
            {
                using (var fs = new FileStream(tmp, FileMode.CreateNew, FileAccess.Write, FileShare.None)) enc.Save(fs);
                SwapIn(tmp, path);
            }
            finally { TryDelete(tmp); }
        }

        static string SiblingName(string path, string ext) =>
            Path.Combine(Path.GetDirectoryName(Path.GetFullPath(path)), "~" + Path.GetRandomFileName() + ext);

        /// <summary>Moves the finished file into place. The old file is never removed before the new one is there.</summary>
        static void SwapIn(string tmp, string path)
        {
            if (!File.Exists(path))
            {
                File.Move(tmp, path);
                return;
            }
            string bak = SiblingName(path, ".bak");
            try
            {
                try { File.Replace(tmp, path, bak, true); }
                catch (IOException) when (File.Exists(tmp) && File.Exists(path) && !File.Exists(bak))
                {
                    // Nothing was moved (some network drives can't replace files): swap by hand.
                    File.Move(path, bak);
                    File.Move(tmp, path);
                }
            }
            finally
            {
                // If the old file was moved aside but the new one didn't take its place, put the old one back.
                if (!File.Exists(path) && File.Exists(bak))
                {
                    try { File.Move(bak, path); }
                    catch (IOException) { }
                    catch (UnauthorizedAccessException) { }
                }
                else TryDelete(bak);
            }
        }

        static void TryDelete(string file)
        {
            try { if (File.Exists(file)) File.Delete(file); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }

        static SolidColorBrush Frozen(Color c)
        {
            var b = new SolidColorBrush(c);
            b.Freeze();
            return b;
        }

        /// <summary>Runs work on a dedicated STA thread (WPF imaging and text need STA).</summary>
        public static Task<T> RunSta<T>(Func<T> work)
        {
            var tcs = new TaskCompletionSource<T>();
            var t = new Thread(() =>
            {
                try { tcs.SetResult(work()); }
                catch (Exception ex) { tcs.SetException(ex); }
                // WPF gives the thread a dispatcher (with a hidden window and render resources) the first time it
                // touches WPF objects. Shut it down, or every call leaks about a dozen handles and some memory.
                finally { Dispatcher.FromThread(Thread.CurrentThread)?.InvokeShutdown(); }
            });
            t.SetApartmentState(ApartmentState.STA);
            t.IsBackground = true;
            t.Start();
            return tcs.Task;
        }
    }
}
