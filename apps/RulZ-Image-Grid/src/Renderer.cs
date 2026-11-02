using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RulZImageGrid
{
    public static class Renderer
    {
        /// <summary>Renders the grid to a bitmap. Loads each source image at exactly the size it needs.</summary>
        public static BitmapSource Render(GridSettings s, IList<ImageItem> items, double scale, bool forceOpaque,
            Action<string> progress)
        {
            var layout = GridLayout.Compute(s, items.Select(i => new Size(i.PixelW, i.PixelH)).ToList());
            int w = Math.Max(1, (int)Math.Round(layout.Width * scale));
            int h = Math.Max(1, (int)Math.Round(layout.Height * scale));

            var dv = new DrawingVisual();
            RenderOptions.SetBitmapScalingMode(dv, BitmapScalingMode.HighQuality);
            using (var dc = dv.RenderOpen())
            {
                dc.PushTransform(new ScaleTransform(scale, scale));
                Color bg = s.Transparent ? (forceOpaque ? Colors.White : Colors.Transparent) : s.Background;
                dc.DrawRectangle(Frozen(bg), null, new Rect(0, 0, layout.Width, layout.Height));
                var textBrush = Frozen(s.ResolveTextColor());

                for (int i = 0; i < items.Count; i++)
                {
                    progress?.Invoke($"Loading image {i + 1} of {items.Count}…");
                    var it = items[i];
                    var r = layout.ImageRect(i, it.PixelW, it.PixelH);
                    int target = Math.Max(1, (int)Math.Ceiling(r.Width * scale));
                    var bmp = ImageLoader.Load(it.Path, it.Orientation, target < it.PixelW ? target : 0);
                    dc.DrawImage(bmp, r);

                    if (s.ShowCaptions)
                    {
                        var cr = layout.CaptionRect(i);
                        var ft = TextMetrics.Make(it.Name, s.CaptionFont, false, textBrush);
                        ft.MaxTextWidth = cr.Width;
                        ft.MaxLineCount = 1;
                        ft.Trimming = TextTrimming.CharacterEllipsis;
                        ft.TextAlignment = TextAlignment.Center;
                        dc.DrawText(ft, cr.TopLeft);
                    }
                }

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

            progress?.Invoke("Compositing…");
            var rtb = new RenderTargetBitmap(w, h, 96, 96, PixelFormats.Pbgra32);
            rtb.Render(dv);
            rtb.Freeze();
            return rtb;
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
            var tmp = path + ".tmp";
            using (var fs = File.Create(tmp)) enc.Save(fs);
            if (File.Exists(path)) File.Delete(path);
            File.Move(tmp, path);
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
            });
            t.SetApartmentState(ApartmentState.STA);
            t.IsBackground = true;
            t.Start();
            return tcs.Task;
        }
    }
}
