using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RulZImageGrid
{
    /// <summary>All user-adjustable grid options. Sizes are in output pixels at 100% scale.</summary>
    public class GridSettings
    {
        public int Columns = 3;
        public int Padding = 20;
        public bool ShowCaptions = true;
        public double CaptionFont = 24;
        public bool ShowColLabels = true;
        public bool ShowRowLabels = true;
        public double LabelFont = 30;
        public List<string> ColLabels = new List<string>();
        public List<string> RowLabels = new List<string>();
        public Color Background = Colors.White;
        public bool Transparent;
        public int TextColorMode; // 0 auto, 1 black, 2 white, 3 gray
        public bool Upscale = true;

        public string ColLabel(int c) => c < ColLabels.Count ? ColLabels[c].Trim() : "";
        public string RowLabel(int r) => r < RowLabels.Count ? RowLabels[r].Trim() : "";

        public Color ResolveTextColor()
        {
            switch (TextColorMode)
            {
                case 1: return Colors.Black;
                case 2: return Colors.White;
                case 3: return Color.FromRgb(0x80, 0x80, 0x80);
            }
            if (Transparent) return Color.FromRgb(0x20, 0x20, 0x20);
            double lum = (0.299 * Background.R + 0.587 * Background.G + 0.114 * Background.B) / 255.0;
            return lum > 0.5 ? Color.FromRgb(0x1E, 0x1E, 0x1E) : Color.FromRgb(0xF0, 0xF0, 0xF0);
        }

        public GridSettings Clone()
        {
            var c = (GridSettings)MemberwiseClone();
            c.ColLabels = new List<string>(ColLabels);
            c.RowLabels = new List<string>(RowLabels);
            return c;
        }

        /// <summary>
        /// The settings as saved. The editor keeps room for faint "Column 1" / "Row 1" placeholders, but a label
        /// set with nothing typed for any of the grid's columns (or rows) takes no space in the saved image.
        /// </summary>
        public GridSettings ForExport(int imageCount)
        {
            var c = Clone();
            int cols = Math.Max(1, Columns);
            int rows = Math.Max(1, (int)Math.Ceiling(imageCount / (double)cols));
            if (!Enumerable.Range(0, cols).Any(i => ColLabel(i).Length > 0)) c.ShowColLabels = false;
            if (!Enumerable.Range(0, rows).Any(i => RowLabel(i).Length > 0)) c.ShowRowLabels = false;
            return c;
        }
    }

    public class ImageItem
    {
        public string Path;
        public string Name;
        public int PixelW, PixelH;   // display size (after EXIF orientation)
        public int Orientation = 1;  // EXIF orientation tag
        public BitmapSource Thumb;
    }

    public static class TextMetrics
    {
        // One pair per thread: the editor draws every file name each rebuild, and the export runs on its own thread.
        [ThreadStatic] static Typeface regular, bold;

        public static Typeface Face(bool semiBold) => semiBold
            ? bold ?? (bold = new Typeface(new FontFamily("Segoe UI"), FontStyles.Normal, FontWeights.SemiBold, FontStretches.Normal))
            : regular ?? (regular = new Typeface(new FontFamily("Segoe UI"), FontStyles.Normal, FontWeights.Normal, FontStretches.Normal));

        public static FormattedText Make(string text, double size, bool bold, Brush brush, double pixelsPerDip = 1.0) =>
            new FormattedText(text ?? "", CultureInfo.CurrentUICulture, FlowDirection.LeftToRight,
                Face(bold), Math.Max(1, size), brush ?? Brushes.Black, pixelsPerDip);

        /// <summary>Draws a file name under its tile. The export and the editor both use this, so they match.</summary>
        public static void DrawCaption(DrawingContext dc, GridLayout layout, int index, string name, double size,
            Brush brush, double pixelsPerDip = 1.0)
        {
            var cr = layout.CaptionRect(index);
            var ft = Make(name, size, false, brush, pixelsPerDip);
            ft.MaxTextWidth = cr.Width;
            ft.MaxLineCount = 1;
            ft.Trimming = TextTrimming.CharacterEllipsis;
            ft.TextAlignment = TextAlignment.Center;
            dc.DrawText(ft, cr.TopLeft);
        }

        public static double LineHeight(double size, bool bold) => Make("Ag", size, bold, null).Height;

        public static double Width(string text, double size, bool bold) =>
            Make(text, size, bold, null).WidthIncludingTrailingWhitespace;
    }

    public static class ImageLoader
    {
        public static readonly string[] Extensions =
            { ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".heic", ".heif", ".jfif", ".ico" };

        /// <summary>
        /// Longest side of the preview kept in memory for each image. The editor loads sharper copies of the
        /// tiles on screen when they're shown bigger than this, so it can stay small.
        /// </summary>
        public const int ThumbSide = 512;

        public static bool IsImage(string path) =>
            Extensions.Contains(System.IO.Path.GetExtension(path).ToLowerInvariant());

        static bool Rotated90(int o) => o >= 5 && o <= 8;

        /// <summary>
        /// The image files among dropped or picked paths. A folder adds its own images (not subfolders') in name order,
        /// img2 before img10. A folder or file that can't be read is listed in <c>failed</c> instead of stopping the rest.
        /// Can be slow on big or network folders, so the window calls it off the UI thread.
        /// </summary>
        public static (List<string> files, List<string> failed) FindImages(IEnumerable<string> paths)
        {
            var natural = new NaturalComparer();
            var files = new List<string>();
            var failed = new List<string>();
            foreach (var p in paths)
            {
                try
                {
                    if (Directory.Exists(p))
                        files.AddRange(Directory.EnumerateFiles(p).Where(IsImage)
                            .OrderBy(f => System.IO.Path.GetFileName(f), natural));
                    else if (File.Exists(p) && IsImage(p))
                        files.Add(p);
                }
                catch (Exception ex) when (ex is IOException || ex is UnauthorizedAccessException
                                           || ex is System.Security.SecurityException || ex is ArgumentException)
                {
                    failed.Add(ex.Message);
                }
            }
            return (files, failed);
        }

        /// <summary>
        /// Reads many images on several background threads at once. Results keep the order of <paramref name="files"/>.
        /// <paramref name="progress"/> is called from the worker threads with the number of files done so far.
        /// </summary>
        public static async Task<(List<ImageItem> ok, List<string> failed)> ReadInfoMany(IList<string> files, Action<int> progress)
        {
            var results = new ImageItem[files.Count];
            var errors = new string[files.Count];
            int next = -1, done = 0;
            // A few threads give most of the speed-up; more would mostly add memory for big PNGs decoded at once.
            int workers = Math.Max(1, Math.Min(Math.Min(Environment.ProcessorCount, 6), files.Count));
            var tasks = new Task<bool>[workers];
            for (int w = 0; w < workers; w++)
                tasks[w] = Renderer.RunSta(() =>
                {
                    int i;
                    while ((i = Interlocked.Increment(ref next)) < files.Count)
                    {
                        try { results[i] = ReadInfo(files[i]); }
                        catch (Exception ex) { errors[i] = $"{System.IO.Path.GetFileName(files[i])}: {ex.Message}"; }
                        progress?.Invoke(Interlocked.Increment(ref done));
                    }
                    return true;
                });
            await Task.WhenAll(tasks);
            return (results.Where(r => r != null).ToList(), errors.Where(e => e != null).ToList());
        }

        /// <summary>Reads dimensions and EXIF orientation, then loads a small thumbnail.</summary>
        public static ImageItem ReadInfo(string path)
        {
            int w, h, orient = 1;
            using (var fs = File.OpenRead(path))
            {
                var dec = BitmapDecoder.Create(fs, BitmapCreateOptions.DelayCreation, BitmapCacheOption.None);
                var f = dec.Frames[0];
                w = f.PixelWidth;
                h = f.PixelHeight;
                try
                {
                    if (f.Metadata is BitmapMetadata md)
                    {
                        foreach (var q in new[] { "/app1/ifd/{ushort=274}", "/ifd/{ushort=274}" })
                        {
                            try
                            {
                                if (md.ContainsQuery(q))
                                {
                                    var o = md.GetQuery(q);
                                    if (o != null) orient = Convert.ToInt32(o);
                                    break;
                                }
                            }
                            catch { }
                        }
                    }
                }
                catch { }
            }
            if (orient < 1 || orient > 8) orient = 1;
            if (Rotated90(orient)) { var t = w; w = h; h = t; }

            var item = new ImageItem
            {
                Path = path,
                Name = System.IO.Path.GetFileNameWithoutExtension(path),
                PixelW = w,
                PixelH = h,
                Orientation = orient,
            };
            double s = Math.Min(1.0, (double)ThumbSide / Math.Max(w, h));
            item.Thumb = Load(path, orient, s < 1 ? Math.Max(1, (int)Math.Round(w * s)) : 0);
            return item;
        }

        /// <summary>Loads an image with orientation applied. displayWidth = 0 loads full resolution.</summary>
        public static BitmapSource Load(string path, int orientation, int displayWidth)
        {
            var bi = new BitmapImage();
            using (var fs = File.OpenRead(path))
            {
                bi.BeginInit();
                bi.CacheOption = BitmapCacheOption.OnLoad;
                bi.StreamSource = fs; // stream-loaded images bypass WPF's URI cache (IgnoreImageCache would throw here)
                if (displayWidth > 0)
                {
                    if (Rotated90(orientation)) bi.DecodePixelHeight = displayWidth;
                    else bi.DecodePixelWidth = displayWidth;
                }
                bi.EndInit();
            }
            bi.Freeze();

            BitmapSource src = bi;
            Transform t = OrientationTransform(orientation);
            if (t != null)
            {
                try
                {
                    var tb = new TransformedBitmap(src, t);
                    tb.Freeze();
                    src = tb;
                }
                catch { }
            }
            return src;
        }

        static Transform OrientationTransform(int o)
        {
            var flip = new ScaleTransform(-1, 1);
            switch (o)
            {
                case 2: return flip;
                case 3: return new RotateTransform(180);
                case 4: return new ScaleTransform(1, -1);
                case 5: return new TransformGroup { Children = { new RotateTransform(90), flip } };
                case 6: return new RotateTransform(90);
                case 7: return new TransformGroup { Children = { new RotateTransform(270), flip } };
                case 8: return new RotateTransform(270);
            }
            return null;
        }
    }

    /// <summary>
    /// Computes where everything goes, in output pixels at 100% scale.
    /// Shared by the on-screen editor and the exporter so they always match.
    /// </summary>
    public class GridLayout
    {
        public int Cols, Rows, Count;
        public double P, CellW, CellH, CaptionGap, CaptionH, LabelGap, LabelLineH, ColHeaderH, RowHeaderW;
        public double X0, Y0, Width, Height;
        public bool Upscale;

        public static GridLayout Compute(GridSettings s, IList<Size> sizes)
        {
            var L = new GridLayout();
            L.Count = sizes.Count;
            L.Cols = Math.Max(1, s.Columns);
            L.Rows = Math.Max(1, (int)Math.Ceiling(L.Count / (double)L.Cols));
            L.P = Math.Max(0, s.Padding);
            L.Upscale = s.Upscale;
            L.CellW = sizes.Count > 0 ? Math.Round(sizes.Max(z => z.Width)) : 400;
            L.CellH = sizes.Count > 0 ? Math.Round(sizes.Max(z => z.Height)) : 300;

            if (s.ShowCaptions)
            {
                L.CaptionGap = Math.Round(s.CaptionFont * 0.3);
                L.CaptionH = L.CaptionGap + Math.Ceiling(TextMetrics.LineHeight(s.CaptionFont, false));
            }

            L.LabelGap = Math.Max(L.P, Math.Round(s.LabelFont * 0.4));
            L.LabelLineH = Math.Ceiling(TextMetrics.LineHeight(s.LabelFont, true));
            if (s.ShowColLabels)
                L.ColHeaderH = L.LabelLineH + L.LabelGap;
            if (s.ShowRowLabels)
            {
                double w = 0;
                for (int r = 0; r < L.Rows; r++)
                    w = Math.Max(w, TextMetrics.Width(s.RowLabel(r), s.LabelFont, true));
                if (w <= 0) w = TextMetrics.Width("Row 00", s.LabelFont, true); // room for the on-screen placeholder
                L.RowHeaderW = Math.Ceiling(w) + L.LabelGap;
            }

            L.X0 = L.P + L.RowHeaderW;
            L.Y0 = L.P + L.ColHeaderH;
            L.Width = L.X0 + L.Cols * L.CellW + (L.Cols - 1) * L.P + L.P;
            L.Height = L.Y0 + L.Rows * (L.CellH + L.CaptionH) + (L.Rows - 1) * L.P + L.P;
            return L;
        }

        public double RowStride => CellH + CaptionH + P;
        public double ColX(int c) => X0 + c * (CellW + P);
        public double RowY(int r) => Y0 + r * RowStride;

        public Rect CellRect(int i) => new Rect(ColX(i % Cols), RowY(i / Cols), CellW, CellH);

        public Rect CaptionRect(int i) =>
            new Rect(ColX(i % Cols), RowY(i / Cols) + CellH + CaptionGap, CellW, Math.Max(1, CaptionH - CaptionGap));

        /// <summary>Image fitted inside its cell, centred horizontally and sitting on the caption.</summary>
        public Rect ImageRect(int i, double w, double h)
        {
            var cell = CellRect(i);
            double sc = Math.Min(CellW / w, CellH / h);
            if (!Upscale) sc = Math.Min(sc, 1);
            double iw = Math.Max(1, Math.Round(w * sc)), ih = Math.Max(1, Math.Round(h * sc));
            return new Rect(cell.X + Math.Round((CellW - iw) / 2), cell.Y + (CellH - ih), iw, ih);
        }

        public Rect ColHeaderRect(int c) => new Rect(ColX(c), P, CellW, LabelLineH);
        public Rect RowHeaderRect(int r) => new Rect(P, RowY(r), Math.Max(1, RowHeaderW - LabelGap), CellH);

        /// <summary>Index of the tile under the point, or -1.</summary>
        public int HitTest(Point p)
        {
            if (p.X < X0 || p.Y < Y0) return -1;
            int c = (int)Math.Floor((p.X - X0) / (CellW + P));
            int r = (int)Math.Floor((p.Y - Y0) / RowStride);
            if (c < 0 || c >= Cols || r < 0) return -1;
            if (p.X > ColX(c) + CellW || p.Y > RowY(r) + CellH + CaptionH) return -1;
            int i = r * Cols + c;
            return i < Count ? i : -1;
        }

        /// <summary>Nearest grid slot to the point, clamped to [0, maxIndex].</summary>
        public int DropIndex(Point p, int maxIndex)
        {
            int c = (int)Math.Floor((p.X - X0 + P / 2) / (CellW + P));
            int r = (int)Math.Floor((p.Y - Y0 + P / 2) / RowStride);
            c = Math.Max(0, Math.Min(Cols - 1, c));
            r = Math.Max(0, r);
            return Math.Max(0, Math.Min(maxIndex, r * Cols + c));
        }
    }
}
