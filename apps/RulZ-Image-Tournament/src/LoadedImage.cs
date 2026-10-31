using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Media.Imaging;

namespace RulZ.ImageTournament
{
    // An image decoded fully into memory (so the file is never locked and can be moved or
    // deleted), plus progressively halved copies used to draw quickly when zoomed out.
    sealed class LoadedImage : IDisposable
    {
        static readonly HashSet<string> Extensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".dib", ".gif", ".tif", ".tiff",
            ".webp", ".heic", ".heif", ".avif", ".jxr", ".wdp", ".ico",
        };

        const int OrientationTag = 0x0112;
        const int SmallestMip = 1024;

        readonly List<Bitmap> _mips = new List<Bitmap>();

        public string Path { get; private set; }
        public long FileSize { get; private set; }
        public string Error { get; private set; }
        public Bitmap Bitmap { get { return _mips.Count > 0 ? _mips[0] : null; } }
        public IList<Bitmap> Mips { get { return _mips; } }
        public string FileName { get { return System.IO.Path.GetFileName(Path); } }

        public static bool IsImageFile(string path)
        {
            return Extensions.Contains(System.IO.Path.GetExtension(path));
        }

        public static string FileDialogFilter
        {
            get { return "Images|*" + string.Join(";*", Extensions) + "|All files|*.*"; }
        }

        public static LoadedImage Load(string path)
        {
            var image = new LoadedImage { Path = path };
            try
            {
                byte[] data = File.ReadAllBytes(path);
                image.FileSize = data.Length;
                Bitmap bmp = DecodeGdi(data) ?? DecodeWic(data);
                image._mips.Add(bmp);
                while (Math.Max(bmp.Width, bmp.Height) > SmallestMip)
                {
                    bmp = Half(bmp);
                    image._mips.Add(bmp);
                }
            }
            catch (Exception ex)
            {
                image.Dispose();
                image.Error = ex is NotSupportedException || ex is FileFormatException || ex is COMException
                    ? "Unsupported or damaged image." : ex.Message;
            }
            return image;
        }

        public void Dispose()
        {
            foreach (var m in _mips) m.Dispose();
            _mips.Clear();
        }

        static Bitmap DecodeGdi(byte[] data)
        {
            try
            {
                using (var ms = new MemoryStream(data))
                using (var img = Image.FromStream(ms, false, true))
                {
                    Bitmap bmp = ToPArgb(img);
                    if (Array.IndexOf(img.PropertyIdList, OrientationTag) >= 0)
                    {
                        byte[] v = img.GetPropertyItem(OrientationTag).Value;
                        if (v != null && v.Length >= 2) Orient(bmp, BitConverter.ToUInt16(v, 0));
                    }
                    return bmp;
                }
            }
            catch (ArgumentException) { return null; }      // GDI+ reports unknown formats this way
            catch (OutOfMemoryException) { return null; }   // ...or this way
            catch (ExternalException) { return null; }
        }

        // Windows Imaging Component handles formats GDI+ can't, e.g. WebP and HEIC (with codec installed).
        static Bitmap DecodeWic(byte[] data)
        {
            using (var ms = new MemoryStream(data))
            {
                var decoder = BitmapDecoder.Create(ms, BitmapCreateOptions.PreservePixelFormat | BitmapCreateOptions.IgnoreColorProfile, BitmapCacheOption.OnLoad);
                BitmapFrame frame = decoder.Frames[0];
                var src = new FormatConvertedBitmap(frame, System.Windows.Media.PixelFormats.Pbgra32, null, 0);
                int w = src.PixelWidth, h = src.PixelHeight;
                var bmp = new Bitmap(w, h, PixelFormat.Format32bppPArgb);
                var bits = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format32bppPArgb);
                try { src.CopyPixels(System.Windows.Int32Rect.Empty, bits.Scan0, bits.Stride * h, bits.Stride); }
                finally { bmp.UnlockBits(bits); }

                try
                {
                    var meta = frame.Metadata as BitmapMetadata;
                    object o = meta != null ? meta.GetQuery("System.Photo.Orientation") : null;
                    if (o != null) Orient(bmp, Convert.ToInt32(o));
                }
                catch (NotSupportedException) { }
                catch (InvalidOperationException) { }
                catch (ArgumentException) { }
                return bmp;
            }
        }

        static void Orient(Bitmap bmp, int exifOrientation)
        {
            switch (exifOrientation)
            {
                case 2: bmp.RotateFlip(RotateFlipType.RotateNoneFlipX); break;
                case 3: bmp.RotateFlip(RotateFlipType.Rotate180FlipNone); break;
                case 4: bmp.RotateFlip(RotateFlipType.Rotate180FlipX); break;
                case 5: bmp.RotateFlip(RotateFlipType.Rotate90FlipX); break;
                case 6: bmp.RotateFlip(RotateFlipType.Rotate90FlipNone); break;
                case 7: bmp.RotateFlip(RotateFlipType.Rotate270FlipX); break;
                case 8: bmp.RotateFlip(RotateFlipType.Rotate270FlipNone); break;
            }
        }

        // Premultiplied ARGB is the format GDI+ draws fastest.
        static Bitmap ToPArgb(Image img)
        {
            var bmp = new Bitmap(img.Width, img.Height, PixelFormat.Format32bppPArgb);
            using (var g = Graphics.FromImage(bmp))
            {
                g.CompositingMode = CompositingMode.SourceCopy;
                g.InterpolationMode = InterpolationMode.NearestNeighbor;
                g.PixelOffsetMode = PixelOffsetMode.Half;
                g.DrawImage(img, new Rectangle(0, 0, img.Width, img.Height), 0, 0, img.Width, img.Height, GraphicsUnit.Pixel);
            }
            return bmp;
        }

        static Bitmap Half(Bitmap src)
        {
            int w = Math.Max(1, src.Width / 2), h = Math.Max(1, src.Height / 2);
            var bmp = new Bitmap(w, h, PixelFormat.Format32bppPArgb);
            using (var g = Graphics.FromImage(bmp))
            using (var attrs = new ImageAttributes())
            {
                attrs.SetWrapMode(WrapMode.TileFlipXY);
                g.CompositingMode = CompositingMode.SourceCopy;
                // At exactly 2:1 with half-pixel offsets, bilinear samples the centre of each 2x2 block,
                // i.e. a box filter: the same quality as bicubic here at a fraction of the cost.
                g.InterpolationMode = InterpolationMode.Bilinear;
                g.PixelOffsetMode = PixelOffsetMode.Half;
                g.DrawImage(src, new Rectangle(0, 0, w, h), 0, 0, src.Width, src.Height, GraphicsUnit.Pixel, attrs);
            }
            return bmp;
        }
    }
}
