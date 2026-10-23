using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Windows.Forms;

namespace RulZ.ImageTournament
{
    enum ViewMode { SideBySide, Split }

    // Draws two images side by side, or overlapped with a draggable divider (split).
    // Zoom and pan are shared: each image is fitted to its pane, then both are zoomed by the
    // same factor around the same point, measured as a fraction of the image's width and
    // height. That keeps both images showing the same spot relative to their middle.
    sealed class CompareView : Control
    {
        const float MaxZoom = 400f;
        const int DividerGrab = 10;

        enum DragKind { None, Pan, Divider }

        readonly Timer _settle = new Timer { Interval = 200 };
        readonly ImageAttributes _edgeClamp = new ImageAttributes();
        readonly Font _bold = new Font("Segoe UI", 10f, FontStyle.Bold);
        readonly Font _small = new Font("Segoe UI", 9f);
        readonly Font _big = new Font("Segoe UI", 16f);
        readonly Font _banner = new Font("Segoe UI", 20f, FontStyle.Bold);

        LoadedImage _left, _right;
        string _message = "", _bannerText;
        bool _dropHint, _loading, _fast;
        ViewMode _mode;
        float _zoom = 1f, _split = 0.5f;
        PointF _center = new PointF(0.5f, 0.5f);

        DragKind _drag;
        Point _lastMouse;
        Rectangle _dragPane;
        Size _dragImageSize;

        public CompareView()
        {
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer
                     | ControlStyles.ResizeRedraw | ControlStyles.Selectable, true);
            TabStop = true;
            _edgeClamp.SetWrapMode(WrapMode.TileFlipXY);
            // Draw with a fast filter while zooming/panning, then redraw in high quality once it settles.
            _settle.Tick += delegate { _settle.Stop(); _fast = false; Invalidate(); };
        }

        // Shown above the welcome message.
        public Image Logo { get; set; }

        public ViewMode Mode
        {
            get { return _mode; }
            set { _mode = value; Invalidate(); }
        }

        public bool DropHint
        {
            get { return _dropHint; }
            set { if (_dropHint != value) { _dropHint = value; Invalidate(); } }
        }

        public bool Loading
        {
            get { return _loading; }
            set { if (_loading != value) { _loading = value; Invalidate(); } }
        }

        public void ShowMessage(string message)
        {
            _left = _right = null;
            _bannerText = null;
            _message = message;
            _loading = false;
            Invalidate();
        }

        public void ShowPair(LoadedImage left, LoadedImage right, bool resetView)
        {
            _left = left;
            _right = right;
            _bannerText = null;
            _loading = false;
            if (resetView) ResetView();
            Invalidate();
        }

        public void ShowSingle(LoadedImage image, string banner)
        {
            _left = image;
            _right = null;
            _bannerText = banner;
            _loading = false;
            ResetView();
            Invalidate();
        }

        public void ResetView()
        {
            _zoom = 1f;
            _center = new PointF(0.5f, 0.5f);
            Invalidate();
        }

        public void ZoomBy(float factor)
        {
            ZoomAt(new Point(ClientSize.Width / 2, ClientSize.Height / 2), factor);
        }

        bool IsPair { get { return _right != null; } }
        int DividerX { get { return (int)Math.Round(_split * ClientSize.Width); } }
        float Dpi { get { return DeviceDpi / 96f; } }

        void GetPanes(out Rectangle a, out Rectangle b)
        {
            Rectangle r = ClientRectangle;
            if (!IsPair || _mode == ViewMode.Split) { a = b = r; return; }
            int gap = (int)(6 * Dpi), half = (r.Width - gap) / 2;
            a = new Rectangle(r.X, r.Y, half, r.Height);
            b = new Rectangle(r.X + half + gap, r.Y, r.Width - half - gap, r.Height);
        }

        static float FitScale(Size image, Rectangle pane)
        {
            return Math.Min((float)pane.Width / image.Width, (float)pane.Height / image.Height);
        }

        // Where the whole image lands on screen for the current zoom and pan.
        RectangleF Place(Size image, Rectangle pane)
        {
            float s = FitScale(image, pane) * _zoom;
            float w = image.Width * s, h = image.Height * s;
            return new RectangleF(pane.X + pane.Width / 2f - _center.X * w, pane.Y + pane.Height / 2f - _center.Y * h, w, h);
        }

        // The pane under a point and an image whose geometry drives zoom/pan there.
        bool HitPane(Point p, out Rectangle pane, out Size imageSize)
        {
            Rectangle a, b;
            GetPanes(out a, out b);
            bool right = IsPair && (_mode == ViewMode.Split ? p.X >= DividerX : p.X >= b.X);
            pane = right ? b : a;
            LoadedImage img = right ? _right : _left;
            if (img == null || img.Bitmap == null) img = right ? _left : _right;
            imageSize = img != null && img.Bitmap != null ? img.Bitmap.Size : Size.Empty;
            return !imageSize.IsEmpty && pane.Width > 0 && pane.Height > 0;
        }

        void ZoomAt(Point p, float factor)
        {
            Rectangle pane;
            Size size;
            if (!HitPane(p, out pane, out size)) return;
            float zoom = Math.Max(1f, Math.Min(MaxZoom, _zoom * factor));
            if (zoom == _zoom) return;

            // Keep the image point under the cursor fixed while the scale changes.
            RectangleF r = Place(size, pane);
            float ux = (p.X - r.X) / r.Width, uy = (p.Y - r.Y) / r.Height;
            float w = r.Width * zoom / _zoom, h = r.Height * zoom / _zoom;
            _zoom = zoom;
            _center = zoom <= 1f
                ? new PointF(0.5f, 0.5f)
                : new PointF((pane.X + pane.Width / 2f - (p.X - ux * w)) / w, (pane.Y + pane.Height / 2f - (p.Y - uy * h)) / h);
            ClampCenter();
            Interacted();
        }

        void ClampCenter()
        {
            _center = new PointF(Math.Max(0f, Math.Min(1f, _center.X)), Math.Max(0f, Math.Min(1f, _center.Y)));
        }

        void Interacted()
        {
            _fast = true;
            _settle.Stop();
            _settle.Start();
            Invalidate();
        }

        bool NearDivider(Point p)
        {
            return IsPair && _mode == ViewMode.Split && Math.Abs(p.X - DividerX) <= DividerGrab * Dpi;
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            base.OnMouseDown(e);
            Focus();
            if (e.Button != MouseButtons.Left) return;
            if (NearDivider(e.Location)) _drag = DragKind.Divider;
            else if (HitPane(e.Location, out _dragPane, out _dragImageSize)) _drag = DragKind.Pan;
            _lastMouse = e.Location;
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (_drag == DragKind.Divider)
            {
                _split = Math.Max(0f, Math.Min(1f, (float)e.X / Math.Max(1, ClientSize.Width)));
                Invalidate();
            }
            else if (_drag == DragKind.Pan)
            {
                RectangleF r = Place(_dragImageSize, _dragPane);
                _center = new PointF(_center.X - (e.X - _lastMouse.X) / r.Width, _center.Y - (e.Y - _lastMouse.Y) / r.Height);
                ClampCenter();
                Interacted();
            }
            else
            {
                Cursor = NearDivider(e.Location) ? Cursors.VSplit : _left != null ? Cursors.Hand : Cursors.Default;
            }
            _lastMouse = e.Location;
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            base.OnMouseUp(e);
            _drag = DragKind.None;
        }

        protected override void OnMouseDoubleClick(MouseEventArgs e)
        {
            base.OnMouseDoubleClick(e);
            if (e.Button == MouseButtons.Left && !NearDivider(e.Location)) ResetView();
        }

        protected override void OnMouseWheel(MouseEventArgs e)
        {
            base.OnMouseWheel(e);
            ZoomAt(e.Location, (float)Math.Pow(1.2, e.Delta / 120.0));
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.Clear(BackColor);

            if (_left == null)
                DrawMessage(g);
            else if (!IsPair)
            {
                DrawImage(g, _left, ClientRectangle, ClientRectangle);
                DrawLabel(g, _left, ClientRectangle, null, false);
            }
            else
            {
                Rectangle a, b;
                GetPanes(out a, out b);
                if (_mode == ViewMode.SideBySide)
                {
                    DrawImage(g, _left, a, a);
                    DrawImage(g, _right, b, b);
                    DrawLabel(g, _left, a, "LEFT", false);
                    DrawLabel(g, _right, b, "RIGHT", false);
                }
                else
                {
                    int x = DividerX;
                    DrawImage(g, _left, a, new Rectangle(a.X, a.Y, x - a.X, a.Height));
                    DrawImage(g, _right, b, new Rectangle(x, b.Y, b.Right - x, b.Height));
                    DrawDivider(g, x);
                    DrawLabel(g, _left, a, "LEFT", false);
                    DrawLabel(g, _right, b, "RIGHT", true);
                }
            }

            if (_bannerText != null) DrawBanner(g);
            if (_loading) DrawPill(g, "Loading…", _bold, Theme.Text, ClientSize.Height / 2);
            if (_dropHint) DrawDropHint(g);
        }

        void DrawImage(Graphics g, LoadedImage image, Rectangle pane, Rectangle clip)
        {
            if (image == null || clip.Width <= 0 || clip.Height <= 0 || pane.Width <= 0 || pane.Height <= 0) return;
            if (image.Bitmap == null)
            {
                TextRenderer.DrawText(g, "Couldn't open this image\n\n" + image.FileName + "\n" + image.Error, _bold, clip,
                    Color.Salmon, TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.WordBreak);
                return;
            }

            RectangleF dest = Place(image.Bitmap.Size, pane);
            RectangleF visible = RectangleF.Intersect(dest, clip);
            if (visible.Width <= 0 || visible.Height <= 0) return;

            // Use the smallest pre-shrunk copy that is still at least as big as it appears on screen.
            Bitmap src = image.Bitmap;
            for (int i = 1; i < image.Mips.Count && dest.Width <= image.Mips[i].Width; i++) src = image.Mips[i];

            // Only draw the visible part of the source; this is what keeps deep zoom fast.
            float sx = dest.Width / src.Width, sy = dest.Height / src.Height;
            int x0 = Math.Max(0, (int)Math.Floor((visible.Left - dest.X) / sx) - 1);
            int y0 = Math.Max(0, (int)Math.Floor((visible.Top - dest.Y) / sy) - 1);
            int x1 = Math.Min(src.Width, (int)Math.Ceiling((visible.Right - dest.X) / sx) + 1);
            int y1 = Math.Min(src.Height, (int)Math.Ceiling((visible.Bottom - dest.Y) / sy) + 1);
            var corners = new[]
            {
                new PointF(dest.X + x0 * sx, dest.Y + y0 * sy),
                new PointF(dest.X + x1 * sx, dest.Y + y0 * sy),
                new PointF(dest.X + x0 * sx, dest.Y + y1 * sy),
            };

            float pixelScale = dest.Width / image.Bitmap.Width;
            g.SetClip(clip);
            // Show crisp, true pixels once zoomed well past 100%.
            g.InterpolationMode = pixelScale >= 3f ? InterpolationMode.NearestNeighbor
                : _fast ? InterpolationMode.Bilinear : InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.Half;
            g.DrawImage(src, corners, new RectangleF(x0, y0, x1 - x0, y1 - y0), GraphicsUnit.Pixel, _edgeClamp);
            g.ResetClip();
        }

        void DrawLabel(Graphics g, LoadedImage image, Rectangle pane, string side, bool alignRight)
        {
            if (pane.Width < 60) return;
            string title = side == null ? image.FileName : side + "  ·  " + image.FileName;
            string info = image.Bitmap == null ? "" : string.Format("{0} × {1}   ·   {2}   ·   {3:0}%",
                image.Bitmap.Width, image.Bitmap.Height, FormatBytes(image.FileSize),
                100 * FitScale(image.Bitmap.Size, pane) * _zoom);

            var flags = TextFormatFlags.NoPrefix | TextFormatFlags.SingleLine | TextFormatFlags.NoPadding;
            Size s1 = TextRenderer.MeasureText(g, title, _bold, Size.Empty, flags);
            Size s2 = TextRenderer.MeasureText(g, info, _small, Size.Empty, flags);
            flags |= TextFormatFlags.EndEllipsis;
            int pad = (int)(8 * Dpi), margin = (int)(10 * Dpi);
            int w = Math.Min(Math.Max(s1.Width, s2.Width) + 2 * pad, (IsPair && _mode == ViewMode.Split ? pane.Width / 2 : pane.Width) - 2 * margin);
            int h = s1.Height + s2.Height + 2 * pad;
            int x = alignRight ? pane.Right - margin - w : pane.X + margin;
            var box = new Rectangle(x, pane.Y + margin, w, h);

            using (var bg = new SolidBrush(Color.FromArgb(190, 16, 16, 18))) g.FillRectangle(bg, box);
            if (alignRight) flags |= TextFormatFlags.Right;
            TextRenderer.DrawText(g, title, _bold, new Rectangle(x + pad, box.Y + pad, w - 2 * pad, s1.Height), side == null ? Theme.Text : Theme.AccentText, flags);
            TextRenderer.DrawText(g, info, _small, new Rectangle(x + pad, box.Y + pad + s1.Height, w - 2 * pad, s2.Height), Theme.Dim, flags);
        }

        void DrawDivider(Graphics g, int x)
        {
            int h = ClientSize.Height, cy = h / 2, r = (int)(16 * Dpi), t = (int)(6 * Dpi);
            using (var shadow = new Pen(Color.FromArgb(140, 0, 0, 0), 4 * Dpi)) g.DrawLine(shadow, x, 0, x, h);
            using (var line = new Pen(Theme.Accent, 2 * Dpi)) g.DrawLine(line, x, 0, x, h);
            g.SmoothingMode = SmoothingMode.AntiAlias;
            using (var fill = new SolidBrush(Theme.Accent)) g.FillEllipse(fill, x - r, cy - r, 2 * r, 2 * r);
            using (var arrow = new SolidBrush(Theme.Text))
            {
                int gap = (int)(2 * Dpi);
                g.FillPolygon(arrow, new[] { new Point(x - gap, cy - t), new Point(x - gap, cy + t), new Point(x - gap - t, cy) });
                g.FillPolygon(arrow, new[] { new Point(x + gap, cy - t), new Point(x + gap, cy + t), new Point(x + gap + t, cy) });
            }
            g.SmoothingMode = SmoothingMode.None;
        }

        void DrawMessage(Graphics g)
        {
            Rectangle r = ClientRectangle;
            int w = Math.Min(r.Width - 40, (int)(620 * Dpi)), h = Math.Min(r.Height - 40, (int)(260 * Dpi));
            if (w <= 0 || h <= 0) return;

            // The logo sits above the drop box when there's room for both.
            int logoW = 0, logoH = 0, gap = (int)(36 * Dpi);
            if (Logo != null)
            {
                logoW = Math.Min(w * 7 / 10, (int)(440 * Dpi));
                logoH = logoW * Logo.Height / Logo.Width;
                if (logoH + gap + h > r.Height - 40) logoH = 0;
            }
            int top = r.Y + (r.Height - h - (logoH > 0 ? logoH + gap : 0)) / 2;
            if (logoH > 0)
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.DrawImage(Logo, new Rectangle(r.X + (r.Width - logoW) / 2, top, logoW, logoH));
                top += logoH + gap;
            }

            var box = new Rectangle(r.X + (r.Width - w) / 2, top, w, h);
            using (var pen = new Pen(Theme.Border, 2 * Dpi) { DashStyle = DashStyle.Dash }) g.DrawRectangle(pen, box);
            TextRenderer.DrawText(g, _message, _big, box, Theme.Dim,
                TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.WordBreak | TextFormatFlags.NoPrefix);
        }

        void DrawBanner(Graphics g)
        {
            DrawPill(g, _bannerText, _banner, Theme.AccentText, ClientSize.Height - (int)(60 * Dpi));
        }

        void DrawPill(Graphics g, string text, Font font, Color color, int centerY)
        {
            Size s = TextRenderer.MeasureText(g, text, font);
            int padX = (int)(24 * Dpi), padY = (int)(10 * Dpi);
            var box = new Rectangle((ClientSize.Width - s.Width) / 2 - padX, centerY - s.Height / 2 - padY, s.Width + 2 * padX, s.Height + 2 * padY);
            using (var bg = new SolidBrush(Color.FromArgb(210, 16, 16, 18))) g.FillRectangle(bg, box);
            TextRenderer.DrawText(g, text, font, box, color, TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.NoPrefix);
        }

        void DrawDropHint(Graphics g)
        {
            Rectangle r = ClientRectangle;
            using (var bg = new SolidBrush(Color.FromArgb(150, 0, 0, 0))) g.FillRectangle(bg, r);
            int m = (int)(16 * Dpi);
            using (var pen = new Pen(Theme.Accent, 3 * Dpi) { DashStyle = DashStyle.Dash })
                g.DrawRectangle(pen, r.X + m, r.Y + m, r.Width - 2 * m, r.Height - 2 * m);
            TextRenderer.DrawText(g, "Drop to add images", _banner, r, Theme.AccentText,
                TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter);
        }

        static string FormatBytes(long bytes)
        {
            if (bytes >= 1L << 20) return (bytes / (double)(1L << 20)).ToString("0.0") + " MB";
            if (bytes >= 1L << 10) return (bytes / (double)(1L << 10)).ToString("0") + " KB";
            return bytes + " B";
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _settle.Dispose();
                _edgeClamp.Dispose();
                _bold.Dispose();
                _small.Dispose();
                _big.Dispose();
                _banner.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
