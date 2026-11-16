using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace RulZ.ImageTournament
{
    // Owner-drawn buttons and check boxes in the RulZ look: rounded corners, a solid accent for the selected and
    // primary states, and a 2px accent ring for keyboard focus instead of the dotted rectangle. They keep the
    // standard Button/CheckBox behaviour (clicks, Checked, Enabled, access keys); only the painting is theirs.
    // Colors come from the same properties the stock controls use (BackColor, ForeColor, FlatAppearance), which
    // MainForm.ApplyTheme sets, plus HoverForeColor and HoverBorderColor below.
    static class RoundPaint
    {
        public static GraphicsPath Path(RectangleF r, float radius)
        {
            var p = new GraphicsPath();
            float d = Math.Min(radius * 2, Math.Min(r.Width, r.Height));
            if (d <= 0.5f) { p.AddRectangle(r); return p; }
            p.AddArc(r.X, r.Y, d, d, 180, 90);
            p.AddArc(r.Right - d, r.Y, d, d, 270, 90);
            p.AddArc(r.Right - d, r.Bottom - d, d, d, 0, 90);
            p.AddArc(r.X, r.Bottom - d, d, d, 90, 90);
            p.CloseFigure();
            return p;
        }

        // Fill + 1px outline of a rounded rectangle filling the client area.
        public static void Surface(Graphics g, Size size, float k, int radius, Color fill, Color border)
        {
            g.SmoothingMode = SmoothingMode.AntiAlias;
            var r = new RectangleF(0.5f, 0.5f, size.Width - 1f, size.Height - 1f);
            using (var path = Path(r, radius * k))
            {
                using (var b = new SolidBrush(fill)) g.FillPath(b, path);
                using (var pen = new Pen(border, Math.Max(1f, k))) g.DrawPath(pen, path);
            }
        }

        // The keyboard focus ring: 2px, drawn just inside the edge.
        public static void Ring(Graphics g, Size size, float k, int radius, Color color)
        {
            g.SmoothingMode = SmoothingMode.AntiAlias;
            float w = 2f * k;
            var r = new RectangleF(w / 2, w / 2, size.Width - w, size.Height - w);
            using (var path = Path(r, radius * k))
            using (var pen = new Pen(color, w)) g.DrawPath(pen, path);
        }

        public static Color Blend(Color top, Color under, double amount)
        {
            return Color.FromArgb(
                (int)(top.R * amount + under.R * (1 - amount)),
                (int)(top.G * amount + under.G * (1 - amount)),
                (int)(top.B * amount + under.B * (1 - amount)));
        }

        public static bool IsAccentFill(Color c)
        {
            return c.ToArgb() == Theme.Accent.ToArgb() || c.ToArgb() == Theme.AccentFill.ToArgb()
                || c.ToArgb() == Theme.AccentFillHover.ToArgb() || c.ToArgb() == Theme.AccentPressed.ToArgb();
        }

        public const TextFormatFlags Centered = TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter
            | TextFormatFlags.NoPrefix | TextFormatFlags.EndEllipsis | TextFormatFlags.SingleLine;
    }

    sealed class RoundButton : Button
    {
        bool _hover, _down;

        /// <summary>Radius in 96-DPI pixels; 8 for buttons, 10 for the big Keep buttons.</summary>
        public int Radius = 8;
        /// <summary>Text color while the mouse is over the button; Empty keeps ForeColor.</summary>
        public Color HoverForeColor = Color.Empty;
        /// <summary>Outline color while the mouse is over the button; Empty keeps the normal outline.</summary>
        public Color HoverBorderColor = Color.Empty;

        /// <summary>Side padding in 96-DPI pixels when the button sizes itself to its text (small button: 12).</summary>
        public int SidePadding = 12;

        // Text width plus 12px each side, at least 32px high: the spec's small button.
        public override Size GetPreferredSize(Size proposedSize)
        {
            float k = DeviceDpi / 96f;
            Size t = TextRenderer.MeasureText(Text, Font, new Size(int.MaxValue, int.MaxValue), TextFormatFlags.NoPrefix | TextFormatFlags.SingleLine);
            return new Size(t.Width + (int)Math.Ceiling(2 * SidePadding * k), Math.Max(t.Height + (int)(8 * k), (int)Math.Ceiling(32 * k)));
        }

        public RoundButton()
        {
            FlatStyle = FlatStyle.Flat;
            UseVisualStyleBackColor = false;
            SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer
                     | ControlStyles.ResizeRedraw, true);
        }

        protected override void OnMouseEnter(EventArgs e) { base.OnMouseEnter(e); _hover = true; Invalidate(); }
        protected override void OnMouseLeave(EventArgs e) { base.OnMouseLeave(e); _hover = false; _down = false; Invalidate(); }
        protected override void OnMouseDown(MouseEventArgs e) { base.OnMouseDown(e); if (e.Button == MouseButtons.Left) { _down = true; Invalidate(); } }
        protected override void OnMouseUp(MouseEventArgs e) { base.OnMouseUp(e); _down = false; Invalidate(); }
        protected override void OnGotFocus(EventArgs e) { base.OnGotFocus(e); Invalidate(); }
        protected override void OnLostFocus(EventArgs e) { base.OnLostFocus(e); Invalidate(); }
        protected override void OnEnabledChanged(EventArgs e) { base.OnEnabledChanged(e); Invalidate(); }

        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;
            float k = DeviceDpi / 96f;
            Color parent = Parent != null ? Parent.BackColor : Theme.Panel;
            g.Clear(parent);

            bool on = Enabled;
            Color fill = BackColor, border = FlatAppearance.BorderColor, fore = ForeColor;
            if (on && _down) fill = FlatAppearance.MouseDownBackColor;
            else if (on && _hover)
            {
                fill = FlatAppearance.MouseOverBackColor;
                if (HoverBorderColor != Color.Empty) border = HoverBorderColor;
                if (HoverForeColor != Color.Empty) fore = HoverForeColor;
            }
            if (RoundPaint.IsAccentFill(fill)) border = fill;
            if (!on)
            {
                fill = RoundPaint.Blend(fill, parent, 0.45);
                border = RoundPaint.Blend(border, parent, 0.45);
                fore = RoundPaint.Blend(fore, parent, 0.45);
            }

            RoundPaint.Surface(g, ClientSize, k, Radius, fill, border);
            if (Focused && ShowFocusCues)
                RoundPaint.Ring(g, ClientSize, k, Radius, RoundPaint.IsAccentFill(fill) ? Theme.OnAccent : Theme.Accent);
            TextRenderer.DrawText(g, Text, Font, Rectangle.Inflate(ClientRectangle, -(int)(6 * k), 0), fore, RoundPaint.Centered);
        }
    }

    // A two-state button (the Side by Side / Split switch). Checked fills solid accent with white text.
    sealed class RoundToggle : CheckBox
    {
        bool _hover, _down;

        public int Radius = 8;
        public Color HoverBorderColor = Color.Empty;

        public RoundToggle()
        {
            Appearance = Appearance.Button;
            FlatStyle = FlatStyle.Flat;
            UseVisualStyleBackColor = false;
            SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer
                     | ControlStyles.ResizeRedraw, true);
        }

        protected override void OnMouseEnter(EventArgs e) { base.OnMouseEnter(e); _hover = true; Invalidate(); }
        protected override void OnMouseLeave(EventArgs e) { base.OnMouseLeave(e); _hover = false; _down = false; Invalidate(); }
        protected override void OnMouseDown(MouseEventArgs e) { base.OnMouseDown(e); if (e.Button == MouseButtons.Left) { _down = true; Invalidate(); } }
        protected override void OnMouseUp(MouseEventArgs e) { base.OnMouseUp(e); _down = false; Invalidate(); }
        protected override void OnGotFocus(EventArgs e) { base.OnGotFocus(e); Invalidate(); }
        protected override void OnLostFocus(EventArgs e) { base.OnLostFocus(e); Invalidate(); }
        protected override void OnEnabledChanged(EventArgs e) { base.OnEnabledChanged(e); Invalidate(); }
        protected override void OnCheckedChanged(EventArgs e) { base.OnCheckedChanged(e); Invalidate(); }

        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;
            float k = DeviceDpi / 96f;
            Color parent = Parent != null ? Parent.BackColor : Theme.Panel;
            g.Clear(parent);

            bool on = Enabled;
            Color fill = Checked ? FlatAppearance.CheckedBackColor : BackColor;
            Color border = Checked ? fill : FlatAppearance.BorderColor;
            Color fore = Checked ? Theme.OnAccent : ForeColor;
            if (on && _down) fill = Checked ? Theme.AccentPressed : FlatAppearance.MouseDownBackColor;
            else if (on && _hover)
            {
                if (Checked) { fill = Theme.AccentFillHover; border = fill; }
                else { fill = FlatAppearance.MouseOverBackColor; if (HoverBorderColor != Color.Empty) border = HoverBorderColor; }
            }
            if (!on)
            {
                fill = RoundPaint.Blend(fill, parent, 0.45);
                border = RoundPaint.Blend(border, parent, 0.45);
                fore = RoundPaint.Blend(fore, parent, 0.45);
            }

            RoundPaint.Surface(g, ClientSize, k, Radius, fill, border);
            if (Focused && ShowFocusCues)
                RoundPaint.Ring(g, ClientSize, k, Radius, Checked ? Theme.OnAccent : Theme.Accent);
            TextRenderer.DrawText(g, Text, Font, Rectangle.Inflate(ClientRectangle, -(int)(4 * k), 0), fore, RoundPaint.Centered);
        }
    }

    // A check box: 18px square, 4px radius, border-strong outline on the raised fill; checked = accent with a white tick.
    sealed class ThemedCheck : CheckBox
    {
        bool _hover;

        public ThemedCheck()
        {
            SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer
                     | ControlStyles.ResizeRedraw, true);
        }

        const int Box = 18, Lead = 4, Gap = 8;

        public override Size GetPreferredSize(Size proposedSize)
        {
            float k = DeviceDpi / 96f;
            Size t = string.IsNullOrEmpty(Text) ? Size.Empty
                : TextRenderer.MeasureText(Text, Font, new Size(int.MaxValue, int.MaxValue), TextFormatFlags.NoPrefix | TextFormatFlags.SingleLine);
            int w = (int)Math.Ceiling((Lead + Box + Gap) * k) + t.Width + (int)(2 * k);
            int h = Math.Max(t.Height + (int)(2 * k), (int)Math.Ceiling((Box + 6) * k));
            return new Size(w, h);
        }

        protected override void OnMouseEnter(EventArgs e) { base.OnMouseEnter(e); _hover = true; Invalidate(); }
        protected override void OnMouseLeave(EventArgs e) { base.OnMouseLeave(e); _hover = false; Invalidate(); }
        protected override void OnGotFocus(EventArgs e) { base.OnGotFocus(e); Invalidate(); }
        protected override void OnLostFocus(EventArgs e) { base.OnLostFocus(e); Invalidate(); }
        protected override void OnEnabledChanged(EventArgs e) { base.OnEnabledChanged(e); Invalidate(); }
        protected override void OnCheckedChanged(EventArgs e) { base.OnCheckedChanged(e); Invalidate(); }

        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;
            float k = DeviceDpi / 96f;
            g.Clear(BackColor);

            bool on = Enabled;
            Color fill = Checked ? Theme.Accent : Theme.Control;
            Color border = Checked ? Theme.Accent : (_hover ? Theme.Dim : Theme.BorderStrong);
            Color fore = ForeColor;
            if (!on)
            {
                fill = RoundPaint.Blend(fill, BackColor, 0.45);
                border = RoundPaint.Blend(border, BackColor, 0.45);
                fore = RoundPaint.Blend(fore, BackColor, 0.45);
            }

            int box = (int)Math.Round(Box * k), x = (int)Math.Round(Lead * k), y = (ClientSize.Height - box) / 2;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            var r = new RectangleF(x + 0.5f, y + 0.5f, box - 1f, box - 1f);
            using (var path = RoundPaint.Path(r, 4 * k))
            {
                using (var b = new SolidBrush(fill)) g.FillPath(b, path);
                using (var pen = new Pen(border, Math.Max(1f, k))) g.DrawPath(pen, path);
            }
            if (Checked)
            {
                using (var pen = new Pen(Color.White, 2f * k) { StartCap = LineCap.Round, EndCap = LineCap.Round, LineJoin = LineJoin.Round })
                    g.DrawLines(pen, new[]
                    {
                        new PointF(x + box * 0.25f, y + box * 0.52f),
                        new PointF(x + box * 0.43f, y + box * 0.70f),
                        new PointF(x + box * 0.76f, y + box * 0.30f),
                    });
            }
            if (Focused && ShowFocusCues)
            {
                float w = 2f * k, off = 2f * k;
                var ring = new RectangleF(x - off - w / 2 + 0.5f, y - off - w / 2 + 0.5f, box + 2 * off + w - 1f, box + 2 * off + w - 1f);
                using (var path = RoundPaint.Path(ring, 6 * k))
                using (var pen = new Pen(Theme.Accent, w)) g.DrawPath(pen, path);
            }
            var textRect = new Rectangle(x + box + (int)(Gap * k), 0, Math.Max(0, ClientSize.Width - x - box - (int)(Gap * k)), ClientSize.Height);
            TextRenderer.DrawText(g, Text, Font, textRect, fore,
                TextFormatFlags.Left | TextFormatFlags.VerticalCenter | TextFormatFlags.NoPrefix | TextFormatFlags.SingleLine);
        }
    }
}
