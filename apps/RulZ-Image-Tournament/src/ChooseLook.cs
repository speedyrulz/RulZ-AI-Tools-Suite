using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace RulZ.ImageTournament
{
    // The first-run "Choose your look" prompt: two cards, Dark (preselected) and Light, and Continue.
    // The dialog itself previews the selected look. Closing it without pressing Continue means Dark.
    sealed class ChooseLookDialog : Form
    {
        readonly LookCard _dark, _light;
        readonly Label _title, _hint;
        readonly Button _continue;
        Palette _selected = Palette.Dark;

        // Returns the chosen palette (Dark unless Light was picked and Continue pressed).
        public static Palette Ask()
        {
            using (var d = new ChooseLookDialog())
                return d.ShowDialog() == DialogResult.OK ? d._selected : Palette.Dark;
        }

        ChooseLookDialog()
        {
            SuspendLayout();
            AutoScaleDimensions = new SizeF(96f, 96f);
            AutoScaleMode = AutoScaleMode.Dpi;
            Text = "RulZ Image Tournament";
            Font = new Font("Segoe UI", 9.5f);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = MinimizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            ShowInTaskbar = true;
            ClientSize = new Size(520, 380);
            KeyPreview = true;
            try { Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath); }
            catch (ArgumentException) { }

            _title = new Label
            {
                Text = "Choose your look",
                Font = new Font("Segoe UI", 16f, FontStyle.Bold),
                AutoSize = false,
                Bounds = new Rectangle(28, 20, 464, 38),
                UseMnemonic = false,
            };
            _dark = new LookCard(Palette.Dark, "Dark") { Bounds = new Rectangle(28, 70, 224, 196) };
            _light = new LookCard(Palette.Light, "Light") { Bounds = new Rectangle(268, 70, 224, 196) };
            _dark.Click += delegate { Pick(Palette.Dark); };
            _light.Click += delegate { Pick(Palette.Light); };
            _dark.DoubleClick += delegate { Pick(Palette.Dark); Finish(); };
            _light.DoubleClick += delegate { Pick(Palette.Light); Finish(); };

            _hint = new Label
            {
                Text = "You can change this later with the \"Light mode\" checkbox at the right end of the toolbar.",
                AutoSize = false,
                Bounds = new Rectangle(28, 278, 464, 40),
                UseMnemonic = false,
            };
            _continue = new Button
            {
                Text = "Continue",
                FlatStyle = FlatStyle.Flat,
                UseVisualStyleBackColor = false,
                Font = new Font("Segoe UI", 10f, FontStyle.Bold),
                Bounds = new Rectangle(372, 326, 120, 36),
            };
            _continue.FlatAppearance.BorderSize = 0;
            _continue.Click += delegate { Finish(); };
            AcceptButton = _continue;

            Controls.AddRange(new Control[] { _title, _dark, _light, _hint, _continue });
            ResumeLayout(false);
            Pick(Palette.Dark);
        }

        void Pick(Palette p)
        {
            _selected = p;
            _dark.Selected = p == Palette.Dark;
            _light.Selected = p == Palette.Light;
            BackColor = p.Back;
            ForeColor = p.Text;
            _title.ForeColor = p.Text;
            _hint.ForeColor = p.Dim;
            _dark.BackColor = _light.BackColor = p.Back;
            _dark.Chrome = _light.Chrome = p;
            _continue.BackColor = p.Accent;
            _continue.ForeColor = Color.White;
            _continue.FlatAppearance.MouseOverBackColor = Color.FromArgb(200, 0, 11);
            _continue.FlatAppearance.MouseDownBackColor = Color.FromArgb(170, 0, 9);
            if (IsHandleCreated) NativeMethods.UseDarkTitleBar(Handle, !p.IsLight);
            Invalidate(true);
        }

        void Finish()
        {
            DialogResult = DialogResult.OK;
            Close();
        }

        protected override void OnHandleCreated(EventArgs e)
        {
            base.OnHandleCreated(e);
            NativeMethods.UseDarkTitleBar(Handle, !_selected.IsLight);
        }

        protected override void OnShown(EventArgs e)
        {
            base.OnShown(e);
            Activate();
            _continue.Focus();
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            switch (keyData)
            {
                case Keys.Left: Pick(Palette.Dark); return true;
                case Keys.Right: Pick(Palette.Light); return true;
                case Keys.Escape: DialogResult = DialogResult.Cancel; Close(); return true;
            }
            return base.ProcessCmdKey(ref msg, keyData);
        }

        // A selectable card with a tiny drawing of the app in its palette.
        sealed class LookCard : Control
        {
            readonly Palette _look;
            readonly string _name;
            bool _selected, _hover;

            public LookCard(Palette look, string name)
            {
                _look = look;
                _name = name;
                Chrome = Palette.Dark;
                Cursor = Cursors.Hand;
                SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer
                         | ControlStyles.ResizeRedraw, true);
            }

            // The palette of the dialog around the card (for its frame and caption).
            public Palette Chrome { get; set; }

            public bool Selected
            {
                get { return _selected; }
                set { _selected = value; Invalidate(); }
            }

            protected override void OnMouseEnter(EventArgs e) { base.OnMouseEnter(e); _hover = true; Invalidate(); }
            protected override void OnMouseLeave(EventArgs e) { base.OnMouseLeave(e); _hover = false; Invalidate(); }

            protected override void OnPaint(PaintEventArgs e)
            {
                Graphics g = e.Graphics;
                float k = DeviceDpi / 96f;
                int W = ClientSize.Width, H = ClientSize.Height;
                g.Clear(Chrome.Panel);

                // The frame: red and thick when selected.
                int bw = (int)Math.Round((_selected ? 3 : 1) * k);
                Color frame = _selected ? Chrome.Accent : _hover ? Chrome.Dim : Chrome.Border;
                using (var pen = new Pen(frame, bw) { Alignment = PenAlignment.Inset })
                    g.DrawRectangle(pen, 0, 0, W - 1, H - 1);

                // The preview: a little window with a toolbar, two images and the red Keep buttons.
                int m = (int)(14 * k);
                var win = new Rectangle(m, m, W - 2 * m, (int)(120 * k));
                using (var b = new SolidBrush(_look.Back)) g.FillRectangle(b, win);
                using (var pen = new Pen(_look.Border)) g.DrawRectangle(pen, win.X, win.Y, win.Width - 1, win.Height - 1);
                int bar = (int)(18 * k);
                using (var b = new SolidBrush(_look.Panel))
                {
                    g.FillRectangle(b, win.X + 1, win.Y + 1, win.Width - 2, bar);
                    g.FillRectangle(b, win.X + 1, win.Bottom - bar - 1, win.Width - 2, bar);
                }
                using (var b = new SolidBrush(_look.Accent)) g.FillRectangle(b, win.X + (int)(6 * k), win.Y + (int)(6 * k), (int)(10 * k), (int)(6 * k));
                using (var b = new SolidBrush(_look.Dim))
                    for (int i = 0; i < 3; i++)
                        g.FillRectangle(b, win.X + (int)((22 + 18 * i) * k), win.Y + (int)(7 * k), (int)(14 * k), (int)(4 * k));

                int gap = (int)(6 * k), top = win.Y + bar + gap, ph = win.Height - 2 * bar - 2 * gap - 2;
                int pw = (win.Width - 3 * gap) / 2;
                var left = new Rectangle(win.X + gap, top, pw, ph);
                var right = new Rectangle(left.Right + gap, top, pw, ph);
                using (var sky = new LinearGradientBrush(left, Color.FromArgb(255, 170, 90), Color.FromArgb(90, 60, 140), 90f))
                    g.FillRectangle(sky, left);
                using (var sky = new LinearGradientBrush(right, Color.FromArgb(120, 190, 230), Color.FromArgb(30, 80, 120), 90f))
                    g.FillRectangle(sky, right);

                int kh = bar - (int)(8 * k), ky = win.Bottom - bar + (int)(3 * k);
                using (var b = new SolidBrush(_look.AccentFill))
                {
                    g.FillRectangle(b, win.X + gap, ky, pw * 2 / 3, kh);
                    g.FillRectangle(b, right.Right - pw * 2 / 3, ky, pw * 2 / 3, kh);
                }

                // The caption.
                using (var f = new Font("Segoe UI", 12f, FontStyle.Bold))
                {
                    var cap = new Rectangle(0, win.Bottom + (int)(8 * k), W, H - win.Bottom - (int)(12 * k));
                    string text = _name + (_look == Palette.Dark ? "  (default)" : "");
                    TextRenderer.DrawText(g, text, f, cap, Chrome.Text,
                        TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.NoPrefix);
                }
            }
        }
    }
}
