using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Timer = System.Windows.Forms.Timer;

namespace RulZ.ImageTournament
{
    sealed class MainForm : Form
    {
        enum LoserAction { Move, RecycleBin, Delete, Keep }

        static readonly string[] LoserActionNames = { "Move to Losers folder", "Send to Recycle Bin", "Delete permanently", "Leave in place" };

        const string AppName = "RulZ Image Tournament";
        const string WelcomeText = "Drop images or a folder here\nor use Add Images / Add Folder above\n\n"
            + "← / A  keep left        → / D  keep right\nWheel  zoom      Drag  pan      Double-click  fit\nV  switch view      Ctrl+Z  undo";
        const int MaxUndo = 500;
        // A pick that comes this soon after a new pair appeared is ignored: it is the second click of a
        // double-click on a Keep button, not a decision about images nobody has looked at yet.
        const int PickGuardMs = 250;

        sealed class UndoEntry
        {
            public Tournament.Snapshot Before;
            public string Loser;
            public string MovedTo;
            public bool Reversible;
        }

        // A decoded image, or one being decoded; Cancel tells a decode that hasn't started yet to skip it.
        sealed class CacheEntry
        {
            public Task<LoadedImage> Task;
            public CancellationTokenSource Cancel;
        }

        static string StatePath { get { return Path.Combine(Settings.Folder, "tournament.txt"); } }

        // At most two full-size decodes at once: the current pair goes first and the preloads after it, and a
        // burst of undos can't start dozens of decodes that fill memory.
        static readonly SemaphoreSlim DecodeSlots = new SemaphoreSlim(2);

        readonly Settings _settings = Settings.Load();
        readonly string[] _startupPaths;
        readonly List<UndoEntry> _history = new List<UndoEntry>();
        readonly Dictionary<string, CacheEntry> _cache = new Dictionary<string, CacheEntry>(StringComparer.OrdinalIgnoreCase);
        readonly Random _random = new Random();
        readonly ToolTip _tips = new ToolTip();
        readonly Timer _saveTimer = new Timer { Interval = 750 };
        readonly List<string> _pendingPaths = new List<string>();
        Tournament _t = new Tournament();
        int _showToken, _lastActionIndex, _pairShownAt;
        bool _ready, _syncingMode, _collecting, _closing, _settingsLoaded;
        // _resuming: the saved tournament's files are being checked; adds wait for it.
        // _holdState: leave tournament.txt alone (while resuming, or after "not now" on missing images)
        //   until a new tournament is started.
        // _deleteConfirmed: "Delete permanently" was confirmed in this session.
        // _busy: a pick is being carried out (see Choose); _closeWhenDone: the window was closed meanwhile.
        bool _resuming, _holdState, _deleteConfirmed, _busy, _closeWhenDone;
        string _shownLeft, _shownRight;
        string[] _onScreen = new string[0]; // the images the view is drawing right now

        Bitmap _logo;
        CompareView _view;
        LogoBox _logoBox;
        FlowLayoutPanel _bar;
        TableLayoutPanel _bottom;
        Button _btnLeft, _btnRight, _btnUndo, _btnBrowse;
        Label _lblStatus;
        ComboBox _cmbAction;
        TextBox _txtLosers;
        CheckBox _chkSubfolders, _chkShuffle, _chkKeepZoom, _chkLight, _tglSide, _tglSplit;

        // Themed controls by role, re-colored by ApplyTheme.
        readonly List<Button> _buttons = new List<Button>();
        readonly List<CheckBox> _toggles = new List<CheckBox>();
        readonly List<CheckBox> _checks = new List<CheckBox>();
        readonly List<Label> _labels = new List<Label>();
        readonly List<Label> _separators = new List<Label>();

        public MainForm(string[] startupPaths)
        {
            _startupPaths = startupPaths;
            if (!_settings.Has("theme"))
            {
                // First launch: ask once, before the main window appears, and remember the answer.
                _settings.Set("theme", ChooseLookDialog.Ask().IsLight ? "light" : "dark");
                _settings.Save();
            }
            Theme.Current = _settings.Get("theme", "dark") == "light" ? Palette.Light : Palette.Dark;
            _saveTimer.Tick += delegate { FlushState(); };
            BuildUi();
            ApplySettings();
            ApplyTheme();
            _settingsLoaded = true;
        }

        // ---------- UI construction ----------

        void BuildUi()
        {
            SuspendLayout();
            AutoScaleDimensions = new SizeF(96f, 96f);
            AutoScaleMode = AutoScaleMode.Dpi;
            Text = AppName;
            Font = new Font("Segoe UI", 9f);
            MinimumSize = new Size(900, 560);
            Size = new Size(1400, 900);
            StartPosition = FormStartPosition.CenterScreen;
            AllowDrop = true;
            try { Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath); }
            catch (ArgumentException) { }

            _logo = Theme.LoadLogo(Theme.Current);
            _view = new CompareView { Dock = DockStyle.Fill, AllowDrop = true, Logo = _logo };
            _view.ShowMessage(WelcomeText);

            var bar = _bar = new FlowLayoutPanel
            {
                Dock = DockStyle.Top,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                WrapContents = true,
                Padding = new Padding(6, 5, 6, 5),
                AllowDrop = true,
            };

            var btnImages = MakeButton("Add Images…", "Pick one or more images (Ctrl+O)");
            btnImages.Click += delegate { AddImagesDialog(); };
            var btnFolder = MakeButton("Add Folder…", "Pick a folder of images (Ctrl+Shift+O)");
            btnFolder.Click += delegate { AddFolderDialog(); };
            _chkSubfolders = MakeCheck("Subfolders", "Also include images in subfolders when adding a folder");
            _chkShuffle = MakeCheck("Shuffle", "Randomize the order of newly added images");

            _cmbAction = new ComboBox
            {
                DropDownStyle = ComboBoxStyle.DropDownList,
                FlatStyle = FlatStyle.Flat,
                Width = 160,
                Anchor = AnchorStyles.Left,
                Margin = new Padding(3, 3, 3, 3),
            };
            _cmbAction.Items.AddRange(LoserActionNames);
            _cmbAction.SelectedIndexChanged += OnLoserActionChanged;
            // Like the checkboxes: once a choice is made, the wheel should zoom again, not change the action.
            _cmbAction.DropDownClosed += delegate { BeginInvoke(new Action(() => _view.Focus())); };
            _tips.SetToolTip(_cmbAction, "What happens to the image you didn't pick");

            _txtLosers = new TextBox
            {
                Width = 140,
                BorderStyle = BorderStyle.FixedSingle,
                Anchor = AnchorStyles.Left,
            };
            _txtLosers.HandleCreated += delegate
            {
                NativeMethods.SendMessage(_txtLosers.Handle, NativeMethods.EM_SETCUEBANNER, (IntPtr)1, "A \"Losers\" folder next to each image");
            };
            _txtLosers.Leave += delegate { SaveOptions(); };
            _tips.SetToolTip(_txtLosers, "Where losing images are moved. Leave empty to use a \"Losers\" folder beside each image.\nA relative name is created beside each image.");
            _btnBrowse = MakeButton("Browse…", "Choose the Losers folder");
            _btnBrowse.Click += delegate { BrowseLosersFolder(); };

            _tglSide = MakeToggle("Side by Side", "Show the two images next to each other (V)");
            _tglSplit = MakeToggle("Split", "Overlap the images with a movable divider (V)");
            _tglSide.CheckedChanged += delegate { if (_tglSide.Checked) SetMode(ViewMode.SideBySide); else if (!_syncingMode) _tglSide.Checked = !_tglSplit.Checked; };
            _tglSplit.CheckedChanged += delegate { if (_tglSplit.Checked) SetMode(ViewMode.Split); else if (!_syncingMode) _tglSplit.Checked = !_tglSide.Checked; };

            var btnFit = MakeButton("Fit", "Reset zoom and pan (F or double-click)");
            btnFit.Click += delegate { _view.ResetView(); _view.Focus(); };
            _chkKeepZoom = MakeCheck("Keep zoom", "Keep the zoom and position when moving to the next match");
            foreach (var option in new[] { _chkSubfolders, _chkShuffle, _chkKeepZoom })
                option.CheckedChanged += delegate { SaveOptions(); };
            _btnUndo = MakeButton("Undo", "Undo the last pick (Ctrl+Z). Not available after deleting or recycling.");
            _btnUndo.Click += delegate { Undo(); };
            var btnNew = MakeButton("New Tournament", "Clear the current tournament");
            btnNew.Click += delegate { NewTournament(true); };
            _chkLight = MakeCheck("Light mode", "Switch between the dark and light look");
            _chkLight.Checked = Theme.IsLight;
            _chkLight.CheckedChanged += delegate { SetLightMode(_chkLight.Checked); };

            if (_logo != null)
                bar.Controls.Add(_logoBox = new LogoBox(_logo)
                {
                    Size = new Size(20 * _logo.Width / _logo.Height, 20),
                    Anchor = AnchorStyles.Left,
                    Margin = new Padding(4, 3, 12, 3),
                });
            bar.Controls.AddRange(new Control[]
            {
                btnImages, btnFolder, _chkSubfolders, _chkShuffle, MakeSeparator(),
                MakeLabel("Loser:"), _cmbAction, _txtLosers, _btnBrowse, MakeSeparator(),
                _tglSide, _tglSplit, btnFit, _chkKeepZoom, MakeSeparator(),
                _btnUndo, btnNew, MakeSeparator(), _chkLight,
            });

            var bottom = _bottom = new TableLayoutPanel
            {
                Dock = DockStyle.Bottom,
                Height = 74,
                ColumnCount = 3,
                RowCount = 1,
                Padding = new Padding(6),
                AllowDrop = true,
            };
            bottom.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 30));
            bottom.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 40));
            bottom.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 30));
            bottom.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

            _btnLeft = MakeBigButton();
            _btnLeft.Click += delegate { OnLeftButton(); };
            _btnRight = MakeBigButton();
            _btnRight.Click += delegate { OnRightButton(); };
            _lblStatus = new Label
            {
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Segoe UI", 10f),
                UseMnemonic = false,
            };
            bottom.Controls.Add(_btnLeft, 0, 0);
            bottom.Controls.Add(_lblStatus, 1, 0);
            bottom.Controls.Add(_btnRight, 2, 0);

            Controls.Add(_view);
            Controls.Add(bottom);
            Controls.Add(bar);
            WireDragDrop(this);
            ResumeLayout(false);
            PerformLayout();
        }

        Button MakeButton(string text, string tip)
        {
            var b = new Button
            {
                Text = text,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                FlatStyle = FlatStyle.Flat,
                Padding = new Padding(3, 2, 3, 2),
                Margin = new Padding(2, 3, 2, 3),
                UseVisualStyleBackColor = false,
                Anchor = AnchorStyles.Left,
            };
            _buttons.Add(b);
            _tips.SetToolTip(b, tip);
            return b;
        }

        Button MakeBigButton()
        {
            var b = MakeButton("", null);
            b.AutoSize = false;
            b.Dock = DockStyle.Fill;
            b.Font = new Font("Segoe UI", 13f, FontStyle.Bold);
            return b;
        }

        CheckBox MakeToggle(string text, string tip)
        {
            // AutoSize doesn't account for button appearance, so size it from the text. MeasureText works in
            // screen pixels, but sizes set here are in 96-DPI units that the form scales up afterwards, so
            // convert first; otherwise the toggles were scaled twice (too big) above 100% display scaling.
            Size textSize = TextRenderer.MeasureText(text, Font);
            float screenScale;
            using (var g = Graphics.FromHwnd(IntPtr.Zero)) screenScale = Math.Max(1f, g.DpiX / 96f);
            var c = new CheckBox
            {
                Text = text,
                Appearance = Appearance.Button,
                AutoSize = false,
                Size = new Size((int)Math.Ceiling(textSize.Width / screenScale) + 22, (int)Math.Ceiling(textSize.Height / screenScale) + 14),
                FlatStyle = FlatStyle.Flat,
                Margin = new Padding(2, 3, 2, 3),
                TextAlign = ContentAlignment.MiddleCenter,
                Anchor = AnchorStyles.Left,
            };
            // The selected toggle is red, so its text uses the on-accent color.
            c.CheckedChanged += delegate { c.ForeColor = c.Checked ? Theme.OnAccent : Theme.Text; };
            c.Click += delegate { _view.Focus(); };
            _toggles.Add(c);
            _tips.SetToolTip(c, tip);
            return c;
        }

        CheckBox MakeCheck(string text, string tip)
        {
            var c = new CheckBox { Text = text, AutoSize = true, Anchor = AnchorStyles.Left, Margin = new Padding(4, 3, 0, 3) };
            // Hand focus back to the images so the mouse wheel and shortcuts keep working.
            c.Click += delegate { _view.Focus(); };
            _checks.Add(c);
            _tips.SetToolTip(c, tip);
            return c;
        }

        Label MakeLabel(string text)
        {
            var l = new Label { Text = text, AutoSize = true, Anchor = AnchorStyles.Left, Margin = new Padding(3, 3, 0, 3) };
            _labels.Add(l);
            return l;
        }

        Label MakeSeparator()
        {
            var l = new Label { AutoSize = false, Width = 1, Height = 26, Margin = new Padding(5, 4, 5, 4) };
            _separators.Add(l);
            return l;
        }

        // ---------- Theme ----------

        // Colors every control from the current palette; called at startup and when Light mode is toggled.
        void ApplyTheme()
        {
            SuspendLayout();
            BackColor = Theme.Back;
            ForeColor = Theme.Text;
            _view.BackColor = Theme.Back;
            _view.ForeColor = Theme.Text;
            _bar.BackColor = _bottom.BackColor = Theme.Panel;
            _lblStatus.ForeColor = Theme.Text;
            _lblStatus.BackColor = Theme.Panel;

            foreach (var b in _buttons)
            {
                bool big = b == _btnLeft || b == _btnRight;
                b.ForeColor = Theme.Text;
                b.BackColor = Theme.Control;
                b.FlatAppearance.BorderColor = big ? Theme.Accent : Theme.Border;
                b.FlatAppearance.MouseOverBackColor = big ? Theme.AccentFillHover : Theme.Hover;
                b.FlatAppearance.MouseDownBackColor = big ? Theme.AccentFill : Theme.Border;
            }
            foreach (var c in _toggles)
            {
                c.BackColor = Theme.Control;
                c.ForeColor = c.Checked ? Theme.OnAccent : Theme.Text;
                c.FlatAppearance.BorderColor = Theme.Border;
                c.FlatAppearance.CheckedBackColor = Theme.AccentFill;
                c.FlatAppearance.MouseOverBackColor = Theme.Hover;
                c.FlatAppearance.MouseDownBackColor = Theme.Border;
            }
            foreach (var c in _checks) { c.ForeColor = Theme.Text; c.BackColor = Theme.Panel; }
            foreach (var l in _labels) { l.ForeColor = Theme.Dim; l.BackColor = Theme.Panel; }
            foreach (var l in _separators) l.BackColor = Theme.Border;
            _cmbAction.BackColor = _txtLosers.BackColor = Theme.Control;
            _cmbAction.ForeColor = _txtLosers.ForeColor = Theme.Text;

            var oldLogo = _logo;
            _logo = Theme.LoadLogo(Theme.Current);
            _view.Logo = _logo;
            if (_logoBox != null) { _logoBox.BackColor = Theme.Panel; _logoBox.Image = _logo; }
            if (oldLogo != null && oldLogo != _logo) oldLogo.Dispose();

            if (IsHandleCreated) NativeMethods.UseDarkTitleBar(Handle, !Theme.IsLight);
            UpdateStatus();
            ResumeLayout(true);
            Invalidate(true);
        }

        void SetLightMode(bool light)
        {
            if (light == Theme.IsLight) return;
            Theme.Current = light ? Palette.Light : Palette.Dark;
            _settings.Set("theme", light ? "light" : "dark");
            _settings.Save();
            ApplyTheme();
            _view.Focus();
        }

        void WireDragDrop(Control root)
        {
            root.DragEnter += OnDragEnter;
            root.DragOver += OnDragEnter;
            root.DragLeave += delegate { _view.DropHint = false; };
            root.DragDrop += OnDragDrop;
            foreach (Control c in root.Controls)
                if (c.AllowDrop) WireDragDrop(c);
        }

        void SetMode(ViewMode mode)
        {
            _syncingMode = true;
            _tglSide.Checked = mode == ViewMode.SideBySide;
            _tglSplit.Checked = mode == ViewMode.Split;
            _syncingMode = false;
            _view.Mode = mode;
            _view.Focus();
            SaveOptions();
        }

        // ---------- Settings & lifecycle ----------

        void ApplySettings()
        {
            _lastActionIndex = Math.Max(0, Math.Min(LoserActionNames.Length - 1, _settings.GetInt("loserAction", 0)));
            _cmbAction.SelectedIndex = _lastActionIndex;
            _txtLosers.Text = _settings.Get("losersFolder", "");
            _chkSubfolders.Checked = _settings.GetBool("subfolders", false);
            _chkShuffle.Checked = _settings.GetBool("shuffle", false);
            _chkKeepZoom.Checked = _settings.GetBool("keepZoom", false);
            SetMode(_settings.Get("viewMode", "") == "Split" ? ViewMode.Split : ViewMode.SideBySide);
            UpdateLosersControls();
            UpdateStatus();
        }

        protected override void OnHandleCreated(EventArgs e)
        {
            base.OnHandleCreated(e);
            NativeMethods.UseDarkTitleBar(Handle, !Theme.IsLight);
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            var bounds = new Rectangle(_settings.GetInt("x", 0), _settings.GetInt("y", 0), _settings.GetInt("w", 0), _settings.GetInt("h", 0));
            if (bounds.Width >= MinimumSize.Width && bounds.Height >= MinimumSize.Height
                && Screen.AllScreens.Any(s => s.WorkingArea.IntersectsWith(bounds)))
            {
                StartPosition = FormStartPosition.Manual;
                Bounds = bounds;
            }
            if (_settings.GetBool("maximized", false)) WindowState = FormWindowState.Maximized;
        }

        protected override async void OnShown(EventArgs e)
        {
            base.OnShown(e);
            _view.Focus();
            await OfferResume();
            if (_startupPaths.Length > 0 && !IsDisposed && !_closing) AddPaths(_startupPaths);
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);
            // A pick still being carried out (the Recycle Bin keeps the window responsive meanwhile) finishes
            // first and the window closes after it; otherwise its loser could be gone while the saved
            // tournament still lists it.
            if (_busy && e.CloseReason == CloseReason.UserClosing)
            {
                e.Cancel = true;
                _closeWhenDone = true;
                return;
            }
            var b = WindowState == FormWindowState.Normal ? Bounds : RestoreBounds;
            _settings.Set("x", b.X);
            _settings.Set("y", b.Y);
            _settings.Set("w", b.Width);
            _settings.Set("h", b.Height);
            _settings.Set("maximized", WindowState == FormWindowState.Maximized);
            StoreOptions();
            _settings.Save();
            _closing = true;
            FlushState(); // write any pick still waiting on the save timer
        }

        void StoreOptions()
        {
            _settings.Set("loserAction", _cmbAction.SelectedIndex);
            _settings.Set("losersFolder", _txtLosers.Text.Trim());
            _settings.Set("subfolders", _chkSubfolders.Checked);
            _settings.Set("shuffle", _chkShuffle.Checked);
            _settings.Set("keepZoom", _chkKeepZoom.Checked);
            _settings.Set("viewMode", _view.Mode);
            _settings.Set("theme", Theme.IsLight ? "light" : "dark");
        }

        // Options are saved as soon as they change (not only on a clean close), so after a crash losers still
        // go where the user last said. Window placement is saved on close.
        void SaveOptions()
        {
            if (!_settingsLoaded) return;
            StoreOptions();
            _settings.Save();
        }

        // Another launch of the app (Send to, a second double-click, files dropped on the .exe) handed its
        // paths to this window instead of starting a second copy that would share tournament.txt. False if
        // this window is closing and can't take them (that launch then runs itself).
        public bool AddFromAnotherLaunch(string[] paths)
        {
            if (IsDisposed || _closing || _closeWhenDone) return false;
            if (WindowState == FormWindowState.Minimized) NativeMethods.Restore(Handle); // back to maximized if it was
            Activate();
            if (paths.Length > 0) AddPaths(paths);
            return true;
        }

        async Task OfferResume()
        {
            Tournament saved = null;
            try
            {
                if (File.Exists(StatePath))
                    using (var r = new StreamReader(StatePath)) saved = Tournament.Load(r);
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            if (saved == null || !saved.HasMatch) { DeleteState(); return; }

            var answer = MessageBox.Show(this,
                string.Format("Resume your previous tournament?\n\nRound {0}, {1} images still in the running, {2} of {3} matches decided.",
                    saved.Round, saved.Remaining, saved.MatchesPlayed, saved.TotalMatches),
                AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if (answer != DialogResult.Yes) { DeleteState(); return; }

            // Checking thousands of files takes a while (much longer on a network drive), so do it off the UI
            // thread. Until it's settled the saved file is left alone (closing now must not delete it), and
            // images added meanwhile wait so they join the resumed tournament.
            _holdState = _resuming = true;
            List<string> alive = saved.AliveImages().ToList(), missing;
            _view.Loading = true;
            try { missing = await Task.Run(() => alive.Where(p => !File.Exists(p)).ToList()); }
            finally { _resuming = false; }
            if (IsDisposed || _closing) return;
            _view.Loading = false;

            if (missing.Count > 0)
            {
                // Usually a drive that isn't connected yet: don't quietly drop those images and save.
                var keepGoing = MessageBox.Show(this,
                    string.Format("{0} of {1} images in your saved tournament can't be found.\n\n"
                        + "If they are on a drive that isn't connected, choose No, connect the drive and start the app again. "
                        + "The saved tournament is kept until you start a new one.\n\nResume without them?", missing.Count, alive.Count),
                    AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2);
                if (keepGoing != DialogResult.Yes)
                {
                    // Kept for later, unless images dropped during the questions already started a new one.
                    _holdState = _t.IsEmpty;
                    AddPendingPaths();
                    return;
                }
                saved.RemoveAll(missing);
            }
            // Images dropped while the resume question was open join the resumed tournament.
            if (!_t.IsEmpty) saved.Add(_t.AliveImages().Where(p => !saved.Contains(p)).ToList());
            _t = saved;
            _history.Clear();
            _holdState = false;
            if (missing.Count > 0) SaveStateSoon();
            ShowCurrent();
            AddPendingPaths();
        }

        void AddPendingPaths()
        {
            if (_pendingPaths.Count > 0) AddPaths(new string[0]);
        }

        // Writing the whole list on every pick is slow for big tournaments, so picks in quick
        // succession share one write. FlushState writes immediately (used when closing).
        void SaveStateSoon()
        {
            if (!_saveTimer.Enabled) _saveTimer.Start();
        }

        void FlushState()
        {
            _saveTimer.Stop();
            SaveState();
        }

        void SaveState()
        {
            if (_holdState) return;
            try
            {
                if (_t.IsEmpty) { DeleteState(); return; }
                Directory.CreateDirectory(Settings.Folder);
                string tmp = StatePath + ".tmp";
                using (var fs = new FileStream(tmp, FileMode.Create, FileAccess.Write))
                using (var w = new StreamWriter(fs))
                {
                    _t.Save(w);
                    w.Flush();
                    fs.Flush(true); // on disk before it replaces the old file, so a power cut can't leave it empty
                }
                if (File.Exists(StatePath)) File.Replace(tmp, StatePath, null);
                else File.Move(tmp, StatePath);
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }

        static void DeleteState()
        {
            try { File.Delete(StatePath); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }

        // ---------- Adding images ----------

        void AddImagesDialog()
        {
            using (var d = new OpenFileDialog { Multiselect = true, Title = "Add images", Filter = LoadedImage.FileDialogFilter })
            {
                string last = _settings.Get("lastFolder", "");
                if (Directory.Exists(last)) d.InitialDirectory = last;
                if (d.ShowDialog(this) != DialogResult.OK) return;
                _settings.Set("lastFolder", Path.GetDirectoryName(d.FileNames[0]));
                SaveOptions();
                AddPaths(d.FileNames);
            }
        }

        void AddFolderDialog()
        {
            string folder = FolderPicker.Show(this, "Add a folder of images", _settings.Get("lastFolder", ""));
            if (folder == null) return;
            _settings.Set("lastFolder", folder);
            SaveOptions();
            AddPaths(new[] { folder });
        }

        void OnDragEnter(object sender, DragEventArgs e)
        {
            bool files = e.Data.GetDataPresent(DataFormats.FileDrop);
            e.Effect = files ? DragDropEffects.Copy : DragDropEffects.None;
            _view.DropHint = files;
        }

        void OnDragDrop(object sender, DragEventArgs e)
        {
            _view.DropHint = false;
            var paths = e.Data.GetData(DataFormats.FileDrop) as string[];
            // Finish the drop before showing any message box so Explorer isn't left waiting.
            if (paths != null) BeginInvoke(new Action(() => { Activate(); AddPaths(paths); }));
        }

        // Looks for images off the UI thread (large folders take a while), then adds them.
        // Paths dropped while a search is running, or while a saved tournament is being resumed, are queued
        // and handled after it.
        async void AddPaths(IEnumerable<string> paths)
        {
            _pendingPaths.AddRange(paths);
            if (_collecting || _resuming) return;
            _collecting = true;
            UseWaitCursor = true;
            try
            {
                while (_pendingPaths.Count > 0 && !_closing && !_resuming)
                {
                    string[] batch = _pendingPaths.ToArray();
                    _pendingPaths.Clear();
                    bool subfolders = _chkSubfolders.Checked;
                    string losersSetting = _txtLosers.Text;
                    List<string> found;
                    try { found = await Task.Run(() => CollectImages(batch, subfolders, losersSetting)); }
                    catch (Exception ex)
                    {
                        if (_closing || IsDisposed) return;
                        MessageBox.Show(this, "Couldn't look for images there:\n\n" + ex.Message, AppName, MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        continue;
                    }
                    if (_closing || IsDisposed) return;
                    // A saved tournament started resuming while this scan ran: these join it once it's settled.
                    if (_resuming) { _pendingPaths.AddRange(found); continue; }
                    AddFound(found);
                }
            }
            finally
            {
                _collecting = false;
                if (_closing) _pendingPaths.Clear();
                if (!IsDisposed) UseWaitCursor = false;
            }
        }

        void AddFound(List<string> found)
        {
            var fresh = found.Where(p => !_t.Contains(p)).ToList();
            if (fresh.Count == 0)
            {
                MessageBox.Show(this, found.Count == 0 ? "No images were found there." : "Those images are already in the tournament.",
                    AppName, MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            if (_chkShuffle.Checked)
                for (int i = fresh.Count - 1; i > 0; i--)
                {
                    int j = _random.Next(i + 1);
                    string tmp = fresh[i]; fresh[i] = fresh[j]; fresh[j] = tmp;
                }
            if (_t.IsEmpty)
            {
                // A new tournament replaces a saved one that was kept for later (its images couldn't be found
                // when resuming): ask first, so images arriving with the launch don't quietly overwrite it.
                if (_holdState && MessageBox.Show(this,
                        "Start a new tournament with these images?\n\nThe saved tournament you kept for later will be replaced.",
                        AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Question, MessageBoxDefaultButton.Button2) != DialogResult.Yes)
                    return;
                _history.Clear();
                _holdState = false;
            }
            _t.Add(fresh);
            SaveStateSoon();
            ShowCurrent();
        }

        // Runs on a worker thread: touches no controls; the UI settings it needs are passed in.
        static List<string> CollectImages(IEnumerable<string> paths, bool subfolders, string losersSetting)
        {
            var result = new List<string>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var folders = new HashSet<string>(); // folders already scanned, by on-disk identity
            foreach (var raw in paths)
            {
                string path;
                try { path = Path.GetFullPath(raw); }
                catch (ArgumentException) { continue; }
                catch (NotSupportedException) { continue; }
                catch (IOException) { continue; }   // e.g. a path that is too long
                catch (System.Security.SecurityException) { continue; }
                if (Directory.Exists(path)) CollectFolder(path, subfolders, losersSetting, result, seen, folders);
                else if (File.Exists(path) && LoadedImage.IsImageFile(path) && seen.Add(path)) result.Add(path);
            }
            result.Sort(NativeMethods.StrCmpLogicalW); // natural order: 2.jpg before 10.jpg
            return result;
        }

        static void CollectFolder(string folder, bool subfolders, string losersSetting, List<string> result, HashSet<string> seen,
            HashSet<string> folders)
        {
            try
            {
                // A junction or symbolic link can lead back to a folder already scanned (even to a parent,
                // which loops): every folder is scanned once, recognised by its volume and file ID.
                string id = NativeMethods.FileKey(folder);
                if (id != null && !folders.Add(id)) return;
                // FileInfo attributes come with the directory listing, so no extra call per file.
                // "._name.jpg" files are macOS resource forks, not images.
                var dir = new DirectoryInfo(folder);
                foreach (var f in dir.EnumerateFiles())
                    if (LoadedImage.IsImageFile(f.Name) && (f.Attributes & FileAttributes.Hidden) == 0 && !f.Name.StartsWith("._")
                        && seen.Add(f.FullName))
                        result.Add(f.FullName);
                if (!subfolders) return;
                string losers = null;
                try { losers = LosersFolderFor(Path.Combine(folder, "x"), losersSetting); }
                catch (ArgumentException) { }   // an invalid Losers setting shouldn't hide the subfolders
                catch (NotSupportedException) { }
                catch (IOException) { }
                // Hidden folders are skipped like hidden files. Among them are $RECYCLE.BIN, whose recycled
                // images (losers this app recycled included) would otherwise come back when a whole drive is
                // added, System Volume Information, and the profile's legacy junctions such as "My Pictures"
                // that loop back on themselves. (Only Hidden: customized folders can carry the System flag.)
                foreach (var sub in dir.EnumerateDirectories())
                    if ((sub.Attributes & FileAttributes.Hidden) == 0
                        && !string.Equals(sub.FullName, losers, StringComparison.OrdinalIgnoreCase))
                        CollectFolder(sub.FullName, subfolders, losersSetting, result, seen, folders);
            }
            catch (UnauthorizedAccessException) { }
            catch (IOException) { }
            catch (ArgumentException) { }
            catch (System.Security.SecurityException) { }
        }

        // ---------- Showing matches ----------

        Task<LoadedImage> GetImage(string path)
        {
            CacheEntry entry;
            if (!_cache.TryGetValue(path, out entry))
            {
                var cancel = new CancellationTokenSource();
                CancellationToken token = cancel.Token;
                entry = new CacheEntry { Cancel = cancel, Task = Task.Run(() => Decode(path, token)) };
                _cache[path] = entry;
            }
            return entry.Task;
        }

        // Waits for a decode slot, then decodes, unless the image stopped being wanted while it waited.
        static async Task<LoadedImage> Decode(string path, CancellationToken skip)
        {
            await DecodeSlots.WaitAsync().ConfigureAwait(false);
            try { return skip.IsCancellationRequested ? LoadedImage.NotLoaded(path) : LoadedImage.Load(path); }
            finally { DecodeSlots.Release(); }
        }

        // Starts decoding the next match's images in the background.
        void Preload(string path)
        {
            GetImage(path);
        }

        // Frees decoded images that are not in keep, and cancels decodes of them that haven't started.
        void TrimCache(IEnumerable<string> keep)
        {
            var wanted = new HashSet<string>(keep.Where(p => p != null), StringComparer.OrdinalIgnoreCase);
            foreach (var path in _cache.Keys.Where(k => !wanted.Contains(k)).ToList())
            {
                CacheEntry entry = _cache[path];
                entry.Cancel.Cancel();
                entry.Task.ContinueWith(t =>
                {
                    if (t.Status == TaskStatus.RanToCompletion && t.Result != null) t.Result.Dispose();
                    entry.Cancel.Dispose();
                });
                _cache.Remove(path);
            }
        }

        async void ShowCurrent()
        {
            int token = ++_showToken;
            _ready = false;
            UpdateStatus();

            if (_t.IsEmpty)
            {
                _shownLeft = _shownRight = null;
                _view.ShowMessage(WelcomeText);
                _onScreen = new string[0];
                TrimCache(_onScreen);
                return;
            }

            if (_t.Champion != null)
            {
                string champion = _t.Champion;
                // The champion's buttons are live from here (UpdateStatus above), even if its image is still
                // loading, so the double-click guard starts now too.
                _pairShownAt = Environment.TickCount;
                TrimCache(_onScreen.Concat(new[] { champion }));
                var task = GetImage(champion);
                if (!task.IsCompleted) _view.Loading = true;
                LoadedImage image = await task;
                if (token != _showToken) return;
                _shownLeft = _shownRight = null;
                _view.ShowSingle(image, "★   CHAMPION   ★");
                _onScreen = new[] { champion };
                _pairShownAt = Environment.TickCount; // the champion's buttons are guarded like the Keep buttons
                TrimCache(_onScreen);
                UpdateStatus();
                return;
            }

            string left = _t.Left, right = _t.Right;
            var keep = new List<string> { left, right };
            keep.AddRange(_t.Upcoming(2));
            // Before starting new decodes, drop (or cancel) everything not needed now or still on screen, so a
            // quick run of undos or picks doesn't pile up decodes of pairs that were never shown.
            TrimCache(keep.Concat(_onScreen));
            var tl = GetImage(left);
            var tr = GetImage(right);
            foreach (var p in keep.Skip(2)) Preload(p);
            if (!tl.IsCompleted || !tr.IsCompleted) _view.Loading = true;
            await Task.WhenAll(tl, tr);
            if (token != _showToken) return;

            bool samePair = string.Equals(left, _shownLeft, StringComparison.OrdinalIgnoreCase)
                            && string.Equals(right, _shownRight, StringComparison.OrdinalIgnoreCase);
            _view.ShowPair(tl.Result, tr.Result, !samePair && !_chkKeepZoom.Checked);
            if (!samePair) _pairShownAt = Environment.TickCount;
            _shownLeft = left;
            _shownRight = right;
            _onScreen = new[] { left, right };
            TrimCache(keep); // only after the view stops showing the previous pair
            _ready = true;
            UpdateStatus();
        }

        void UpdateStatus()
        {
            string champion = _t.Champion;
            bool match = _t.HasMatch;

            if (match)
            {
                string bye = _t.ByeThisRound;
                _lblStatus.Text = string.Format("Round {0}   ·   Match {1} of {2}{3}\n{4} images left   ·   {5} of {6} matches decided",
                    _t.Round, _t.MatchInRound, _t.MatchesInRound,
                    bye != null ? "   ·   bye: " + Path.GetFileName(bye) : "",
                    _t.Remaining, _t.MatchesPlayed, _t.TotalMatches);
                _btnLeft.Text = "◀   Keep Left";
                _btnRight.Text = "Keep Right   ▶";
                Text = string.Format("{0} — Round {1}, {2} left", AppName, _t.Round, _t.Remaining);
            }
            else if (champion != null)
            {
                _lblStatus.Text = "Champion: " + Path.GetFileName(champion)
                    + (_t.MatchesPlayed > 0 ? string.Format("\nDecided in {0} matches", _t.MatchesPlayed) : "\nAdd more images to challenge it");
                _btnLeft.Text = "Show in Explorer";
                _btnRight.Text = "New Tournament";
                Text = AppName + " — Champion: " + Path.GetFileName(champion);
            }
            else
            {
                _lblStatus.Text = "Add images to start a tournament";
                _btnLeft.Text = "◀   Keep Left";
                _btnRight.Text = "Keep Right   ▶";
                Text = AppName;
            }

            _btnLeft.Enabled = _btnRight.Enabled = match || champion != null;
            _btnLeft.BackColor = _btnRight.BackColor = match ? Theme.AccentFill : Theme.Control;
            _btnLeft.ForeColor = _btnRight.ForeColor = match ? Theme.OnAccent : Theme.Text;
            _btnUndo.Enabled = _history.Count > 0 && _history[_history.Count - 1].Reversible;
        }

        // ---------- Picking ----------

        // True just after a new pair (or the champion) appeared, when a click is the second half of a double-click.
        bool JustShown { get { return unchecked(Environment.TickCount - _pairShownAt) < PickGuardMs; } }

        void OnLeftButton()
        {
            if (_t.HasMatch) Choose(true);
            else if (_t.Champion != null && !JustShown) ShowInExplorer(_t.Champion);
            _view.Focus();
        }

        void OnRightButton()
        {
            // Double-clicking Keep Right in the final would otherwise land the second click on New Tournament.
            if (_t.HasMatch) Choose(false);
            else if (_t.Champion != null && !JustShown) NewTournament(false);
            _view.Focus();
        }

        void Choose(bool keepLeft)
        {
            if (!_ready || !_t.HasMatch || JustShown || _busy) return;
            // "Delete permanently" left selected from an earlier session is confirmed once before it's used.
            if ((LoserAction)_cmbAction.SelectedIndex == LoserAction.Delete && !_deleteConfirmed)
            {
                if (!ConfirmDelete("\"Delete permanently\" is still selected from last time.\n\n"))
                {
                    _cmbAction.SelectedIndex = (int)LoserAction.Move;
                    return;
                }
                _deleteConfirmed = true;
                if (!_t.HasMatch) return; // the pair is read below, after the question
            }
            // The Recycle Bin operation keeps the window responsive while it runs (seconds on a network drive),
            // so a key press or click could start another pick, an undo or a new tournament in the middle of
            // this one and act on the wrong match. Those wait until this pick is done.
            _busy = true;
            try { ChooseNow(keepLeft); }
            finally
            {
                _busy = false;
                if (_closeWhenDone) BeginInvoke(new Action(Close)); // the window was closed during the pick
            }
        }

        void ChooseNow(bool keepLeft)
        {
            string loser = keepLeft ? _t.Right : _t.Left;
            string winner = keepLeft ? _t.Left : _t.Right;
            var action = (LoserAction)_cmbAction.SelectedIndex;
            // Two spellings of one file (a mapped drive and its UNC path, a short name, a link) can meet in a
            // match. The loser is then just a duplicate entry: drop it, and leave the winner's file alone.
            if (action != LoserAction.Keep && NativeMethods.IsSameFile(loser, winner)) action = LoserAction.Keep;
            string movedTo;
            try { movedTo = DisposeOfLoser(loser, action); }
            catch (Exception ex)
            {
                MessageBox.Show(this, "Couldn't handle the losing image:\n" + loser + "\n\n" + ex.Message + "\n\nThe pick was not recorded.",
                    AppName, MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            _history.Add(new UndoEntry
            {
                Before = _t.TakeSnapshot(),
                Loser = loser,
                MovedTo = movedTo,
                Reversible = action == LoserAction.Move || action == LoserAction.Keep,
            });
            if (_history.Count > MaxUndo) _history.RemoveAt(0);

            _t.Choose(keepLeft);
            SaveStateSoon();
            ShowCurrent();
        }

        // Returns the new location when the file was moved, otherwise null.
        string DisposeOfLoser(string path, LoserAction action)
        {
            if (action == LoserAction.Keep || !File.Exists(path)) return null;
            switch (action)
            {
                case LoserAction.Move:
                    string folder = LosersFolderFor(path);
                    if (string.Equals(folder, Path.GetDirectoryName(path), StringComparison.OrdinalIgnoreCase)) return null;
                    Directory.CreateDirectory(folder);
                    string dest = UniquePath(Path.Combine(folder, Path.GetFileName(path)));
                    File.Move(path, dest);
                    return dest;
                case LoserAction.RecycleBin:
                    if (!RecycleBin.Send(this, path))
                        throw new IOException((Path.GetFullPath(path).Length >= 260
                                ? "Windows can't put files with paths longer than 260 characters in the Recycle Bin, so it was left alone."
                                : "Windows can't put this image in the Recycle Bin (it is on a network drive or USB stick, is too big for "
                                  + "the Recycle Bin, or the Recycle Bin is turned off for that drive), so it was left alone.")
                            + "\n\nTo handle images there, choose \"Move to Losers folder\" or \"Delete permanently\".");
                    return null;
                case LoserAction.Delete:
                    File.Delete(path);
                    return null;
            }
            return null;
        }

        string LosersFolderFor(string imagePath)
        {
            return LosersFolderFor(imagePath, _txtLosers.Text);
        }

        internal static string LosersFolderFor(string imagePath, string losersSetting)
        {
            string imageFolder = Path.GetDirectoryName(imagePath);
            string configured = Environment.ExpandEnvironmentVariables(losersSetting.Trim().Trim('"'));
            if (configured.Length == 0) configured = "Losers";
            // "\Rejects" means the root of the image's own drive, not of the drive the app happened to start on.
            else if ((configured[0] == '\\' || configured[0] == '/') && !configured.StartsWith(@"\\") && !configured.StartsWith("//"))
                configured = Path.Combine(Path.GetPathRoot(imageFolder), configured.TrimStart('\\', '/'));
            // Path.Combine returns the configured path itself when it is absolute.
            string full = Path.GetFullPath(Path.Combine(imageFolder, configured));
            // "Losers\" is the same folder as "Losers": drop a trailing separator (but keep a drive root's) so the
            // comparisons with folder paths work. Without this the scan didn't skip the Losers folder, and ".\"
            // renamed losers in place instead of leaving them alone.
            string root = Path.GetPathRoot(full);
            return full.Length > root.Length ? full.TrimEnd('\\', '/') : full;
        }

        static string UniquePath(string path)
        {
            if (!File.Exists(path)) return path;
            string dir = Path.GetDirectoryName(path), name = Path.GetFileNameWithoutExtension(path), ext = Path.GetExtension(path);
            for (int i = 2; ; i++)
            {
                string candidate = Path.Combine(dir, string.Format("{0} ({1}){2}", name, i, ext));
                if (!File.Exists(candidate)) return candidate;
            }
        }

        void Undo()
        {
            if (_history.Count == 0 || _busy) return;
            var last = _history[_history.Count - 1];
            if (!last.Reversible) return;
            if (last.MovedTo != null)
            {
                try
                {
                    if (File.Exists(last.Loser)) throw new IOException("A file with that name already exists at the original location.");
                    File.Move(last.MovedTo, last.Loser);
                }
                catch (Exception ex)
                {
                    MessageBox.Show(this, "Couldn't move the image back from the Losers folder:\n" + last.MovedTo + "\n\n" + ex.Message,
                        AppName, MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }
            }
            _history.RemoveAt(_history.Count - 1);
            _t.Restore(last.Before); // images added since the pick stay in
            SaveStateSoon();
            ShowCurrent();
            _view.Focus();
        }

        void NewTournament(bool confirm)
        {
            // Not in the middle of a pick, nor while a saved tournament is being resumed (it would come back).
            if (_busy || _resuming) return;
            if (confirm && _t.HasMatch && MessageBox.Show(this,
                    "Abandon the current tournament?\n\nImages that already lost stay where they were moved.",
                    AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Question) != DialogResult.Yes)
                return;
            _t = new Tournament();
            _history.Clear();
            _saveTimer.Stop();
            if (!_holdState) DeleteState(); // a saved tournament kept for later stays until a new one starts
            ShowCurrent();
            _view.Focus();
        }

        static void ShowInExplorer(string path)
        {
            try { using (Process.Start("explorer.exe", "/select,\"" + path + "\"")) { } }
            catch (System.ComponentModel.Win32Exception) { }
        }

        // ---------- Losers settings ----------

        void OnLoserActionChanged(object sender, EventArgs e)
        {
            if (_cmbAction.SelectedIndex == (int)LoserAction.Delete && _lastActionIndex != (int)LoserAction.Delete)
            {
                if (!ConfirmDelete(""))
                {
                    _cmbAction.SelectedIndex = _lastActionIndex;
                    return;
                }
                _deleteConfirmed = true;
            }
            _lastActionIndex = _cmbAction.SelectedIndex;
            UpdateLosersControls();
            SaveOptions();
        }

        bool ConfirmDelete(string intro)
        {
            return MessageBox.Show(this, intro + "Losing images will be permanently deleted. They will NOT go to the Recycle Bin and can't be recovered.\n\nUse permanent delete?",
                AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) == DialogResult.Yes;
        }

        void UpdateLosersControls()
        {
            bool move = _cmbAction.SelectedIndex == (int)LoserAction.Move;
            _txtLosers.Enabled = _btnBrowse.Enabled = move;
        }

        void BrowseLosersFolder()
        {
            string current = _txtLosers.Text.Trim();
            string folder = FolderPicker.Show(this, "Choose the Losers folder",
                Path.IsPathRooted(current) ? current : _settings.Get("lastFolder", ""));
            if (folder == null) return;
            _txtLosers.Text = folder;
            SaveOptions(); // the box never had focus, so its Leave (which saves) won't come
        }

        // ---------- Keyboard ----------

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (KeyRouting.IsInputControl(ActiveControl)) return base.ProcessCmdKey(ref msg, keyData);
            switch (keyData)
            {
                case Keys.Left:
                case Keys.A:
                case Keys.D1:
                case Keys.NumPad1:
                    if (_t.HasMatch && !KeyRouting.IsRepeat(msg)) Choose(true);
                    return true;
                case Keys.Right:
                case Keys.D:
                case Keys.D2:
                case Keys.NumPad2:
                    if (_t.HasMatch && !KeyRouting.IsRepeat(msg)) Choose(false);
                    return true;
                case Keys.Control | Keys.Z:
                case Keys.Back:
                    Undo();
                    return true;
                case Keys.V:
                case Keys.S:
                    SetMode(_view.Mode == ViewMode.Split ? ViewMode.SideBySide : ViewMode.Split);
                    return true;
                case Keys.F:
                case Keys.D0:
                case Keys.Home:
                    _view.ResetView();
                    return true;
                case Keys.Oemplus:
                case Keys.Add:
                    _view.ZoomBy(1.25f);
                    return true;
                case Keys.OemMinus:
                case Keys.Subtract:
                    _view.ZoomBy(0.8f);
                    return true;
                case Keys.Control | Keys.O:
                    AddImagesDialog();
                    return true;
                case Keys.Control | Keys.Shift | Keys.O:
                    AddFolderDialog();
                    return true;
            }
            return base.ProcessCmdKey(ref msg, keyData);
        }
    }
}
