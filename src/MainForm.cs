using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.VisualBasic.FileIO;

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

        sealed class UndoEntry
        {
            public Tournament Before;
            public string Loser;
            public string MovedTo;
            public bool Reversible;
        }

        static string StatePath { get { return Path.Combine(Settings.Folder, "tournament.txt"); } }

        readonly Settings _settings = Settings.Load();
        readonly string[] _startupPaths;
        readonly List<UndoEntry> _history = new List<UndoEntry>();
        readonly Dictionary<string, Task<LoadedImage>> _cache = new Dictionary<string, Task<LoadedImage>>(StringComparer.OrdinalIgnoreCase);
        readonly Random _random = new Random();
        readonly ToolTip _tips = new ToolTip();
        Tournament _t = new Tournament();
        int _showToken, _lastActionIndex;
        bool _ready, _syncingMode;
        string _shownLeft, _shownRight;

        Bitmap _logo;
        CompareView _view;
        Button _btnLeft, _btnRight, _btnUndo, _btnBrowse;
        Label _lblStatus;
        ComboBox _cmbAction;
        TextBox _txtLosers;
        CheckBox _chkSubfolders, _chkShuffle, _chkKeepZoom, _tglSide, _tglSplit;

        public MainForm(string[] startupPaths)
        {
            _startupPaths = startupPaths;
            BuildUi();
            ApplySettings();
        }

        // ---------- UI construction ----------

        void BuildUi()
        {
            SuspendLayout();
            AutoScaleDimensions = new SizeF(96f, 96f);
            AutoScaleMode = AutoScaleMode.Dpi;
            Text = AppName;
            BackColor = Theme.Back;
            ForeColor = Theme.Text;
            Font = new Font("Segoe UI", 9f);
            MinimumSize = new Size(900, 560);
            Size = new Size(1400, 900);
            StartPosition = FormStartPosition.CenterScreen;
            AllowDrop = true;
            try { Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath); }
            catch (ArgumentException) { }

            _logo = Theme.LoadLogo();
            _view = new CompareView { Dock = DockStyle.Fill, BackColor = Theme.Back, ForeColor = Theme.Text, AllowDrop = true, Logo = _logo };
            _view.ShowMessage(WelcomeText);

            var bar = new FlowLayoutPanel
            {
                Dock = DockStyle.Top,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                WrapContents = true,
                BackColor = Theme.Panel,
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
                BackColor = Theme.Control,
                ForeColor = Theme.Text,
                Width = 160,
                Anchor = AnchorStyles.Left,
                Margin = new Padding(3, 3, 3, 3),
            };
            _cmbAction.Items.AddRange(LoserActionNames);
            _cmbAction.SelectedIndexChanged += OnLoserActionChanged;
            _tips.SetToolTip(_cmbAction, "What happens to the image you didn't pick");

            _txtLosers = new TextBox
            {
                Width = 140,
                BackColor = Theme.Control,
                ForeColor = Theme.Text,
                BorderStyle = BorderStyle.FixedSingle,
                Anchor = AnchorStyles.Left,
            };
            _txtLosers.HandleCreated += delegate
            {
                NativeMethods.SendMessage(_txtLosers.Handle, NativeMethods.EM_SETCUEBANNER, (IntPtr)1, "A \"Losers\" folder next to each image");
            };
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
            _btnUndo = MakeButton("Undo", "Undo the last pick (Ctrl+Z). Not available after deleting or recycling.");
            _btnUndo.Click += delegate { Undo(); };
            var btnNew = MakeButton("New Tournament", "Clear the current tournament");
            btnNew.Click += delegate { NewTournament(true); };

            if (_logo != null)
                bar.Controls.Add(new LogoBox(_logo)
                {
                    Size = new Size(20 * _logo.Width / _logo.Height, 20),
                    Anchor = AnchorStyles.Left,
                    Margin = new Padding(4, 3, 12, 3),
                    BackColor = Theme.Panel,
                });
            bar.Controls.AddRange(new Control[]
            {
                btnImages, btnFolder, _chkSubfolders, _chkShuffle, MakeSeparator(),
                MakeLabel("Loser:"), _cmbAction, _txtLosers, _btnBrowse, MakeSeparator(),
                _tglSide, _tglSplit, btnFit, _chkKeepZoom, MakeSeparator(),
                _btnUndo, btnNew,
            });

            var bottom = new TableLayoutPanel
            {
                Dock = DockStyle.Bottom,
                Height = 74,
                ColumnCount = 3,
                RowCount = 1,
                BackColor = Theme.Panel,
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
                ForeColor = Theme.Text,
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
                BackColor = Theme.Control,
                ForeColor = Theme.Text,
                Padding = new Padding(3, 2, 3, 2),
                Margin = new Padding(2, 3, 2, 3),
                UseVisualStyleBackColor = false,
                Anchor = AnchorStyles.Left,
            };
            b.FlatAppearance.BorderColor = Theme.Border;
            b.FlatAppearance.MouseOverBackColor = Theme.Hover;
            _tips.SetToolTip(b, tip);
            return b;
        }

        Button MakeBigButton()
        {
            var b = MakeButton("", null);
            b.AutoSize = false;
            b.Dock = DockStyle.Fill;
            b.Font = new Font("Segoe UI", 13f, FontStyle.Bold);
            b.FlatAppearance.BorderColor = Theme.Accent;
            b.FlatAppearance.MouseOverBackColor = Theme.AccentHover;
            return b;
        }

        CheckBox MakeToggle(string text, string tip)
        {
            // AutoSize doesn't account for button appearance, so size it from the text.
            Size textSize = TextRenderer.MeasureText(text, Font);
            var c = new CheckBox
            {
                Text = text,
                Appearance = Appearance.Button,
                AutoSize = false,
                Size = new Size(textSize.Width + 22, textSize.Height + 14),
                FlatStyle = FlatStyle.Flat,
                BackColor = Theme.Control,
                ForeColor = Theme.Text,
                Margin = new Padding(2, 3, 2, 3),
                TextAlign = ContentAlignment.MiddleCenter,
                Anchor = AnchorStyles.Left,
            };
            c.FlatAppearance.BorderColor = Theme.Border;
            c.FlatAppearance.CheckedBackColor = Theme.AccentDark;
            c.FlatAppearance.MouseOverBackColor = Theme.Hover;
            _tips.SetToolTip(c, tip);
            return c;
        }

        CheckBox MakeCheck(string text, string tip)
        {
            var c = new CheckBox { Text = text, AutoSize = true, ForeColor = Theme.Text, Anchor = AnchorStyles.Left, Margin = new Padding(4, 3, 0, 3) };
            _tips.SetToolTip(c, tip);
            return c;
        }

        static Label MakeLabel(string text)
        {
            return new Label { Text = text, AutoSize = true, ForeColor = Theme.Dim, Anchor = AnchorStyles.Left, Margin = new Padding(3, 3, 0, 3) };
        }

        static Label MakeSeparator()
        {
            return new Label { AutoSize = false, Width = 1, Height = 26, BackColor = Theme.Border, Margin = new Padding(5, 4, 5, 4) };
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
            NativeMethods.UseDarkTitleBar(Handle);
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

        protected override void OnShown(EventArgs e)
        {
            base.OnShown(e);
            OfferResume();
            if (_startupPaths.Length > 0) AddPaths(_startupPaths);
            _view.Focus();
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);
            var b = WindowState == FormWindowState.Normal ? Bounds : RestoreBounds;
            _settings.Set("x", b.X);
            _settings.Set("y", b.Y);
            _settings.Set("w", b.Width);
            _settings.Set("h", b.Height);
            _settings.Set("maximized", WindowState == FormWindowState.Maximized);
            _settings.Set("loserAction", _cmbAction.SelectedIndex);
            _settings.Set("losersFolder", _txtLosers.Text.Trim());
            _settings.Set("subfolders", _chkSubfolders.Checked);
            _settings.Set("shuffle", _chkShuffle.Checked);
            _settings.Set("keepZoom", _chkKeepZoom.Checked);
            _settings.Set("viewMode", _view.Mode);
            _settings.Save();
            SaveState();
        }

        void OfferResume()
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

            foreach (var missing in saved.AliveImages().Where(p => !File.Exists(p)).ToList())
                saved.Remove(missing);
            _t = saved;
            ShowCurrent();
        }

        void SaveState()
        {
            try
            {
                if (_t.IsEmpty) { DeleteState(); return; }
                Directory.CreateDirectory(Settings.Folder);
                string tmp = StatePath + ".tmp";
                using (var w = new StreamWriter(tmp)) _t.Save(w);
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
                AddPaths(d.FileNames);
            }
        }

        void AddFolderDialog()
        {
            string folder = FolderPicker.Show(this, "Add a folder of images", _settings.Get("lastFolder", ""));
            if (folder == null) return;
            _settings.Set("lastFolder", folder);
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

        void AddPaths(IEnumerable<string> paths)
        {
            var found = CollectImages(paths);
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
            if (_t.IsEmpty) _history.Clear();
            _t.Add(fresh);
            SaveState();
            ShowCurrent();
        }

        List<string> CollectImages(IEnumerable<string> paths)
        {
            var result = new List<string>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var raw in paths)
            {
                string path;
                try { path = Path.GetFullPath(raw); }
                catch (ArgumentException) { continue; }
                catch (NotSupportedException) { continue; }
                if (Directory.Exists(path)) CollectFolder(path, result, seen);
                else if (File.Exists(path) && LoadedImage.IsImageFile(path) && seen.Add(path)) result.Add(path);
            }
            result.Sort(NativeMethods.StrCmpLogicalW); // natural order: 2.jpg before 10.jpg
            return result;
        }

        void CollectFolder(string folder, List<string> result, HashSet<string> seen)
        {
            try
            {
                foreach (var f in Directory.EnumerateFiles(folder))
                    if (LoadedImage.IsImageFile(f) && (File.GetAttributes(f) & FileAttributes.Hidden) == 0 && seen.Add(f))
                        result.Add(f);
                if (!_chkSubfolders.Checked) return;
                foreach (var sub in Directory.EnumerateDirectories(folder))
                    if (!string.Equals(sub, LosersFolderFor(Path.Combine(folder, "x")), StringComparison.OrdinalIgnoreCase))
                        CollectFolder(sub, result, seen);
            }
            catch (UnauthorizedAccessException) { }
            catch (IOException) { }
            catch (ArgumentException) { }
        }

        // ---------- Showing matches ----------

        Task<LoadedImage> GetImage(string path)
        {
            Task<LoadedImage> task;
            if (!_cache.TryGetValue(path, out task))
            {
                task = Task.Run(() => LoadedImage.Load(path));
                _cache[path] = task;
            }
            return task;
        }

        // Starts decoding the next match's images in the background.
        void Preload(string path)
        {
            GetImage(path);
        }

        // Frees decoded images that are no longer shown or about to be shown.
        void TrimCache(ICollection<string> keep)
        {
            foreach (var path in _cache.Keys.Where(k => !keep.Contains(k, StringComparer.OrdinalIgnoreCase)).ToList())
            {
                _cache[path].ContinueWith(t => t.Result.Dispose());
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
                TrimCache(new string[0]);
                return;
            }

            if (_t.Champion != null)
            {
                string champion = _t.Champion;
                var task = GetImage(champion);
                if (!task.IsCompleted) _view.Loading = true;
                LoadedImage image = await task;
                if (token != _showToken) return;
                _shownLeft = _shownRight = null;
                _view.ShowSingle(image, "★   CHAMPION   ★");
                TrimCache(new[] { champion });
                UpdateStatus();
                return;
            }

            string left = _t.Left, right = _t.Right;
            var keep = new List<string> { left, right };
            keep.AddRange(_t.Upcoming(2));
            var tl = GetImage(left);
            var tr = GetImage(right);
            foreach (var p in keep.Skip(2)) Preload(p);
            if (!tl.IsCompleted || !tr.IsCompleted) _view.Loading = true;
            await Task.WhenAll(tl, tr);
            if (token != _showToken) return;

            bool samePair = string.Equals(left, _shownLeft, StringComparison.OrdinalIgnoreCase)
                            && string.Equals(right, _shownRight, StringComparison.OrdinalIgnoreCase);
            _view.ShowPair(tl.Result, tr.Result, !samePair && !_chkKeepZoom.Checked);
            _shownLeft = left;
            _shownRight = right;
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
            _btnLeft.BackColor = _btnRight.BackColor = match ? Theme.AccentDark : Theme.Control;
            _btnUndo.Enabled = _history.Count > 0 && _history[_history.Count - 1].Reversible;
        }

        // ---------- Picking ----------

        void OnLeftButton()
        {
            if (_t.HasMatch) Choose(true);
            else if (_t.Champion != null) ShowInExplorer(_t.Champion);
            _view.Focus();
        }

        void OnRightButton()
        {
            if (_t.HasMatch) Choose(false);
            else if (_t.Champion != null) NewTournament(false);
            _view.Focus();
        }

        void Choose(bool keepLeft)
        {
            if (!_ready || !_t.HasMatch) return;
            string loser = keepLeft ? _t.Right : _t.Left;
            var action = (LoserAction)_cmbAction.SelectedIndex;
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
                Before = _t.Clone(),
                Loser = loser,
                MovedTo = movedTo,
                Reversible = action == LoserAction.Move || action == LoserAction.Keep,
            });
            if (_history.Count > MaxUndo) _history.RemoveAt(0);

            _t.Choose(keepLeft);
            SaveState();
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
                    FileSystem.DeleteFile(path, UIOption.OnlyErrorDialogs, RecycleOption.SendToRecycleBin);
                    return null;
                case LoserAction.Delete:
                    File.Delete(path);
                    return null;
            }
            return null;
        }

        string LosersFolderFor(string imagePath)
        {
            string configured = Environment.ExpandEnvironmentVariables(_txtLosers.Text.Trim().Trim('"'));
            // Path.Combine returns the configured path itself when it is absolute.
            return Path.GetFullPath(Path.Combine(Path.GetDirectoryName(imagePath), configured.Length == 0 ? "Losers" : configured));
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
            if (_history.Count == 0) return;
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
            _t = last.Before;
            SaveState();
            ShowCurrent();
            _view.Focus();
        }

        void NewTournament(bool confirm)
        {
            if (confirm && _t.HasMatch && MessageBox.Show(this,
                    "Abandon the current tournament?\n\nImages that already lost stay where they were moved.",
                    AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Question) != DialogResult.Yes)
                return;
            _t = new Tournament();
            _history.Clear();
            DeleteState();
            ShowCurrent();
            _view.Focus();
        }

        static void ShowInExplorer(string path)
        {
            try { Process.Start("explorer.exe", "/select,\"" + path + "\""); }
            catch (System.ComponentModel.Win32Exception) { }
        }

        // ---------- Losers settings ----------

        void OnLoserActionChanged(object sender, EventArgs e)
        {
            if (_cmbAction.SelectedIndex == (int)LoserAction.Delete && _lastActionIndex != (int)LoserAction.Delete
                && MessageBox.Show(this, "Losing images will be permanently deleted. They will NOT go to the Recycle Bin and can't be recovered.\n\nUse permanent delete?",
                    AppName, MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) != DialogResult.Yes)
            {
                _cmbAction.SelectedIndex = _lastActionIndex;
                return;
            }
            _lastActionIndex = _cmbAction.SelectedIndex;
            UpdateLosersControls();
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
            if (folder != null) _txtLosers.Text = folder;
        }

        // ---------- Keyboard ----------

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (ActiveControl is TextBox) return base.ProcessCmdKey(ref msg, keyData);
            switch (keyData)
            {
                case Keys.Left:
                case Keys.A:
                case Keys.D1:
                case Keys.NumPad1:
                    if (_t.HasMatch) Choose(true);
                    return true;
                case Keys.Right:
                case Keys.D:
                case Keys.D2:
                case Keys.NumPad2:
                    if (_t.HasMatch) Choose(false);
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
