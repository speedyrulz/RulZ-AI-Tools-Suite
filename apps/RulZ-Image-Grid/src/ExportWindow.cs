using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using Microsoft.Win32;

namespace RulZImageGrid
{
    public class ExportWindow : Window
    {
        const int JpegMaxSide = 65535;
        const double MaxPixels = 500_000_000; // ~2 GB of 32-bit pixels: beyond this WPF can't allocate the bitmap

        readonly GridSettings S;
        readonly List<ImageItem> items;
        readonly GridLayout layout;

        readonly Slider scaleSlider;
        readonly TextBox scaleBox;
        readonly TextBlock resText, detailText, warnText, statusText, previewNote;
        readonly RadioButton pngRadio, jpgRadio;
        readonly Slider qualitySlider;
        readonly TextBlock qualityText;
        readonly FrameworkElement qualityRow;
        readonly Image previewImg;
        readonly Button saveBtn, cancelBtn;
        readonly Panel presets;
        readonly DispatcherTimer previewTimer;
        double percent = 100;
        bool busy, syncing;

        // The preview only changes when its drawing scale (or, for a transparent background, the format) changes.
        string previewKey;               // what the preview shows, or is being rendered to show
        bool previewPending;
        CancellationTokenSource previewCancel;

        public string SavedPath { get; private set; }
        public string SavedSize { get; private set; }

        public ExportWindow(Window owner, GridSettings settings, List<ImageItem> items)
        {
            S = settings;
            this.items = items;
            layout = GridLayout.Compute(S, items.Select(i => new Size(i.PixelW, i.PixelH)).ToList());

            Owner = owner;
            Title = "Export Image";
            Icon = owner.Icon;
            Width = 960;
            Height = 620;
            MinWidth = 760;
            MinHeight = 480;
            WindowStartupLocation = WindowStartupLocation.CenterOwner;
            ShowInTaskbar = false;
            Theme.Attach(this);

            var root = new DockPanel();

            // ── left: options ──
            var opts = new StackPanel { Width = 320, Margin = new Thickness(18, 4, 18, 18) };

            opts.Children.Add(Ui.Section("Size"));
            opts.Children.Add(Ui.Dim(new TextBlock
            {
                Text = $"Full resolution (100%): {Px(layout.Width)} × {Px(layout.Height)} px",
                Margin = new Thickness(0, 0, 0, 8),
                TextWrapping = TextWrapping.Wrap,
            }));

            var scaleRow = new DockPanel();
            scaleBox = new TextBox { Width = 64, Padding = new Thickness(3, 2, 3, 2), Text = "100", HorizontalContentAlignment = HorizontalAlignment.Right };
            scaleBox.TextChanged += (s, e) =>
            {
                if (syncing) return;
                if (double.TryParse(scaleBox.Text, NumberStyles.Float, CultureInfo.CurrentCulture, out var v) && v >= 1 && v <= 200)
                    SetPercent(v, fromBox: true);
            };
            scaleBox.LostFocus += (s, e) => SyncBox();
            var pctLabel = new TextBlock { Text = " %", VerticalAlignment = VerticalAlignment.Center };
            DockPanel.SetDock(pctLabel, Dock.Right);
            DockPanel.SetDock(scaleBox, Dock.Right);
            scaleRow.Children.Add(pctLabel);
            scaleRow.Children.Add(scaleBox);
            scaleSlider = new Slider
            {
                Minimum = 1,
                Maximum = 100,
                Value = 100,
                TickFrequency = 1,
                IsSnapToTickEnabled = true,
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 10, 0),
            };
            scaleSlider.ValueChanged += (s, e) => { if (!syncing) SetPercent(scaleSlider.Value, fromBox: false); };
            scaleRow.Children.Add(scaleSlider);
            opts.Children.Add(Ui.Row("Scale (% of original)", new Border()));
            opts.Children.Add(scaleRow);

            presets = new WrapPanel { Margin = new Thickness(0, 8, 0, 0) };
            foreach (var p in new[] { 100, 75, 50, 25, 10 })
            {
                var b = Ui.Button(p + "%", (s, e) => SetPercent(p, false));
                b.MinWidth = 50;
                b.Padding = new Thickness(6, 3, 6, 3);
                presets.Children.Add(b);
            }
            opts.Children.Add(presets);

            var resBox = new Border
            {
                BorderThickness = new Thickness(3, 0, 0, 0),
                CornerRadius = new CornerRadius(6),
                Padding = new Thickness(12, 10, 12, 10),
                Margin = new Thickness(0, 14, 0, 0),
            };
            resBox.SetResourceReference(Border.BackgroundProperty, "Control");
            resBox.SetResourceReference(Border.BorderBrushProperty, "Accent");
            var resStack = new StackPanel();
            resStack.Children.Add(Ui.Dim(new TextBlock { Text = "Output resolution", FontSize = 12 }));
            resText = new TextBlock { FontSize = 22, FontWeight = FontWeights.SemiBold, Margin = new Thickness(0, 2, 0, 2) };
            detailText = Ui.Dim(new TextBlock { FontSize = 12 });
            resStack.Children.Add(resText);
            resStack.Children.Add(detailText);
            resBox.Child = resStack;
            opts.Children.Add(resBox);

            opts.Children.Add(Ui.Section("Format"));
            var fmt = new StackPanel { Orientation = Orientation.Horizontal };
            pngRadio = new RadioButton { Content = "PNG (lossless)", IsChecked = true, Margin = new Thickness(0, 0, 18, 0) };
            jpgRadio = new RadioButton { Content = "JPG (smaller file)" };
            pngRadio.Checked += (s, e) => FormatChanged();
            jpgRadio.Checked += (s, e) => FormatChanged();
            fmt.Children.Add(pngRadio);
            fmt.Children.Add(jpgRadio);
            opts.Children.Add(fmt);

            var q = new DockPanel { Margin = new Thickness(0, 10, 0, 0) };
            qualityText = new TextBlock { Width = 36, TextAlignment = TextAlignment.Right, VerticalAlignment = VerticalAlignment.Center };
            DockPanel.SetDock(qualityText, Dock.Right);
            q.Children.Add(qualityText);
            q.Children.Add(new TextBlock { Text = "JPG quality", VerticalAlignment = VerticalAlignment.Center, Width = 80 });
            qualitySlider = new Slider { Minimum = 10, Maximum = 100, Value = 92, TickFrequency = 1, IsSnapToTickEnabled = true, VerticalAlignment = VerticalAlignment.Center };
            qualitySlider.ValueChanged += (s, e) => qualityText.Text = ((int)qualitySlider.Value).ToString();
            qualityText.Text = "92";
            q.Children.Add(qualitySlider);
            qualityRow = q;
            opts.Children.Add(q);

            warnText = Ui.Res(new TextBlock
            {
                TextWrapping = TextWrapping.Wrap,
                Margin = new Thickness(0, 14, 0, 0),
            }, TextBlock.ForegroundProperty, "WarnText");
            opts.Children.Add(warnText);

            // ── bottom buttons ──
            var bottom = new DockPanel { Margin = new Thickness(18, 10, 18, 14), LastChildFill = true };
            var btns = new StackPanel { Orientation = Orientation.Horizontal };
            saveBtn = Ui.Button("Save…", (s, e) => Save(), true);
            saveBtn.IsDefault = true;
            cancelBtn = Ui.Button("Cancel", (s, e) => Close());
            cancelBtn.IsCancel = true;
            cancelBtn.Margin = new Thickness(0);
            btns.Children.Add(saveBtn);
            btns.Children.Add(cancelBtn);
            DockPanel.SetDock(btns, Dock.Right);
            bottom.Children.Add(btns);
            statusText = Ui.Dim(new TextBlock { VerticalAlignment = VerticalAlignment.Center, TextTrimming = TextTrimming.CharacterEllipsis });
            bottom.Children.Add(statusText);
            DockPanel.SetDock(bottom, Dock.Bottom);
            root.Children.Add(bottom);

            var optsScroll = new ScrollViewer { Content = opts, VerticalScrollBarVisibility = ScrollBarVisibility.Auto, Width = 356 };
            DockPanel.SetDock(optsScroll, Dock.Left);
            root.Children.Add(optsScroll);

            // ── right: preview ──
            var previewPanel = new DockPanel { Margin = new Thickness(0, 18, 18, 0) };
            previewNote = Ui.Dim(new TextBlock { FontSize = 12, Margin = new Thickness(0, 6, 0, 0) });
            DockPanel.SetDock(previewNote, Dock.Bottom);
            previewPanel.Children.Add(previewNote);
            previewImg = new Image { Stretch = Stretch.Uniform, StretchDirection = StretchDirection.DownOnly };
            RenderOptions.SetBitmapScalingMode(previewImg, BitmapScalingMode.HighQuality);
            previewPanel.Children.Add(Ui.Res(new Border
            {
                CornerRadius = new CornerRadius(6),
                Padding = new Thickness(12),
                Child = previewImg,
            }, Border.BackgroundProperty, "Workspace"));
            root.Children.Add(previewPanel);
            Content = root;

            previewTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
            previewTimer.Tick += (s, e) => { previewTimer.Stop(); RenderPreview(); };

            FormatChanged();
            UpdateResolution();
            Loaded += (s, e) => RenderPreview();
            // Closing mid-save would leave the save running with nowhere to report back to.
            Closing += (s, e) => { if (busy) e.Cancel = true; };
            Closed += (s, e) => { previewTimer.Stop(); previewCancel?.Cancel(); };
        }

        bool IsJpg => jpgRadio.IsChecked == true;
        double Scale => percent / 100.0;
        int OutW => Math.Max(1, (int)Math.Round(layout.Width * Scale));
        int OutH => Math.Max(1, (int)Math.Round(layout.Height * Scale));

        static string Px(double v) => Math.Round(v).ToString("N0");

        void SetPercent(double v, bool fromBox)
        {
            percent = Math.Max(1, Math.Min(200, v));
            syncing = true;
            scaleSlider.Value = Math.Min(100, Math.Round(percent));
            if (!fromBox) scaleBox.Text = percent.ToString("0.##");
            syncing = false;
            UpdateResolution();
            previewTimer.Stop();
            previewTimer.Start();
        }

        void SyncBox()
        {
            syncing = true;
            scaleBox.Text = percent.ToString("0.##");
            syncing = false;
        }

        void FormatChanged()
        {
            if (qualityRow == null) return;
            qualityRow.Visibility = IsJpg ? Visibility.Visible : Visibility.Collapsed;
            UpdateResolution();
            previewTimer?.Stop();
            previewTimer?.Start();
        }

        void UpdateResolution()
        {
            if (resText == null) return;
            int w = OutW, h = OutH;
            double mp = (double)w * h / 1e6;
            resText.Text = $"{w:N0} × {h:N0} px";
            // Drawing needs the finished image plus a full-size working copy: measured at about twice the image.
            detailText.Text = $"{mp:0.0} megapixels · about {(double)w * h * 8 / 1048576:N0} MB in memory while saving";

            var warnings = new List<string>();
            bool blocked = false;
            if ((double)w * h > MaxPixels)
            {
                warnings.Add("This is too large to render. Lower the scale.");
                blocked = true;
            }
            else if (IsJpg && (w > JpegMaxSide || h > JpegMaxSide))
            {
                warnings.Add($"JPG can't be wider or taller than {JpegMaxSide:N0} px. Lower the scale or use PNG.");
                blocked = true;
            }
            else if ((double)w * h > 150_000_000)
                warnings.Add("Very large image — saving may take a while and use a lot of memory.");
            if (percent > 100)
                warnings.Add("Above 100% enlarges the images, so they'll look softer.");
            if (IsJpg && S.Transparent)
                warnings.Add("JPG doesn't support transparency; the background will be white.");
            warnText.Text = string.Join("\n\n", warnings);
            saveBtn.IsEnabled = !blocked && !busy;
        }

        double PreviewScale => Math.Min(Scale, Math.Min(900 / layout.Width, 700 / layout.Height));

        string PreviewNoteText() => PreviewScale < Scale
            ? $"Preview (reduced to fit — the saved file will be {OutW:N0} × {OutH:N0} px)"
            : "Preview (actual size)";

        async void RenderPreview()
        {
            if (!IsLoaded) return;
            double pscale = PreviewScale;
            bool opaque = IsJpg && S.Transparent; // JPG only changes the picture when the background is transparent
            string key = pscale.ToString("R", CultureInfo.InvariantCulture) + (opaque ? "|opaque" : "");
            if (key == previewKey)
            {
                // Same picture as the one shown (e.g. any scale above the fit size): just refresh the size note.
                if (!previewPending) previewNote.Text = PreviewNoteText();
                return;
            }

            previewKey = key;
            previewCancel?.Cancel();
            var cts = previewCancel = new CancellationTokenSource();
            previewPending = true;
            previewNote.Text = "Rendering preview…";
            try
            {
                var bmp = await Renderer.RunSta(() =>
                    Renderer.Render(S, items, pscale, opaque, null, useThumbs: true, cancel: cts.Token));
                if (cts != previewCancel) return; // a newer preview has started
                previewPending = false;
                previewImg.Source = bmp;
                previewNote.Text = PreviewNoteText();
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                if (cts != previewCancel) return;
                previewPending = false;
                previewKey = null; // try again on the next change
                previewNote.Text = "Preview failed: " + ex.Message;
            }
        }

        async void Save()
        {
            bool jpg = IsJpg;
            string ext = jpg ? ".jpg" : ".png";
            var dlg = new SaveFileDialog
            {
                Title = "Save grid image",
                Filter = jpg ? "JPEG image (*.jpg)|*.jpg;*.jpeg" : "PNG image (*.png)|*.png",
                FileName = "RulZ Image Grid" + ext,
                DefaultExt = ext,
                AddExtension = true,
                OverwritePrompt = true,
            };
            if (dlg.ShowDialog(this) != true) return;

            string path = dlg.FileName;
            double scale = Scale;
            string size = $"{OutW:N0} × {OutH:N0}";
            int quality = (int)qualitySlider.Value;
            bool keepAlpha = S.Transparent && !jpg;
            SetBusy(true);
            try
            {
                await Renderer.RunSta(() =>
                {
                    var bmp = Renderer.Render(S, items, scale, jpg, msg => Report(msg));
                    Report("Encoding and writing file…");
                    Renderer.Save(bmp, path, jpg, quality, keepAlpha);
                    return true;
                });
                SavedPath = path;
                SavedSize = size;
                busy = false; // let the window close
                DialogResult = true;
            }
            catch (Exception ex)
            {
                SetBusy(false);
                statusText.Text = "";
                bool memory = ex is OutOfMemoryException || ex.InnerException is OutOfMemoryException;
                MessageBox.Show(this,
                    "Couldn't save the image:\n\n" + ex.Message + (memory ? "\n\nTry a lower scale." : ""),
                    "Export failed", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        void Report(string msg) => Dispatcher.BeginInvoke(new Action(() => statusText.Text = msg));

        void SetBusy(bool on)
        {
            busy = on;
            Cursor = on ? System.Windows.Input.Cursors.Wait : null;
            cancelBtn.IsEnabled = !on;
            scaleSlider.IsEnabled = scaleBox.IsEnabled = pngRadio.IsEnabled = jpgRadio.IsEnabled = !on;
            presets.IsEnabled = qualitySlider.IsEnabled = !on;
            UpdateResolution();
        }
    }
}
