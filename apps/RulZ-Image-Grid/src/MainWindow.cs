using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Threading;
using Microsoft.Win32;

namespace RulZImageGrid
{
    public class MainWindow : Window
    {
        readonly List<ImageItem> items = new List<ImageItem>();
        readonly GridSettings S = new GridSettings();
        readonly Settings settings;
        GridLayout layout;

        // view
        readonly Canvas canvas = new Canvas { Focusable = true, ClipToBounds = true };
        readonly ScaleTransform zoomT = new ScaleTransform(1, 1);
        readonly Border canvasHolder;
        readonly ScrollViewer scroller;
        readonly FrameworkElement emptyHint;
        readonly Slider zoomSlider;
        readonly TextBlock zoomText = new TextBlock { Width = 48, VerticalAlignment = VerticalAlignment.Center };
        readonly TextBlock infoText = Ui.Dim(new TextBlock { TextWrapping = TextWrapping.Wrap, LineHeight = 20 });
        readonly TextBlock statusText = Ui.Dim(new TextBlock { Margin = new Thickness(12, 5, 12, 5), TextTrimming = TextTrimming.CharacterEllipsis });
        readonly Image logoImage = new Image { Height = 22, Margin = new Thickness(2, 0, 10, 0), VerticalAlignment = VerticalAlignment.Center };
        readonly Image hintLogo = new Image { Height = 40, Opacity = 0.9, Margin = new Thickness(0, 0, 0, 18), HorizontalAlignment = HorizontalAlignment.Center };
        readonly List<Image> tileImages = new List<Image>();
        Rectangle selRect, dropMarker, fileDropMarker;
        Image ghost;

        // Sharper copies of the tiles on screen, for when they're shown bigger than their small preview.
        readonly Dictionary<ImageItem, BitmapSource> sharp = new Dictionary<ImageItem, BitmapSource>();
        readonly DispatcherTimer sharpenTimer;
        CancellationTokenSource sharpenCancel;

        // settings controls
        IntBox columnsBox, paddingBox, captionFontBox, labelFontBox;
        CheckBox captionsCheck, colLabelsCheck, rowLabelsCheck, upscaleCheck, lightCheck;
        TextBox colLabelsBox, rowLabelsBox;
        ComboBox bgCombo, textColorCombo;

        // state
        int selected = -1;
        bool autoFit = true, suppressZoom, userSized, rebuildPending;
        int dragIndex = -1;
        bool dragging;
        Point dragStart;

        static readonly (string Name, Color Color, bool Transparent)[] Backgrounds =
        {
            ("White", Colors.White, false),
            ("Light gray", Color.FromRgb(0xEE, 0xEE, 0xEE), false),
            ("Dark gray", Color.FromRgb(0x2B, 0x2B, 0x2B), false),
            ("Black", Colors.Black, false),
            ("Transparent (PNG)", Colors.Transparent, true),
        };

        public MainWindow(Settings settings)
        {
            this.settings = settings;
            Theme.Attach(this);
            Title = "RulZ Image Grid";
            Icon = Theme.AppIcon();
            Width = 1320;
            Height = 860;
            MinWidth = 800;
            MinHeight = 500;
            WindowStartupLocation = WindowStartupLocation.CenterScreen;
            AllowDrop = true;

            var root = new DockPanel();

            // ── toolbar ──
            var toolbar = new DockPanel { Margin = new Thickness(12, 8, 12, 8), LastChildFill = false };
            var left = new StackPanel { Orientation = Orientation.Horizontal };
            logoImage.Source = Theme.LoadLogo(Theme.Current);
            RenderOptions.SetBitmapScalingMode(logoImage, BitmapScalingMode.HighQuality);
            left.Children.Add(logoImage);
            left.Children.Add(Ui.Dim(new TextBlock
            {
                Text = "I M A G E   G R I D",
                FontSize = 11,
                FontWeight = FontWeights.SemiBold,
                VerticalAlignment = VerticalAlignment.Center,
            }));
            left.Children.Add(Ui.Divider());
            left.Children.Add(Ui.Button("Add Images…", (s, e) => BrowseForImages(), true));
            left.Children.Add(Ui.Button("Remove", (s, e) => RemoveSelected()));
            left.Children.Add(Ui.Button("Sort A–Z", (s, e) => SortByName()));
            left.Children.Add(Ui.Button("Clear All", (s, e) => ClearAll()));
            left.Children.Add(Ui.Divider());
            left.Children.Add(Ui.Button("Export Image…", (s, e) => ShowExport(), true));
            toolbar.Children.Add(left);

            var right = new StackPanel { Orientation = Orientation.Horizontal };
            right.Children.Add(new TextBlock { Text = "Zoom", VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(0, 0, 8, 0) });
            zoomSlider = new Slider
            {
                Width = 170,
                Minimum = Math.Log(0.01),
                Maximum = Math.Log(4),
                VerticalAlignment = VerticalAlignment.Center,
            };
            zoomSlider.ValueChanged += (s, e) => { if (!suppressZoom) SetZoom(Math.Exp(zoomSlider.Value), true); };
            right.Children.Add(zoomSlider);
            right.Children.Add(new Border { Width = 8 });
            right.Children.Add(zoomText);
            right.Children.Add(Ui.Button("Fit", (s, e) => { autoFit = true; FitZoom(); }));
            right.Children.Add(Ui.Button("100%", (s, e) => SetZoom(1, true)));
            right.Children.Add(Ui.Divider());
            lightCheck = Check("Light mode", Theme.IsLight, SetLightMode);
            lightCheck.ToolTip = "Switch between the dark and light look";
            right.Children.Add(lightCheck);
            DockPanel.SetDock(right, Dock.Right);
            toolbar.Children.Add(right);
            var toolbarBar = Ui.Res(new Border { BorderThickness = new Thickness(0, 0, 0, 1), Child = toolbar }, Border.BackgroundProperty, "Panel");
            toolbarBar.SetResourceReference(Border.BorderBrushProperty, "Border");
            DockPanel.SetDock(toolbarBar, Dock.Top);
            root.Children.Add(toolbarBar);

            // ── status bar ──
            var status = Ui.Res(new Border { BorderThickness = new Thickness(0, 1, 0, 0), Child = statusText }, Border.BackgroundProperty, "Panel");
            status.SetResourceReference(Border.BorderBrushProperty, "Border");
            DockPanel.SetDock(status, Dock.Bottom);
            root.Children.Add(status);
            statusText.Text = "Drag images onto the window or click Add Images… Drag tiles to rearrange. Double-click a header to label it.";

            // ── settings panel ──
            var side = new Border
            {
                Width = 296,
                BorderThickness = new Thickness(0, 0, 1, 0),
                Child = new ScrollViewer
                {
                    VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                    Content = BuildSettingsPanel(),
                },
            };
            side.SetResourceReference(Border.BackgroundProperty, "Panel");
            side.SetResourceReference(Border.BorderBrushProperty, "Border");
            DockPanel.SetDock(side, Dock.Left);
            root.Children.Add(side);

            // ── canvas area ──
            canvasHolder = new Border
            {
                LayoutTransform = zoomT,
                Child = canvas,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(24),
            };
            scroller = new ScrollViewer
            {
                HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
                VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                Content = canvasHolder,
            };
            scroller.SetResourceReference(BackgroundProperty, "Workspace");
            scroller.SizeChanged += (s, e) => { if (autoFit) FitZoom(); };
            scroller.PreviewMouseWheel += Scroller_PreviewMouseWheel;
            sharpenTimer = new DispatcherTimer(DispatcherPriority.Background) { Interval = TimeSpan.FromMilliseconds(200) };
            sharpenTimer.Tick += (s, e) => RefreshSharpness();
            scroller.ScrollChanged += (s, e) => ScheduleSharpen(); // scrolling, zooming and resizing all land here
            DpiChanged += (s, e) => ScheduleSharpen();

            // A dashed drop zone with the logo, shown until images are added.
            var dash = new Rectangle { StrokeThickness = 2, StrokeDashArray = new DoubleCollection { 4, 3 }, RadiusX = 12, RadiusY = 12 };
            dash.SetResourceReference(Shape.StrokeProperty, "Border");
            var hintStack = new StackPanel { Margin = new Thickness(64, 40, 64, 40), HorizontalAlignment = HorizontalAlignment.Center };
            RenderOptions.SetBitmapScalingMode(hintLogo, BitmapScalingMode.HighQuality);
            hintLogo.Source = logoImage.Source;
            hintStack.Children.Add(hintLogo);
            hintStack.Children.Add(new TextBlock { Text = "Drop images here", FontSize = 22, FontWeight = FontWeights.SemiBold, HorizontalAlignment = HorizontalAlignment.Center });
            hintStack.Children.Add(Ui.Dim(new TextBlock { Text = "or click  Add Images…", FontSize = 15, Margin = new Thickness(0, 4, 0, 0), HorizontalAlignment = HorizontalAlignment.Center }));
            var hintGrid = new Grid
            {
                IsHitTestVisible = false,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
            };
            hintGrid.Children.Add(dash);
            hintGrid.Children.Add(hintStack);
            emptyHint = hintGrid;
            var center = new Grid();
            center.Children.Add(scroller);
            center.Children.Add(emptyHint);
            root.Children.Add(center);
            Content = root;

            canvas.MouseLeftButtonDown += Canvas_MouseLeftButtonDown;
            canvas.MouseMove += Canvas_MouseMove;
            canvas.MouseLeftButtonUp += Canvas_MouseLeftButtonUp;
            canvas.MouseRightButtonUp += Canvas_MouseRightButtonUp;
            canvas.LostMouseCapture += (s, e) => { if (dragIndex >= 0) EndDrag(); };

            DragOver += Window_DragOver;
            DragLeave += (s, e) => ShowFileDropMarker(-1);
            Drop += Window_Drop;
            PreviewKeyDown += Window_PreviewKeyDown;
            Theme.Changed += OnThemeChanged;
            Closed += (s, e) =>
            {
                Theme.Changed -= OnThemeChanged;
                sharpenTimer.Stop();
                sharpenCancel?.Cancel();
            };

            SetZoom(1, false);
            Rebuild();
        }

        void OnThemeChanged()
        {
            logoImage.Source = hintLogo.Source = Theme.LoadLogo(Theme.Current);
            if (lightCheck != null && lightCheck.IsChecked != Theme.IsLight) lightCheck.IsChecked = Theme.IsLight;
        }

        void SetLightMode(bool light)
        {
            if (light == Theme.IsLight) return;
            Theme.Apply(light ? Palette.Light : Palette.Dark);
            settings.Set("theme", light ? "light" : "dark");
            settings.Save();
        }

        // ───────────────────────── settings panel ─────────────────────────

        UIElement BuildSettingsPanel()
        {
            var p = new StackPanel { Margin = new Thickness(16, 0, 16, 16) };

            p.Children.Add(Ui.Section("Grid"));
            columnsBox = new IntBox(S.Columns, 1, 100);
            columnsBox.ValueChanged += v => { S.Columns = v; RequestRebuild(); };
            p.Children.Add(Ui.Row("Columns", columnsBox));
            paddingBox = new IntBox(S.Padding, 0, 5000, 2);
            paddingBox.ValueChanged += v => { S.Padding = v; userSized = true; RequestRebuild(); };
            p.Children.Add(Ui.Row("Padding (px)", paddingBox));

            bgCombo = new ComboBox { Width = 150 };
            foreach (var b in Backgrounds) bgCombo.Items.Add(b.Name);
            bgCombo.SelectedIndex = 0;
            bgCombo.SelectionChanged += (s, e) =>
            {
                var b = Backgrounds[bgCombo.SelectedIndex];
                S.Background = b.Color;
                S.Transparent = b.Transparent;
                RequestRebuild();
            };
            p.Children.Add(Ui.Row("Background", bgCombo));

            textColorCombo = new ComboBox { Width = 150 };
            foreach (var t in new[] { "Auto", "Black", "White", "Gray" }) textColorCombo.Items.Add(t);
            textColorCombo.SelectedIndex = 0;
            textColorCombo.SelectionChanged += (s, e) => { S.TextColorMode = textColorCombo.SelectedIndex; RequestRebuild(); };
            p.Children.Add(Ui.Row("Text color", textColorCombo));

            upscaleCheck = Check("Enlarge smaller images to fill tiles", S.Upscale, v => { S.Upscale = v; RequestRebuild(); });
            p.Children.Add(upscaleCheck);

            p.Children.Add(Ui.Section("File names"));
            captionsCheck = Check("Show file name under each image", S.ShowCaptions, v => { S.ShowCaptions = v; RequestRebuild(); });
            p.Children.Add(captionsCheck);
            captionFontBox = new IntBox((int)S.CaptionFont, 4, 2000, 2);
            captionFontBox.ValueChanged += v => { S.CaptionFont = v; userSized = true; RequestRebuild(); };
            p.Children.Add(Ui.Row("File name size (px)", captionFontBox));

            p.Children.Add(Ui.Section("Row & column labels"));
            colLabelsCheck = Check("Show column labels", S.ShowColLabels, v => { S.ShowColLabels = v; RequestRebuild(); });
            p.Children.Add(colLabelsCheck);
            rowLabelsCheck = Check("Show row labels", S.ShowRowLabels, v => { S.ShowRowLabels = v; RequestRebuild(); });
            p.Children.Add(rowLabelsCheck);
            labelFontBox = new IntBox((int)S.LabelFont, 4, 2000, 2);
            labelFontBox.ValueChanged += v => { S.LabelFont = v; userSized = true; RequestRebuild(); };
            p.Children.Add(Ui.Row("Label size (px)", labelFontBox));

            p.Children.Add(new TextBlock { Text = "Column labels (one per line, left → right)", Margin = new Thickness(0, 8, 0, 3) });
            colLabelsBox = LabelsBox();
            colLabelsBox.TextChanged += (s, e) => { S.ColLabels = SplitLines(colLabelsBox.Text); RequestRebuild(); };
            p.Children.Add(colLabelsBox);
            p.Children.Add(new TextBlock { Text = "Row labels (one per line, top → bottom)", Margin = new Thickness(0, 8, 0, 3) });
            rowLabelsBox = LabelsBox();
            rowLabelsBox.TextChanged += (s, e) => { S.RowLabels = SplitLines(rowLabelsBox.Text); RequestRebuild(); };
            p.Children.Add(rowLabelsBox);
            p.Children.Add(Ui.Dim(new TextBlock
            {
                Text = "Tip: double-click a label on the grid to edit it. Labels left empty take no space in the saved image.",
                FontSize = 12,
                Margin = new Thickness(0, 4, 0, 0),
                TextWrapping = TextWrapping.Wrap,
            }));

            var auto = Ui.Button("Auto-size text & padding", (s, e) => { AutoSize(); RequestRebuild(); });
            auto.Margin = new Thickness(0, 14, 0, 0);
            p.Children.Add(auto);

            p.Children.Add(Ui.Section("Output"));
            p.Children.Add(infoText);
            return p;
        }

        static CheckBox Check(string text, bool value, Action<bool> changed)
        {
            var c = new CheckBox { Content = text, IsChecked = value, Margin = new Thickness(0, 4, 0, 4) };
            c.Checked += (s, e) => changed(true);
            c.Unchecked += (s, e) => changed(false);
            return c;
        }

        static TextBox LabelsBox() => new TextBox
        {
            AcceptsReturn = true,
            Height = 96,
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            Padding = new Thickness(3),
        };

        static List<string> SplitLines(string text) =>
            text.Replace("\r", "").Split('\n').ToList();

        /// <summary>Scales text and padding to suit the image size so they're readable at 100%.</summary>
        void AutoSize()
        {
            if (items.Count == 0) return;
            double cellW = items.Max(i => i.PixelW), cellH = items.Max(i => i.PixelH);
            int caption = (int)Math.Max(12, Math.Min(600, Math.Round(cellH * 0.05)));
            S.CaptionFont = caption;
            S.LabelFont = Math.Round(caption * 1.25);
            S.Padding = (int)Math.Max(4, Math.Min(1000, Math.Round(Math.Max(cellW, cellH) * 0.03)));
            captionFontBox.SetSilently((int)S.CaptionFont);
            labelFontBox.SetSilently((int)S.LabelFont);
            paddingBox.SetSilently(S.Padding);
        }

        // ───────────────────────── rendering the editor ─────────────────────────

        /// <summary>
        /// Rebuilds once pending input has been handled, so typing in a label box or holding a spin button on a
        /// big grid runs one rebuild for the burst instead of one per keystroke.
        /// </summary>
        void RequestRebuild()
        {
            if (rebuildPending) return;
            rebuildPending = true;
            Dispatcher.BeginInvoke(DispatcherPriority.Background, new Action(() => { if (rebuildPending) Rebuild(); }));
        }

        void Rebuild()
        {
            if (canvasHolder == null) return; // still constructing
            rebuildPending = false;
            if (selected >= items.Count) selected = -1;
            // The tiles are about to be replaced, so a tile being pressed or dragged would point at the wrong image
            // (or past the end) once images finish loading or one is removed.
            if (dragIndex >= 0) EndDrag();
            layout = GridLayout.Compute(S, items.Select(i => new Size(i.PixelW, i.PixelH)).ToList());
            canvas.Children.Clear();
            tileImages.Clear();
            selRect = dropMarker = fileDropMarker = null;
            ghost = null;
            ScheduleSharpen();

            bool empty = items.Count == 0;
            emptyHint.Visibility = empty ? Visibility.Visible : Visibility.Collapsed;
            canvasHolder.Visibility = empty ? Visibility.Collapsed : Visibility.Visible;
            UpdateInfo();
            if (empty) return;

            canvas.Width = layout.Width;
            canvas.Height = layout.Height;
            canvas.Background = S.Transparent ? CheckerBrush() : Ui.Frozen(S.Background);
            var textBrush = Ui.Frozen(S.ResolveTextColor());
            var family = new FontFamily("Segoe UI");

            for (int i = 0; i < items.Count; i++)
            {
                var it = items[i];
                var r = layout.ImageRect(i, it.PixelW, it.PixelH);
                var img = new Image { Source = SourceFor(it), Width = r.Width, Height = r.Height, Stretch = Stretch.Fill };
                RenderOptions.SetBitmapScalingMode(img, BitmapScalingMode.HighQuality);
                Place(img, r.X, r.Y);
                tileImages.Add(img);
            }

            // File names are drawn in one pass, the way the export draws them: one TextBlock per tile made a rebuild
            // of a few hundred images take several times longer.
            if (S.ShowCaptions)
                Place(new CaptionLayer(layout, items.Select(i => i.Name).ToArray(), S.CaptionFont, textBrush), 0, 0);

            if (S.ShowColLabels)
                for (int c = 0; c < layout.Cols; c++)
                {
                    var hr = layout.ColHeaderRect(c);
                    Place(HeaderBlock(S.ColLabel(c), $"Column {c + 1}", hr, TextAlignment.Center, textBrush, family, true, c), hr.X, hr.Y);
                }

            if (S.ShowRowLabels)
                for (int r = 0; r < layout.Rows; r++)
                {
                    var hr = layout.RowHeaderRect(r);
                    var tb = HeaderBlock(S.RowLabel(r), $"Row {r + 1}", hr, TextAlignment.Right, textBrush, family, false, r);
                    double lineH = layout.LabelLineH;
                    tb.Height = lineH;
                    Place(tb, hr.X, hr.Y + (hr.Height - lineH) / 2);
                }

            if (selected >= items.Count) selected = -1;
            UpdateSelection();
            if (autoFit) FitZoom(); else UpdateStrokes();
        }

        TextBlock HeaderBlock(string text, string placeholder, Rect r, TextAlignment align, Brush brush,
            FontFamily family, bool isColumn, int index)
        {
            bool isEmpty = text.Length == 0;
            var tb = new TextBlock
            {
                Text = isEmpty ? placeholder : text,
                Width = r.Width,
                Height = r.Height,
                FontFamily = family,
                FontSize = S.LabelFont,
                FontWeight = FontWeights.SemiBold,
                FontStyle = isEmpty ? FontStyles.Italic : FontStyles.Normal,
                Opacity = isEmpty ? 0.3 : 1, // placeholders are on-screen only, never exported
                Foreground = brush,
                TextAlignment = align,
                TextTrimming = TextTrimming.CharacterEllipsis,
                Background = Brushes.Transparent,
                Cursor = Cursors.Hand,
                ToolTip = $"Double-click to edit {(isColumn ? "column" : "row")} {index + 1} label",
            };
            tb.MouseLeftButtonDown += (s, e) =>
            {
                e.Handled = true;
                if (e.ClickCount == 2) EditLabel(isColumn, index);
            };
            return tb;
        }

        void EditLabel(bool isColumn, int index)
        {
            var current = isColumn ? S.ColLabel(index) : S.RowLabel(index);
            var result = InputDialog.Ask(this, "Edit label",
                $"{(isColumn ? "Column" : "Row")} {index + 1} label:", current);
            if (result == null) return;

            var list = new List<string>(isColumn ? S.ColLabels : S.RowLabels);
            while (list.Count <= index) list.Add("");
            list[index] = result.Replace("\r", "").Replace("\n", " ");
            while (list.Count > 0 && list[list.Count - 1].Trim().Length == 0) list.RemoveAt(list.Count - 1);
            (isColumn ? colLabelsBox : rowLabelsBox).Text = string.Join(Environment.NewLine, list);
        }

        void Place(UIElement el, double x, double y)
        {
            Canvas.SetLeft(el, x);
            Canvas.SetTop(el, y);
            canvas.Children.Add(el);
        }

        Brush CheckerBrush()
        {
            double size = Math.Max(8, Math.Max(layout.Width, layout.Height) / 80);
            var g = new DrawingGroup();
            g.Children.Add(new GeometryDrawing(Brushes.White, null, new RectangleGeometry(new Rect(0, 0, 2, 2))));
            var gray = Ui.Frozen(Color.FromRgb(0xDD, 0xDD, 0xDD));
            g.Children.Add(new GeometryDrawing(gray, null, new RectangleGeometry(new Rect(0, 0, 1, 1))));
            g.Children.Add(new GeometryDrawing(gray, null, new RectangleGeometry(new Rect(1, 1, 1, 1))));
            return new DrawingBrush(g)
            {
                TileMode = TileMode.Tile,
                Viewport = new Rect(0, 0, size * 2, size * 2),
                ViewportUnits = BrushMappingMode.Absolute,
            };
        }

        void UpdateInfo()
        {
            if (items.Count == 0)
            {
                infoText.Text = "No images yet.";
                return;
            }
            // The saved image can be smaller than the editor's canvas: empty label sets take no space in it.
            var saved = GridLayout.Compute(S.ForExport(items.Count), items.Select(i => new Size(i.PixelW, i.PixelH)).ToList());
            infoText.Text =
                $"{items.Count} image{(items.Count == 1 ? "" : "s")} · {layout.Cols} × {layout.Rows} grid\n" +
                $"Tile: {layout.CellW:0} × {layout.CellH:0} px\n" +
                $"Full-size output: {Math.Round(saved.Width):N0} × {Math.Round(saved.Height):N0} px";
        }

        // ───────────────────────── zoom ─────────────────────────

        double Zoom => zoomT.ScaleX;

        void SetZoom(double z, bool fromUser)
        {
            z = Math.Max(0.01, Math.Min(4, z));
            zoomT.ScaleX = zoomT.ScaleY = z;
            suppressZoom = true;
            zoomSlider.Value = Math.Log(z);
            suppressZoom = false;
            zoomText.Text = $"{z * 100:0.#}%";
            if (fromUser) autoFit = false;
            UpdateStrokes();
            ScheduleSharpen();
        }

        void FitZoom()
        {
            if (layout == null || items.Count == 0 || scroller.ActualWidth <= 0) return;
            double vw = scroller.ActualWidth - 64, vh = scroller.ActualHeight - 64;
            if (vw <= 0 || vh <= 0) return;
            SetZoom(Math.Min(1, Math.Min(vw / layout.Width, vh / layout.Height)), false);
        }

        void Scroller_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            if ((Keyboard.Modifiers & ModifierKeys.Control) == 0) return;
            SetZoom(Zoom * (e.Delta > 0 ? 1.15 : 1 / 1.15), true);
            e.Handled = true;
        }

        /// <summary>Keeps selection/drop outlines a constant on-screen thickness regardless of zoom.</summary>
        void UpdateStrokes()
        {
            double t = 3 / Zoom;
            if (selRect != null)
            {
                selRect.StrokeThickness = t;
                PositionOutline(selRect, selected, t);
            }
            if (dropMarker != null) dropMarker.StrokeThickness = t;
        }

        // ───────────────────────── sharper tiles when zoomed in ─────────────────────────
        // Each image keeps only a small preview (ImageLoader.ThumbSide). Tiles near the visible area that are shown
        // bigger than their preview get a sharper copy loaded in the background; the copy is dropped again when
        // the tile scrolls away or shrinks, so memory stays around one screenful.

        BitmapSource SourceFor(ImageItem it) => sharp.TryGetValue(it, out var b) ? b : it.Thumb;

        void ScheduleSharpen()
        {
            if (sharpenTimer == null) return; // still constructing
            sharpenTimer.Stop();
            sharpenTimer.Start();
        }

        void SetTileSource(ImageItem it, BitmapSource src)
        {
            int i = items.IndexOf(it);
            if (i >= 0 && i < tileImages.Count) tileImages[i].Source = src;
        }

        void RefreshSharpness()
        {
            sharpenTimer.Stop();
            var want = new Dictionary<ImageItem, int>(); // tile → on-screen width in device pixels
            if (layout != null && items.Count > 0 && canvasHolder.IsVisible && scroller.ViewportWidth > 0)
            {
                Rect view;
                try
                {
                    view = scroller.TransformToDescendant(canvas)
                        .TransformBounds(new Rect(0, 0, scroller.ViewportWidth, scroller.ViewportHeight));
                }
                catch (InvalidOperationException) { return; }
                view.Inflate(view.Width * 0.25, view.Height * 0.25); // load a little ahead of scrolling
                double dpi = VisualTreeHelper.GetDpi(this).DpiScaleX;
                for (int i = 0; i < items.Count; i++)
                {
                    var it = items[i];
                    var r = layout.ImageRect(i, it.PixelW, it.PixelH);
                    if (!r.IntersectsWith(view) || it.Thumb == null || it.Thumb.PixelWidth >= it.PixelW) continue;
                    int need = (int)Math.Ceiling(r.Width * Zoom * dpi);
                    // A full-size copy is as sharp as the tile gets, even shown bigger than that; asking for more made
                    // every scroll or zoom reload it.
                    if (need > it.Thumb.PixelWidth) want[it] = Math.Min(need, it.PixelW);
                }
            }

            foreach (var it in sharp.Keys.Where(k => !want.ContainsKey(k)).ToList())
            {
                sharp.Remove(it);
                SetTileSource(it, it.Thumb);
            }

            // Load with headroom so the next few zoom steps don't each need a reload; swap a copy that's far bigger
            // than needed (after zooming out a lot) for a smaller one.
            var jobs = want
                .Where(kv => !sharp.TryGetValue(kv.Key, out var b) || b.PixelWidth < kv.Value || b.PixelWidth > kv.Value * 4)
                .Select(kv => (item: kv.Key, width: Math.Min(kv.Key.PixelW, (int)Math.Ceiling(kv.Value * 1.5))))
                .ToList();
            sharpenCancel?.Cancel();
            sharpenCancel = null;
            if (jobs.Count == 0) return;
            var cts = sharpenCancel = new CancellationTokenSource();
            var cancel = cts.Token;
            Renderer.RunSta(() =>
            {
                foreach (var (item, width) in jobs)
                {
                    if (cancel.IsCancellationRequested) break;
                    BitmapSource bmp;
                    try { bmp = ImageLoader.Load(item.Path, item.Orientation, width >= item.PixelW ? 0 : width); }
                    catch (Exception) { continue; } // moved or unreadable since it was added: keep the preview
                    Dispatcher.BeginInvoke(new Action(() =>
                    {
                        if (cancel.IsCancellationRequested || !items.Contains(item)) return;
                        sharp[item] = bmp;
                        SetTileSource(item, bmp);
                    }));
                }
                return true;
            });
        }

        // ───────────────────────── selection ─────────────────────────

        void Select(int idx)
        {
            selected = idx;
            UpdateSelection();
        }

        void UpdateSelection()
        {
            if (selected < 0)
            {
                if (selRect != null) selRect.Visibility = Visibility.Collapsed;
                return;
            }
            if (selRect == null)
            {
                selRect = new Rectangle { Stroke = Ui.Red, IsHitTestVisible = false };
                Panel.SetZIndex(selRect, 50);
                canvas.Children.Add(selRect);
            }
            selRect.Visibility = Visibility.Visible;
            selRect.StrokeThickness = 3 / Zoom;
            PositionOutline(selRect, selected, selRect.StrokeThickness);
            statusText.Text = $"Selected: {items[selected].Name}  ({items[selected].PixelW} × {items[selected].PixelH} px)  —  {items[selected].Path}";
        }

        void PositionOutline(Rectangle rect, int idx, double thickness)
        {
            if (idx < 0 || layout == null) return;
            var r = layout.CellRect(idx);
            r.Height += layout.CaptionH;
            r.Inflate(thickness + 2 / Zoom, thickness + 2 / Zoom);
            rect.Width = r.Width;
            rect.Height = r.Height;
            Canvas.SetLeft(rect, r.X);
            Canvas.SetTop(rect, r.Y);
        }

        // ───────────────────────── drag to rearrange ─────────────────────────

        void Canvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            canvas.Focus();
            int idx = layout.HitTest(e.GetPosition(canvas));
            Select(idx);
            if (idx < 0) return;
            dragIndex = idx;
            dragStart = e.GetPosition(this);
            canvas.CaptureMouse();
        }

        void Canvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (dragIndex < 0 || e.LeftButton != MouseButtonState.Pressed) return;
            if (!dragging)
            {
                if ((e.GetPosition(this) - dragStart).Length < 6) return;
                BeginDrag();
            }

            var p = e.GetPosition(canvas);
            Canvas.SetLeft(ghost, p.X - ghost.Width / 2);
            Canvas.SetTop(ghost, p.Y - ghost.Height / 2);
            int target = layout.DropIndex(p, items.Count - 1);
            PositionOutline(dropMarker, target, dropMarker.StrokeThickness);
            AutoScroll(e.GetPosition(scroller));
        }

        void BeginDrag()
        {
            dragging = true;
            var it = items[dragIndex];
            var r = layout.ImageRect(dragIndex, it.PixelW, it.PixelH);
            ghost = new Image
            {
                Source = it.Thumb,
                Width = r.Width * 0.6,
                Height = r.Height * 0.6,
                Stretch = Stretch.Fill,
                Opacity = 0.75,
                IsHitTestVisible = false,
            };
            Panel.SetZIndex(ghost, 100);
            canvas.Children.Add(ghost);

            dropMarker = NewDropMarker();
            canvas.Children.Add(dropMarker);
            tileImages[dragIndex].Opacity = 0.3;
            if (selRect != null) selRect.Visibility = Visibility.Collapsed;
            Cursor = Cursors.SizeAll;
        }

        /// <summary>The dashed red slot outline shown where a dragged tile or dropped files will go.</summary>
        Rectangle NewDropMarker()
        {
            var r = new Rectangle
            {
                Stroke = Ui.Red,
                StrokeThickness = 3 / Zoom,
                StrokeDashArray = new DoubleCollection { 3, 2 },
                Fill = Ui.Frozen(Color.FromArgb(0x30, 237, 0, 13)),
                IsHitTestVisible = false,
            };
            Panel.SetZIndex(r, 90);
            return r;
        }

        void AutoScroll(Point p)
        {
            const double edge = 40, step = 25;
            if (p.X < edge) scroller.ScrollToHorizontalOffset(scroller.HorizontalOffset - step);
            else if (p.X > scroller.ViewportWidth - edge) scroller.ScrollToHorizontalOffset(scroller.HorizontalOffset + step);
            if (p.Y < edge) scroller.ScrollToVerticalOffset(scroller.VerticalOffset - step);
            else if (p.Y > scroller.ViewportHeight - edge) scroller.ScrollToVerticalOffset(scroller.VerticalOffset + step);
        }

        void Canvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (dragIndex < 0) return;
            int from = dragIndex;
            bool wasDragging = dragging;
            int target = layout.DropIndex(e.GetPosition(canvas), items.Count - 1);
            EndDrag();
            if (wasDragging && target != from)
            {
                var it = items[from];
                items.RemoveAt(from);
                items.Insert(target, it);
                selected = target;
                Rebuild();
                statusText.Text = $"Moved \"{it.Name}\" to position {target + 1}.";
            }
            else if (wasDragging) Rebuild();
        }

        void EndDrag()
        {
            dragIndex = -1;
            dragging = false;
            Cursor = loading > 0 ? Cursors.Wait : null;
            if (ghost != null) { canvas.Children.Remove(ghost); ghost = null; }
            if (dropMarker != null) { canvas.Children.Remove(dropMarker); dropMarker = null; }
            foreach (var img in tileImages) img.Opacity = 1;
            if (canvas.IsMouseCaptured) canvas.ReleaseMouseCapture();
            UpdateSelection();
        }

        void Canvas_MouseRightButtonUp(object sender, MouseButtonEventArgs e)
        {
            int idx = layout.HitTest(e.GetPosition(canvas));
            if (idx < 0) return;
            Select(idx);
            var menu = new ContextMenu();
            var remove = new MenuItem { Header = "Remove from grid", InputGestureText = "Del" };
            remove.Click += (s, a) => RemoveSelected();
            var show = new MenuItem { Header = "Show in File Explorer" };
            var path = items[idx].Path;
            show.Click += (s, a) => Process.Start("explorer.exe", $"/select,\"{path}\"");
            menu.Items.Add(remove);
            menu.Items.Add(show);
            menu.IsOpen = true;
            e.Handled = true;
        }

        // ───────────────────────── adding / removing images ─────────────────────────

        void BrowseForImages()
        {
            var dlg = new OpenFileDialog
            {
                Title = "Add images",
                Multiselect = true,
                Filter = "Images|" + string.Join(";", ImageLoader.Extensions.Select(x => "*" + x)) + "|All files|*.*",
            };
            if (dlg.ShowDialog(this) == true) AddFiles(dlg.FileNames);
        }

        void Window_DragOver(object sender, DragEventArgs e)
        {
            bool files = e.Data.GetDataPresent(DataFormats.FileDrop);
            e.Effects = files ? DragDropEffects.Copy : DragDropEffects.None;
            e.Handled = true;
            ShowFileDropMarker(files ? FileDropIndex(e) : -1);
        }

        void Window_Drop(object sender, DragEventArgs e)
        {
            ShowFileDropMarker(-1);
            if (!(e.Data.GetData(DataFormats.FileDrop) is string[] paths)) return;
            AddFiles(paths, FileDropIndex(e));
        }

        /// <summary>
        /// The grid slot dropped files go into, or -1 (the end) when they aren't dropped on the grid. Uses the
        /// element the drop lands on: IsMouseOver isn't updated while files are dragged in from another app.
        /// </summary>
        int FileDropIndex(DragEventArgs e)
        {
            if (items.Count == 0 || layout == null || !(e.OriginalSource is Visual v)) return -1;
            if (v != canvas && !canvas.IsAncestorOf(v)) return -1;
            return layout.DropIndex(e.GetPosition(canvas), items.Count);
        }

        void ShowFileDropMarker(int idx)
        {
            if (idx < 0)
            {
                if (fileDropMarker != null) fileDropMarker.Visibility = Visibility.Collapsed;
                return;
            }
            if (fileDropMarker == null)
            {
                fileDropMarker = NewDropMarker();
                canvas.Children.Add(fileDropMarker);
            }
            fileDropMarker.Visibility = Visibility.Visible;
            fileDropMarker.StrokeThickness = 3 / Zoom;
            PositionOutline(fileDropMarker, idx, fileDropMarker.StrokeThickness);
        }

        public async void AddFiles(IEnumerable<string> paths, int insertAt = -1)
        {
            var dropped = paths.ToArray();
            // The tile the new images go in front of (null: the end), so they still land there if the grid changes
            // while they load.
            ImageItem before = insertAt >= 0 && insertAt < items.Count ? items[insertAt] : null;
            LoadingStarted();
            try
            {
                statusText.Text = "Looking for images…";
                // Off the UI thread: listing a big or network folder can take a while.
                var (files, failed) = await Task.Run(() => ImageLoader.FindImages(dropped));
                var ok = new List<ImageItem>();
                if (files.Count > 0)
                {
                    statusText.Text = $"Loading {files.Count} image{(files.Count == 1 ? "" : "s")}…";
                    int step = Math.Max(1, files.Count / 50); // about 50 progress updates at most
                    var result = await ImageLoader.ReadInfoMany(files, done =>
                    {
                        if (done % step == 0 && done < files.Count)
                            Dispatcher.BeginInvoke(new Action(() => statusText.Text = $"Loading images… {done} of {files.Count}"));
                    });
                    ok = result.ok;
                    failed.AddRange(result.failed);
                }

                if (ok.Count > 0)
                {
                    bool wasEmpty = items.Count == 0;
                    int at = before != null ? items.IndexOf(before) : items.Count;
                    if (at < 0) at = Math.Min(insertAt, items.Count); // that tile was removed meanwhile
                    var sel = selected >= 0 && selected < items.Count ? items[selected] : null;
                    items.InsertRange(at, ok);
                    selected = sel == null ? -1 : items.IndexOf(sel); // the same tile stays selected
                    if (wasEmpty)
                    {
                        if (!userSized) AutoSize();
                        autoFit = true;
                    }
                    Rebuild();
                }

                statusText.Text = files.Count == 0 && failed.Count == 0
                    ? "No supported image files found."
                    : $"Added {ok.Count} image{(ok.Count == 1 ? "" : "s")}." +
                      (failed.Count > 0 ? $" {failed.Count} could not be opened." : "");
                if (failed.Count > 0)
                    MessageBox.Show(this, "These could not be opened:\n\n" + string.Join("\n", failed.Take(15)) +
                        (failed.Count > 15 ? $"\n…and {failed.Count - 15} more" : ""),
                        "RulZ Image Grid", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
            finally { LoadingEnded(); }
        }

        int loading; // AddFiles calls still reading files; they can overlap

        void LoadingStarted()
        {
            if (loading++ == 0 && dragIndex < 0) Cursor = Cursors.Wait;
        }

        void LoadingEnded()
        {
            if (--loading == 0 && dragIndex < 0) Cursor = null;
        }

        void RemoveSelected()
        {
            if (selected < 0 || selected >= items.Count) return;
            var name = items[selected].Name;
            items.RemoveAt(selected);
            selected = items.Count == 0 ? -1 : Math.Min(selected, items.Count - 1);
            Rebuild();
            statusText.Text = $"Removed \"{name}\".";
        }

        void SortByName()
        {
            var sel = selected >= 0 ? items[selected] : null;
            var sorted = items.OrderBy(i => i.Name, new NaturalComparer()).ToList();
            items.Clear();
            items.AddRange(sorted);
            selected = sel == null ? -1 : items.IndexOf(sel);
            Rebuild();
        }

        void ClearAll()
        {
            if (items.Count == 0) return;
            if (MessageBox.Show(this, "Remove all images from the grid?", "RulZ Image Grid",
                    MessageBoxButton.OKCancel, MessageBoxImage.Question) != MessageBoxResult.OK) return;
            items.Clear();
            selected = -1;
            Rebuild();
            statusText.Text = "Cleared.";
        }

        void ShowExport()
        {
            if (items.Count == 0)
            {
                MessageBox.Show(this, "Add some images first.", "RulZ Image Grid", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }
            var win = new ExportWindow(this, S.ForExport(items.Count), items.ToList());
            if (win.ShowDialog() == true)
                statusText.Text = $"Saved {win.SavedSize} image to {win.SavedPath}";
        }

        void Window_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            bool ctrl = (Keyboard.Modifiers & ModifierKeys.Control) != 0;
            if (ctrl && e.Key == Key.O) { BrowseForImages(); e.Handled = true; return; }
            if (ctrl && e.Key == Key.E) { ShowExport(); e.Handled = true; return; }
            if (e.Key == Key.Escape && dragIndex >= 0) { EndDrag(); Rebuild(); e.Handled = true; return; }
            if (Keyboard.FocusedElement is TextBox) return;
            if (e.Key == Key.Delete)
            {
                if (dragIndex < 0) RemoveSelected(); // not while a tile is held down
                e.Handled = true;
            }
        }
    }

    /// <summary>Every file name under the tiles, drawn by the same code as the export so the two match.</summary>
    sealed class CaptionLayer : FrameworkElement
    {
        readonly GridLayout layout;
        readonly string[] names;
        readonly double size;
        readonly Brush brush;

        public CaptionLayer(GridLayout layout, string[] names, double size, Brush brush)
        {
            this.layout = layout;
            this.names = names;
            this.size = size;
            this.brush = brush;
            Width = layout.Width;
            Height = layout.Height;
            IsHitTestVisible = false;
        }

        protected override void OnRender(DrawingContext dc)
        {
            double ppd = VisualTreeHelper.GetDpi(this).PixelsPerDip;
            for (int i = 0; i < names.Length; i++) TextMetrics.DrawCaption(dc, layout, i, names[i], size, brush, ppd);
        }

        // Moving to a monitor with other scaling only re-measures elements, and this one's size never changes, so ask
        // for a redraw to lay the text out for the new monitor.
        protected override void OnDpiChanged(DpiScale oldDpi, DpiScale newDpi) => InvalidateVisual();
    }

    /// <summary>Sorts "img2" before "img10".</summary>
    class NaturalComparer : IComparer<string>
    {
        [System.Runtime.InteropServices.DllImport("shlwapi.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode)]
        static extern int StrCmpLogicalW(string a, string b);
        public int Compare(string a, string b) => StrCmpLogicalW(a, b);
    }
}
