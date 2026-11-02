using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
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
        readonly List<Image> tileImages = new List<Image>();
        Rectangle selRect, dropMarker;
        Image ghost;

        // settings controls
        IntBox columnsBox, paddingBox, captionFontBox, labelFontBox;
        CheckBox captionsCheck, colLabelsCheck, rowLabelsCheck, upscaleCheck, lightCheck;
        TextBox colLabelsBox, rowLabelsBox;
        ComboBox bgCombo, textColorCombo;

        // state
        int selected = -1;
        bool autoFit = true, suppressZoom, userSized;
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

            // A dashed drop zone with the logo, shown until images are added.
            var dash = new Rectangle { StrokeThickness = 2, StrokeDashArray = new DoubleCollection { 4, 3 }, RadiusX = 12, RadiusY = 12 };
            dash.SetResourceReference(Shape.StrokeProperty, "Border");
            var hintStack = new StackPanel { Margin = new Thickness(64, 40, 64, 40), HorizontalAlignment = HorizontalAlignment.Center };
            var hintLogo = new Image { Height = 40, Opacity = 0.9, Margin = new Thickness(0, 0, 0, 18), HorizontalAlignment = HorizontalAlignment.Center };
            RenderOptions.SetBitmapScalingMode(hintLogo, BitmapScalingMode.HighQuality);
            hintLogo.Source = logoImage.Source;
            Theme.Changed += () => hintLogo.Source = Theme.LoadLogo(Theme.Current);
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
            Drop += Window_Drop;
            PreviewKeyDown += Window_PreviewKeyDown;
            Theme.Changed += OnThemeChanged;
            Closed += (s, e) => Theme.Changed -= OnThemeChanged;

            SetZoom(1, false);
            Rebuild();
        }

        void OnThemeChanged()
        {
            logoImage.Source = Theme.LoadLogo(Theme.Current);
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
            columnsBox.ValueChanged += v => { S.Columns = v; Rebuild(); };
            p.Children.Add(Ui.Row("Columns", columnsBox));
            paddingBox = new IntBox(S.Padding, 0, 5000, 2);
            paddingBox.ValueChanged += v => { S.Padding = v; userSized = true; Rebuild(); };
            p.Children.Add(Ui.Row("Padding (px)", paddingBox));

            bgCombo = new ComboBox { Width = 150 };
            foreach (var b in Backgrounds) bgCombo.Items.Add(b.Name);
            bgCombo.SelectedIndex = 0;
            bgCombo.SelectionChanged += (s, e) =>
            {
                var b = Backgrounds[bgCombo.SelectedIndex];
                S.Background = b.Color;
                S.Transparent = b.Transparent;
                Rebuild();
            };
            p.Children.Add(Ui.Row("Background", bgCombo));

            textColorCombo = new ComboBox { Width = 150 };
            foreach (var t in new[] { "Auto", "Black", "White", "Gray" }) textColorCombo.Items.Add(t);
            textColorCombo.SelectedIndex = 0;
            textColorCombo.SelectionChanged += (s, e) => { S.TextColorMode = textColorCombo.SelectedIndex; Rebuild(); };
            p.Children.Add(Ui.Row("Text color", textColorCombo));

            upscaleCheck = Check("Enlarge smaller images to fill tiles", S.Upscale, v => { S.Upscale = v; Rebuild(); });
            p.Children.Add(upscaleCheck);

            p.Children.Add(Ui.Section("File names"));
            captionsCheck = Check("Show file name under each image", S.ShowCaptions, v => { S.ShowCaptions = v; Rebuild(); });
            p.Children.Add(captionsCheck);
            captionFontBox = new IntBox((int)S.CaptionFont, 4, 2000, 2);
            captionFontBox.ValueChanged += v => { S.CaptionFont = v; userSized = true; Rebuild(); };
            p.Children.Add(Ui.Row("File name size (px)", captionFontBox));

            p.Children.Add(Ui.Section("Row & column labels"));
            colLabelsCheck = Check("Show column labels", S.ShowColLabels, v => { S.ShowColLabels = v; Rebuild(); });
            p.Children.Add(colLabelsCheck);
            rowLabelsCheck = Check("Show row labels", S.ShowRowLabels, v => { S.ShowRowLabels = v; Rebuild(); });
            p.Children.Add(rowLabelsCheck);
            labelFontBox = new IntBox((int)S.LabelFont, 4, 2000, 2);
            labelFontBox.ValueChanged += v => { S.LabelFont = v; userSized = true; Rebuild(); };
            p.Children.Add(Ui.Row("Label size (px)", labelFontBox));

            p.Children.Add(new TextBlock { Text = "Column labels (one per line, left → right)", Margin = new Thickness(0, 8, 0, 3) });
            colLabelsBox = LabelsBox();
            colLabelsBox.TextChanged += (s, e) => { S.ColLabels = SplitLines(colLabelsBox.Text); Rebuild(); };
            p.Children.Add(colLabelsBox);
            p.Children.Add(new TextBlock { Text = "Row labels (one per line, top → bottom)", Margin = new Thickness(0, 8, 0, 3) });
            rowLabelsBox = LabelsBox();
            rowLabelsBox.TextChanged += (s, e) => { S.RowLabels = SplitLines(rowLabelsBox.Text); Rebuild(); };
            p.Children.Add(rowLabelsBox);
            p.Children.Add(Ui.Dim(new TextBlock
            {
                Text = "Tip: double-click a label on the grid to edit it.",
                FontSize = 12,
                Margin = new Thickness(0, 4, 0, 0),
                TextWrapping = TextWrapping.Wrap,
            }));

            var auto = Ui.Button("Auto-size text & padding", (s, e) => { AutoSize(); Rebuild(); });
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

        void Rebuild()
        {
            if (canvasHolder == null) return; // still constructing
            layout = GridLayout.Compute(S, items.Select(i => new Size(i.PixelW, i.PixelH)).ToList());
            canvas.Children.Clear();
            tileImages.Clear();
            selRect = dropMarker = null;
            ghost = null;

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
                var img = new Image { Source = it.Thumb, Width = r.Width, Height = r.Height, Stretch = Stretch.Fill };
                RenderOptions.SetBitmapScalingMode(img, BitmapScalingMode.HighQuality);
                Place(img, r.X, r.Y);
                tileImages.Add(img);

                if (S.ShowCaptions)
                {
                    var cr = layout.CaptionRect(i);
                    Place(new TextBlock
                    {
                        Text = it.Name,
                        Width = cr.Width,
                        FontFamily = family,
                        FontSize = S.CaptionFont,
                        Foreground = textBrush,
                        TextAlignment = TextAlignment.Center,
                        TextTrimming = TextTrimming.CharacterEllipsis,
                        IsHitTestVisible = false,
                    }, cr.X, cr.Y);
                }
            }

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
            infoText.Text =
                $"{items.Count} image{(items.Count == 1 ? "" : "s")} · {layout.Cols} × {layout.Rows} grid\n" +
                $"Tile: {layout.CellW:0} × {layout.CellH:0} px\n" +
                $"Full-size output: {Math.Round(layout.Width):N0} × {Math.Round(layout.Height):N0} px";
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

            dropMarker = new Rectangle
            {
                Stroke = Ui.Red,
                StrokeThickness = 3 / Zoom,
                StrokeDashArray = new DoubleCollection { 3, 2 },
                Fill = Ui.Frozen(Color.FromArgb(0x30, 237, 0, 13)),
                IsHitTestVisible = false,
            };
            Panel.SetZIndex(dropMarker, 90);
            canvas.Children.Add(dropMarker);
            tileImages[dragIndex].Opacity = 0.3;
            if (selRect != null) selRect.Visibility = Visibility.Collapsed;
            Cursor = Cursors.SizeAll;
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
            Cursor = null;
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
            e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop) ? DragDropEffects.Copy : DragDropEffects.None;
            e.Handled = true;
        }

        void Window_Drop(object sender, DragEventArgs e)
        {
            if (!(e.Data.GetData(DataFormats.FileDrop) is string[] paths)) return;
            int insertAt = -1;
            if (items.Count > 0 && canvas.IsMouseOver)
                insertAt = layout.DropIndex(e.GetPosition(canvas), items.Count);
            AddFiles(paths, insertAt);
        }

        public async void AddFiles(IEnumerable<string> paths, int insertAt = -1)
        {
            var files = new List<string>();
            foreach (var p in paths)
            {
                if (Directory.Exists(p))
                    files.AddRange(Directory.GetFiles(p).Where(ImageLoader.IsImage).OrderBy(f => f, StringComparer.CurrentCultureIgnoreCase));
                else if (File.Exists(p) && ImageLoader.IsImage(p))
                    files.Add(p);
            }
            if (files.Count == 0)
            {
                statusText.Text = "No supported image files found.";
                return;
            }

            statusText.Text = $"Loading {files.Count} image{(files.Count == 1 ? "" : "s")}…";
            Cursor = Cursors.Wait;
            var result = await Renderer.RunSta(() =>
            {
                var ok = new List<ImageItem>();
                var failed = new List<string>();
                foreach (var f in files)
                {
                    try { ok.Add(ImageLoader.ReadInfo(f)); }
                    catch (Exception ex) { failed.Add($"{System.IO.Path.GetFileName(f)}: {ex.Message}"); }
                }
                return (ok, failed);
            });
            Cursor = null;

            bool wasEmpty = items.Count == 0;
            if (insertAt < 0 || insertAt > items.Count) insertAt = items.Count;
            items.InsertRange(insertAt, result.ok);
            if (wasEmpty && result.ok.Count > 0)
            {
                if (!userSized) AutoSize();
                autoFit = true;
            }
            Rebuild();

            statusText.Text = $"Added {result.ok.Count} image{(result.ok.Count == 1 ? "" : "s")}." +
                (result.failed.Count > 0 ? $" {result.failed.Count} could not be opened." : "");
            if (result.failed.Count > 0)
                MessageBox.Show(this, "These files could not be opened:\n\n" + string.Join("\n", result.failed.Take(15)),
                    "RulZ Image Grid", MessageBoxButton.OK, MessageBoxImage.Warning);
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
            var win = new ExportWindow(this, S.Clone(), items.ToList());
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
            if (e.Key == Key.Delete) { RemoveSelected(); e.Handled = true; }
        }
    }

    /// <summary>Sorts "img2" before "img10".</summary>
    class NaturalComparer : IComparer<string>
    {
        [System.Runtime.InteropServices.DllImport("shlwapi.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode)]
        static extern int StrCmpLogicalW(string a, string b);
        public int Compare(string a, string b) => StrCmpLogicalW(a, b);
    }
}
