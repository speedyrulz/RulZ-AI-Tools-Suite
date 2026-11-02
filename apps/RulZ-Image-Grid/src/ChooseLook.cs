using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Media;

namespace RulZImageGrid
{
    // The first-run "Choose your look" prompt: two cards, Dark (preselected) and Light, and Continue.
    // The dialog itself previews the selected look. Closing it without pressing Continue means Dark.
    public sealed class ChooseLookDialog : Window
    {
        readonly Border darkCard, lightCard;
        Palette selected = Palette.Dark;
        bool confirmed;

        // Returns the chosen palette (Dark unless Light was picked and Continue pressed).
        public static Palette Ask()
        {
            var d = new ChooseLookDialog();
            d.ShowDialog();
            return d.confirmed ? d.selected : Palette.Dark;
        }

        ChooseLookDialog()
        {
            Theme.Attach(this);
            Title = "RulZ Image Grid";
            Icon = Theme.AppIcon();
            SizeToContent = SizeToContent.WidthAndHeight;
            ResizeMode = ResizeMode.NoResize;
            WindowStartupLocation = WindowStartupLocation.CenterScreen;

            var root = new StackPanel { Margin = new Thickness(28, 22, 28, 22) };
            root.Children.Add(new TextBlock { Text = "Choose your look", FontSize = 22, FontWeight = FontWeights.Bold });

            var cards = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 16, 0, 0) };
            darkCard = Card(Palette.Dark, "Dark  (default)");
            lightCard = Card(Palette.Light, "Light");
            lightCard.Margin = new Thickness(16, 0, 0, 0);
            cards.Children.Add(darkCard);
            cards.Children.Add(lightCard);
            root.Children.Add(cards);

            root.Children.Add(Ui.Dim(new TextBlock
            {
                Text = "You can change this later with the \"Light mode\" checkbox at the right end of the toolbar.",
                TextWrapping = TextWrapping.Wrap,
                MaxWidth = 520,
                HorizontalAlignment = HorizontalAlignment.Left,
                Margin = new Thickness(0, 16, 0, 0),
            }));

            var go = Ui.Button("Continue", (s, e) => Finish(), true);
            go.IsDefault = true;
            go.MinWidth = 120;
            go.Padding = new Thickness(16, 7, 16, 7);
            go.Margin = new Thickness(0, 18, 0, 0);
            go.HorizontalAlignment = HorizontalAlignment.Right;
            root.Children.Add(go);
            Content = root;

            PreviewKeyDown += (s, e) =>
            {
                if (e.Key == Key.Left) { Pick(Palette.Dark); e.Handled = true; }
                else if (e.Key == Key.Right) { Pick(Palette.Light); e.Handled = true; }
                else if (e.Key == Key.Escape) Close();
            };
            Loaded += (s, e) => { Activate(); go.Focus(); };
            Pick(Palette.Dark);
        }

        void Pick(Palette p)
        {
            selected = p;
            Theme.Apply(p);
            Frame(darkCard, p == Palette.Dark);
            Frame(lightCard, p == Palette.Light);
        }

        static void Frame(Border card, bool on)
        {
            card.SetResourceReference(Border.BorderBrushProperty, on ? "Accent" : "Border");
            card.BorderThickness = new Thickness(on ? 3 : 1);
            card.Padding = new Thickness(on ? 12 : 14);
        }

        void Finish()
        {
            confirmed = true;
            Close();
        }

        // A selectable card with a tiny drawing of the app in its palette.
        Border Card(Palette look, string name)
        {
            var stack = new StackPanel();
            stack.Children.Add(Preview(look));
            stack.Children.Add(new TextBlock
            {
                Text = name,
                FontSize = 15,
                FontWeight = FontWeights.SemiBold,
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(0, 10, 0, 0),
            });
            var card = new Border { Child = stack, CornerRadius = new CornerRadius(8), Cursor = Cursors.Hand };
            card.SetResourceReference(Border.BackgroundProperty, "Panel");
            card.MouseLeftButtonDown += (s, e) =>
            {
                Pick(look);
                if (e.ClickCount == 2) Finish();
            };
            return card;
        }

        static FrameworkElement Preview(Palette p)
        {
            Brush B(Color c) => Ui.Frozen(c);

            var toolbar = new Border { Height = 24, Background = B(p.Panel), BorderBrush = B(p.Border), BorderThickness = new Thickness(0, 0, 0, 1) };
            var bar = new StackPanel { Orientation = Orientation.Horizontal, VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(6, 0, 0, 0) };
            var logo = Theme.LoadLogo(p);
            if (logo != null) bar.Children.Add(new Image { Source = logo, Height = 10, Margin = new Thickness(0, 0, 8, 0) });
            bar.Children.Add(new Border { Width = 26, Height = 10, CornerRadius = new CornerRadius(2), Background = B(p.AccentFill), Margin = new Thickness(0, 0, 4, 0) });
            bar.Children.Add(new Border { Width = 22, Height = 10, CornerRadius = new CornerRadius(2), Background = B(p.Control), BorderBrush = B(p.Border), BorderThickness = new Thickness(1) });
            toolbar.Child = bar;
            DockPanel.SetDock(toolbar, Dock.Top);

            var side = new StackPanel { Width = 58, Background = B(p.Panel) };
            for (int i = 0; i < 5; i++)
            {
                var row = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(6, i == 0 ? 8 : 5, 0, 0) };
                if (i % 2 == 1) row.Children.Add(new Border { Width = 6, Height = 6, CornerRadius = new CornerRadius(1), Background = B(p.Accent), Margin = new Thickness(0, 0, 4, 0) });
                row.Children.Add(new Border { Width = i % 2 == 1 ? 30 : 40, Height = 4, CornerRadius = new CornerRadius(2), Background = B(p.Dim) });
                side.Children.Add(row);
            }
            DockPanel.SetDock(side, Dock.Left);

            var tiles = new UniformGrid { Columns = 3, Rows = 2 };
            var colors = new[] { Color.FromRgb(240, 150, 80), Color.FromRgb(90, 170, 220), Color.FromRgb(120, 190, 110),
                                 Color.FromRgb(170, 110, 200), Color.FromRgb(230, 200, 90), Color.FromRgb(220, 100, 120) };
            foreach (var c in colors)
            {
                var t = new StackPanel { Margin = new Thickness(2) };
                t.Children.Add(new Border { Height = 18, Background = B(c) });
                t.Children.Add(new Border { Height = 2, Width = 14, Margin = new Thickness(0, 2, 0, 0), Background = B(Color.FromRgb(160, 160, 160)) });
                tiles.Children.Add(t);
            }
            var sheet = new Border { Background = Brushes.White, Padding = new Thickness(3), VerticalAlignment = VerticalAlignment.Center, Child = tiles };
            var workspace = new Border { Background = B(p.Workspace), Padding = new Thickness(10), Child = sheet };

            var dock = new DockPanel();
            dock.Children.Add(toolbar);
            dock.Children.Add(side);
            dock.Children.Add(workspace);
            return new Border
            {
                Width = 236,
                Height = 140,
                Background = B(p.Back),
                BorderBrush = B(p.Border),
                BorderThickness = new Thickness(1),
                Child = dock,
                ClipToBounds = true,
            };
        }
    }
}
