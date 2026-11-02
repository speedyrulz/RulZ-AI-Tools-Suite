using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RulZImageGrid
{
    public static class Ui
    {
        /// <summary>The RulZ logo red, for outlines drawn on the grid itself (same in both looks).</summary>
        public static readonly Brush Red = Frozen(Color.FromRgb(237, 0, 13));

        public static SolidColorBrush Frozen(Color c)
        {
            var b = new SolidColorBrush(c);
            b.Freeze();
            return b;
        }

        /// <summary>Points a property at one of the theme brushes (Back, Panel, Text, Dim, Accent, ...).</summary>
        public static T Res<T>(T el, DependencyProperty prop, string key) where T : FrameworkElement
        {
            el.SetResourceReference(prop, key);
            return el;
        }

        public static TextBlock Dim(TextBlock tb) => Res(tb, TextBlock.ForegroundProperty, "Dim");

        /// <summary>A section heading with the slanted red bar used across the RulZ apps.</summary>
        public static FrameworkElement Section(string text)
        {
            var p = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 18, 0, 8) };
            p.Children.Add(Res(new Border
            {
                Width = 4,
                Height = 12,
                Margin = new Thickness(1, 0, 8, 0),
                RenderTransform = new SkewTransform(-20, 0),
                RenderTransformOrigin = new Point(0.5, 0.5),
                VerticalAlignment = VerticalAlignment.Center,
            }, Border.BackgroundProperty, "Accent"));
            p.Children.Add(Dim(new TextBlock
            {
                Text = text.ToUpperInvariant(),
                FontSize = 11,
                FontWeight = FontWeights.SemiBold,
                VerticalAlignment = VerticalAlignment.Center,
            }));
            return p;
        }

        /// <summary>A thin vertical divider for toolbars.</summary>
        public static FrameworkElement Divider() => Res(new Border
        {
            Width = 1,
            Height = 22,
            Margin = new Thickness(8, 0, 14, 0),
            VerticalAlignment = VerticalAlignment.Center,
        }, Border.BackgroundProperty, "Border");

        public static FrameworkElement Row(string label, UIElement control)
        {
            var g = new Grid { Margin = new Thickness(0, 3, 0, 3) };
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            g.Children.Add(new TextBlock { Text = label, VerticalAlignment = VerticalAlignment.Center });
            Grid.SetColumn(control, 1);
            g.Children.Add(control);
            return g;
        }

        public static Button Button(string text, RoutedEventHandler click, bool primary = false)
        {
            var b = new Button
            {
                Content = text,
                Padding = new Thickness(12, 5, 12, 5),
                Margin = new Thickness(0, 0, 6, 0),
                MinWidth = 70,
            };
            if (primary) b.SetResourceReference(FrameworkElement.StyleProperty, "PrimaryButton");
            b.Click += click;
            return b;
        }
    }

    /// <summary>Integer text box with up/down buttons. Updates live as valid numbers are typed.</summary>
    public class IntBox : Grid
    {
        readonly TextBox box;
        readonly int min, max;
        int value;
        public event Action<int> ValueChanged;

        public IntBox(int value, int min, int max, int step = 1)
        {
            this.min = min;
            this.max = max;
            this.value = Clamp(value);
            ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });

            box = new TextBox
            {
                Text = this.value.ToString(),
                Width = 62,
                Height = 24,
                Padding = new Thickness(3, 1, 3, 1),
                VerticalContentAlignment = VerticalAlignment.Center,
                HorizontalContentAlignment = HorizontalAlignment.Right,
            };
            box.TextChanged += (s, e) =>
            {
                if (int.TryParse(box.Text, out var v) && v >= this.min && v <= this.max && v != this.value)
                {
                    this.value = v;
                    ValueChanged?.Invoke(v);
                }
            };
            box.LostFocus += (s, e) => box.Text = this.value.ToString();
            box.PreviewKeyDown += (s, e) =>
            {
                if (e.Key == Key.Up) { Value += step; e.Handled = true; }
                else if (e.Key == Key.Down) { Value -= step; e.Handled = true; }
                else if (e.Key == Key.Enter) box.Text = this.value.ToString();
            };
            box.PreviewMouseWheel += (s, e) =>
            {
                if (!box.IsKeyboardFocused) return;
                Value += e.Delta > 0 ? step : -step;
                e.Handled = true;
            };
            Children.Add(box);

            var spin = new StackPanel { Margin = new Thickness(2, 0, 0, 0) };
            spin.Children.Add(SpinButton("▲", () => Value += step));
            spin.Children.Add(SpinButton("▼", () => Value -= step));
            SetColumn(spin, 1);
            Children.Add(spin);
        }

        static RepeatButton SpinButton(string glyph, Action act)
        {
            var b = new RepeatButton
            {
                Content = glyph,
                FontSize = 6,
                Width = 18,
                Height = 12,
                Padding = new Thickness(0),
                Focusable = false,
                Interval = 60,
            };
            b.Click += (s, e) => act();
            return b;
        }

        int Clamp(int v) => Math.Max(min, Math.Min(max, v));

        public int Value
        {
            get => value;
            set
            {
                int v = Clamp(value);
                bool changed = v != this.value;
                this.value = v;
                box.Text = v.ToString();
                if (changed) ValueChanged?.Invoke(v);
            }
        }

        /// <summary>Sets the value without raising ValueChanged.</summary>
        public void SetSilently(int v)
        {
            value = Clamp(v);
            box.Text = value.ToString();
        }
    }

    public class InputDialog : Window
    {
        readonly TextBox box;

        InputDialog(Window owner, string title, string prompt, string initial)
        {
            Theme.Attach(this);
            Owner = owner;
            Title = title;
            SizeToContent = SizeToContent.WidthAndHeight;
            ResizeMode = ResizeMode.NoResize;
            WindowStartupLocation = WindowStartupLocation.CenterOwner;
            ShowInTaskbar = false;

            var panel = new StackPanel { Margin = new Thickness(16), Width = 320 };
            panel.Children.Add(new TextBlock { Text = prompt, Margin = new Thickness(0, 0, 0, 6) });
            box = new TextBox { Text = initial ?? "", Padding = new Thickness(4, 3, 4, 3) };
            panel.Children.Add(box);
            var buttons = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Right,
                Margin = new Thickness(0, 14, 0, 0),
            };
            var ok = Ui.Button("OK", (s, e) => DialogResult = true, true);
            ok.IsDefault = true;
            var cancel = Ui.Button("Cancel", (s, e) => DialogResult = false);
            cancel.IsCancel = true;
            cancel.Margin = new Thickness(0);
            buttons.Children.Add(ok);
            buttons.Children.Add(cancel);
            panel.Children.Add(buttons);
            Content = panel;
            Loaded += (s, e) => { box.Focus(); box.SelectAll(); };
        }

        public static string Ask(Window owner, string title, string prompt, string initial)
        {
            var d = new InputDialog(owner, title, prompt, initial);
            return d.ShowDialog() == true ? d.box.Text : null;
        }
    }
}
