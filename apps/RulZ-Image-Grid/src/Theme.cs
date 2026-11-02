using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RulZImageGrid
{
    // The app's colors, shared with the other RulZ apps. Dark (black and red, matching the RulZ logo) is the
    // default; Light is white with the same red. Theme.Apply swaps these in as brush resources that
    // Theme.xaml and the code reference by name, so switching updates every open window at once.
    public sealed class Palette
    {
        public bool IsLight;
        public Color Back, Panel, Control, Hover, Border, Text, Dim;
        public Color Accent, AccentText, AccentFill, AccentFillHover, OnAccent, Workspace, WarnText;
        public string LogoResource;

        public static readonly Palette Dark = new Palette
        {
            Back = Color.FromRgb(12, 12, 14),
            Panel = Color.FromRgb(22, 22, 25),
            Control = Color.FromRgb(36, 36, 40),
            Hover = Color.FromRgb(54, 54, 60),
            Border = Color.FromRgb(64, 64, 70),
            Text = Color.FromRgb(242, 242, 242),
            Dim = Color.FromRgb(150, 150, 158),
            Accent = Color.FromRgb(237, 0, 13),          // the logo red
            AccentText = Color.FromRgb(255, 70, 70),     // red that stays readable on black
            AccentFill = Color.FromRgb(120, 0, 8),       // primary buttons
            AccentFillHover = Color.FromRgb(165, 0, 12),
            OnAccent = Color.FromRgb(242, 242, 242),
            Workspace = Color.FromRgb(5, 5, 5),          // behind the grid
            WarnText = Color.FromRgb(252, 214, 121),
            LogoResource = "RulZ.ImageGrid.logo.png",
        };

        public static readonly Palette Light = new Palette
        {
            IsLight = true,
            Back = Color.FromRgb(255, 255, 255),
            Panel = Color.FromRgb(245, 245, 247),
            Control = Color.FromRgb(255, 255, 255),
            Hover = Color.FromRgb(235, 235, 239),
            Border = Color.FromRgb(217, 217, 222),
            Text = Color.FromRgb(17, 17, 20),
            Dim = Color.FromRgb(95, 96, 104),
            Accent = Color.FromRgb(237, 0, 13),
            AccentText = Color.FromRgb(200, 0, 11),      // a little darker so small text reads well on white
            AccentFill = Color.FromRgb(237, 0, 13),
            AccentFillHover = Color.FromRgb(200, 0, 11),
            OnAccent = Color.FromRgb(255, 255, 255),
            Workspace = Color.FromRgb(226, 226, 231),
            WarnText = Color.FromRgb(138, 83, 0),
            LogoResource = "RulZ.ImageGrid.logo-light.png",
        };
    }

    public static class Theme
    {
        public static Palette Current = Palette.Dark;
        public static bool IsLight => Current.IsLight;
        public static event Action Changed;

        static ResourceDictionary brushes;

        /// <summary>Loads the control styles (embedded Theme.xaml) and the starting palette.</summary>
        public static void Init(Application app, Palette palette)
        {
            using (var s = typeof(Theme).Assembly.GetManifestResourceStream("RulZ.ImageGrid.Theme.xaml"))
                app.Resources.MergedDictionaries.Add((ResourceDictionary)XamlReader.Load(s));
            Apply(palette);
        }

        public static void Apply(Palette p)
        {
            Current = p;
            var d = new ResourceDictionary();
            void Add(string key, Color c) => d[key] = Ui.Frozen(c);
            Add("Back", p.Back);
            Add("Panel", p.Panel);
            Add("Control", p.Control);
            Add("Hover", p.Hover);
            Add("Border", p.Border);
            Add("Text", p.Text);
            Add("Dim", p.Dim);
            Add("Accent", p.Accent);
            Add("AccentText", p.AccentText);
            Add("AccentFill", p.AccentFill);
            Add("AccentFillHover", p.AccentFillHover);
            Add("OnAccent", p.OnAccent);
            Add("Workspace", p.Workspace);
            Add("WarnText", p.WarnText);

            var merged = Application.Current.Resources.MergedDictionaries;
            if (brushes != null) merged.Remove(brushes);
            brushes = d;
            merged.Add(d);

            foreach (Window w in Application.Current.Windows) UseDarkTitleBar(w, !p.IsLight);
            Changed?.Invoke();
        }

        /// <summary>Themes a window: background, text color and a matching title bar.</summary>
        public static void Attach(Window w)
        {
            w.SetResourceReference(Window.BackgroundProperty, "Back");
            w.SetResourceReference(Window.ForegroundProperty, "Text");
            w.FontFamily = new FontFamily("Segoe UI");
            w.FontSize = 13;
            w.SourceInitialized += (s, e) => UseDarkTitleBar(w, !IsLight);
        }

        // The wordmark, embedded in the .exe by build.ps1: white lettering for dark, the original black for light.
        public static BitmapSource LoadLogo(Palette p) => LoadResourceImage(p.LogoResource);

        public static BitmapSource AppIcon() => LoadResourceImage("RulZ.ImageGrid.app.ico");

        static BitmapSource LoadResourceImage(string name)
        {
            using (var s = typeof(Theme).Assembly.GetManifestResourceStream(name))
            {
                if (s == null) return null;
                if (name.EndsWith(".ico"))
                    return BitmapFrame.Create(s, BitmapCreateOptions.None, BitmapCacheOption.OnLoad);
                var bi = new BitmapImage();
                bi.BeginInit();
                bi.CacheOption = BitmapCacheOption.OnLoad;
                bi.StreamSource = s;
                bi.EndInit();
                bi.Freeze();
                return bi;
            }
        }

        const int DWMWA_USE_IMMERSIVE_DARK_MODE = 20;
        const uint SWP_NOSIZE = 0x1, SWP_NOMOVE = 0x2, SWP_NOZORDER = 0x4, SWP_NOACTIVATE = 0x10, SWP_FRAMECHANGED = 0x20;

        [DllImport("dwmapi.dll")]
        static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int value, int size);

        [DllImport("user32.dll")]
        static extern bool SetWindowPos(IntPtr hWnd, IntPtr after, int x, int y, int cx, int cy, uint flags);

        // A dark or light title bar (Windows 10 20H1+ and 11). Redraws the frame so it also applies to an open window.
        static void UseDarkTitleBar(Window w, bool dark)
        {
            var hwnd = new WindowInteropHelper(w).Handle;
            if (hwnd == IntPtr.Zero) return;
            int on = dark ? 1 : 0;
            try
            {
                DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ref on, sizeof(int));
                SetWindowPos(hwnd, IntPtr.Zero, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
            }
            catch (DllNotFoundException) { }
            catch (EntryPointNotFoundException) { }
        }
    }

    // Simple key=value settings stored in %APPDATA%\RulZ Image Grid.
    // If the RULZ_IMAGEGRID_DATA environment variable names a folder, <that folder>\RulZ Image Grid is used
    // instead (for testing or a portable copy), and the normal settings are left alone.
    public sealed class Settings
    {
        public static readonly string Folder = Path.Combine(DataRoot(), "RulZ Image Grid");

        static string DataRoot()
        {
            string custom = Environment.GetEnvironmentVariable("RULZ_IMAGEGRID_DATA");
            if (!string.IsNullOrWhiteSpace(custom))
            {
                try { return Path.GetFullPath(Environment.ExpandEnvironmentVariables(custom.Trim().Trim('"'))); }
                catch (ArgumentException) { }
                catch (NotSupportedException) { }
            }
            return Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        }

        static string FilePath => Path.Combine(Folder, "settings.ini");

        readonly Dictionary<string, string> values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        public static Settings Load()
        {
            var s = new Settings();
            try
            {
                if (File.Exists(FilePath))
                    foreach (var line in File.ReadAllLines(FilePath))
                    {
                        int eq = line.IndexOf('=');
                        if (eq > 0) s.values[line.Substring(0, eq).Trim()] = line.Substring(eq + 1);
                    }
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            return s;
        }

        public void Save()
        {
            try
            {
                Directory.CreateDirectory(Folder);
                File.WriteAllLines(FilePath, values.Select(kv => kv.Key + "=" + kv.Value));
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }

        public bool Has(string key) => values.ContainsKey(key);

        public string Get(string key, string fallback) => values.TryGetValue(key, out var v) ? v : fallback;

        public void Set(string key, object value) => values[key] = Convert.ToString(value, CultureInfo.InvariantCulture);
    }
}
