using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace RulZ.ImageTournament
{
    // Black and red, matching the RulZ logo.
    static class Theme
    {
        public static readonly Color Back = Color.FromArgb(12, 12, 14);
        public static readonly Color Panel = Color.FromArgb(22, 22, 25);
        public static readonly Color Control = Color.FromArgb(36, 36, 40);
        public static readonly Color Hover = Color.FromArgb(54, 54, 60);
        public static readonly Color Border = Color.FromArgb(64, 64, 70);
        public static readonly Color Text = Color.FromArgb(242, 242, 242);
        public static readonly Color Dim = Color.FromArgb(150, 150, 158);
        public static readonly Color Accent = Color.FromArgb(237, 0, 13);       // the logo red
        public static readonly Color AccentText = Color.FromArgb(255, 70, 70);  // red that stays readable on black
        public static readonly Color AccentDark = Color.FromArgb(120, 0, 8);
        public static readonly Color AccentHover = Color.FromArgb(165, 0, 12);

        // The wordmark, embedded in the .exe by build.ps1 (white lettering for dark backgrounds).
        public static Bitmap LoadLogo()
        {
            using (var s = typeof(Theme).Assembly.GetManifestResourceStream("RulZ.ImageTournament.logo.png"))
            {
                if (s == null) return null;
                using (var img = Image.FromStream(s)) return new Bitmap(img);
            }
        }
    }

    // Draws an image scaled to fit, keeping its aspect ratio, with high-quality filtering.
    sealed class LogoBox : System.Windows.Forms.Control
    {
        readonly Image _image;

        public LogoBox(Image image)
        {
            _image = image;
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer
                     | ControlStyles.ResizeRedraw | ControlStyles.SupportsTransparentBackColor, true);
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            if (_image == null) return;
            float s = Math.Min((float)ClientSize.Width / _image.Width, (float)ClientSize.Height / _image.Height);
            float w = _image.Width * s, h = _image.Height * s;
            e.Graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            e.Graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
            e.Graphics.DrawImage(_image, (ClientSize.Width - w) / 2, (ClientSize.Height - h) / 2, w, h);
        }
    }

    // Simple key=value settings stored in %APPDATA%\RulZ Image Tournament.
    sealed class Settings
    {
        public static readonly string Folder = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "RulZ Image Tournament");

        static string FilePath { get { return Path.Combine(Folder, "settings.ini"); } }

        readonly Dictionary<string, string> _values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        public static Settings Load()
        {
            var s = new Settings();
            try
            {
                if (File.Exists(FilePath))
                    foreach (var line in File.ReadAllLines(FilePath))
                    {
                        int eq = line.IndexOf('=');
                        if (eq > 0) s._values[line.Substring(0, eq).Trim()] = line.Substring(eq + 1);
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
                File.WriteAllLines(FilePath, _values.Select(kv => kv.Key + "=" + kv.Value));
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }

        public string Get(string key, string fallback)
        {
            string v;
            return _values.TryGetValue(key, out v) ? v : fallback;
        }

        public int GetInt(string key, int fallback)
        {
            int v;
            return int.TryParse(Get(key, null), NumberStyles.Integer, CultureInfo.InvariantCulture, out v) ? v : fallback;
        }

        public bool GetBool(string key, bool fallback)
        {
            bool v;
            return bool.TryParse(Get(key, null), out v) ? v : fallback;
        }

        public void Set(string key, object value)
        {
            _values[key] = Convert.ToString(value, CultureInfo.InvariantCulture);
        }
    }

    static class NativeMethods
    {
        public const int EM_SETCUEBANNER = 0x1501;
        const int DWMWA_USE_IMMERSIVE_DARK_MODE = 20;

        [DllImport("shlwapi.dll", CharSet = CharSet.Unicode)]
        public static extern int StrCmpLogicalW(string a, string b);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        public static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, string lParam);

        [DllImport("dwmapi.dll")]
        static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int value, int size);

        public static void UseDarkTitleBar(IntPtr hwnd)
        {
            int on = 1;
            try { DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ref on, sizeof(int)); }
            catch (DllNotFoundException) { }
            catch (EntryPointNotFoundException) { }
        }
    }

    // The modern Explorer-style folder picker (FolderBrowserDialog in .NET Framework is the old tree view).
    static class FolderPicker
    {
        const uint FOS_PICKFOLDERS = 0x20, FOS_FORCEFILESYSTEM = 0x40;
        const uint SIGDN_FILESYSPATH = 0x80058000;
        const int ERROR_CANCELLED = unchecked((int)0x800704C7);

        public static string Show(IWin32Window owner, string title, string initialFolder)
        {
            IFileDialog dialog;
            try { dialog = (IFileDialog)new FileOpenDialogRCW(); }
            catch (COMException) { return ShowLegacy(owner, title, initialFolder); }
            try
            {
                uint options;
                dialog.GetOptions(out options);
                dialog.SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
                dialog.SetTitle(title);
                if (!string.IsNullOrEmpty(initialFolder) && Directory.Exists(initialFolder))
                {
                    IShellItem item;
                    Guid iid = typeof(IShellItem).GUID;
                    if (SHCreateItemFromParsingName(initialFolder, IntPtr.Zero, ref iid, out item) == 0)
                        dialog.SetFolder(item);
                }
                int hr = dialog.Show(owner == null ? IntPtr.Zero : owner.Handle);
                if (hr == ERROR_CANCELLED) return null;
                if (hr != 0) Marshal.ThrowExceptionForHR(hr);
                IShellItem result;
                dialog.GetResult(out result);
                IntPtr name;
                result.GetDisplayName(SIGDN_FILESYSPATH, out name);
                try { return Marshal.PtrToStringUni(name); }
                finally { Marshal.FreeCoTaskMem(name); }
            }
            finally { Marshal.ReleaseComObject(dialog); }
        }

        static string ShowLegacy(IWin32Window owner, string title, string initialFolder)
        {
            using (var d = new FolderBrowserDialog { Description = title, SelectedPath = initialFolder ?? "" })
                return d.ShowDialog(owner) == DialogResult.OK ? d.SelectedPath : null;
        }

        [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
        static extern int SHCreateItemFromParsingName(string path, IntPtr pbc, ref Guid riid, out IShellItem item);

        [ComImport, Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")]
        class FileOpenDialogRCW { }

        [ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IFileDialog
        {
            [PreserveSig] int Show(IntPtr parent);
            void SetFileTypes(uint count, IntPtr filterSpec);
            void SetFileTypeIndex(uint index);
            void GetFileTypeIndex(out uint index);
            void Advise(IntPtr events, out uint cookie);
            void Unadvise(uint cookie);
            void SetOptions(uint options);
            void GetOptions(out uint options);
            void SetDefaultFolder(IShellItem item);
            void SetFolder(IShellItem item);
            void GetFolder(out IShellItem item);
            void GetCurrentSelection(out IShellItem item);
            void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
            void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
            void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
            void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
            void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
            void GetResult(out IShellItem item);
            void AddPlace(IShellItem item, int place);
            void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string ext);
            void Close(int hr);
            void SetClientGuid(ref Guid guid);
            void ClearClientData();
            void SetFilter(IntPtr filter);
        }

        [ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IShellItem
        {
            void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
            void GetParent(out IShellItem parent);
            void GetDisplayName(uint sigdn, out IntPtr name);
            void GetAttributes(uint mask, out uint attributes);
            void Compare(IShellItem other, uint hint, out int order);
        }
    }
}
