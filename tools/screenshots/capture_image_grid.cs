// Screenshot harness for RulZ Image Grid (WPF). Built and run by capture_image_grid.ps1, which compiles it
// against a copy of "RulZ Image Grid.exe" made outside the repo and sets RULZ_IMAGEGRID_DATA to a fresh folder.
//
//   capture_image_grid.exe --images R:\Grid-demo --out R:\grid-out\image-grid [--readme R:\grid-out-readme]
//
// The app's windows are shown off screen (Left = -30000, never activated) and captured one at a time with
// PrintWindow(PW_RENDERFULLCONTENT), title bar included, so nothing else on the desktop can be in a picture.
// The process is DPI-unaware (DisableDpiAwareness), so the images are at 100% scale on a scaled screen.
// The main window gets a fresh Settings object and the settings folder must be the RULZ_IMAGEGRID_DATA one:
// the real %APPDATA%\RulZ Image Grid\settings.ini is never read or written.
//
// Gallery (--out): 01-dark.png, 02-light.png, 03-export.png, 04-choose-your-look.png
// README (--readme): the same four as the README's file names (01-dark.png, 02-light.png, 03-export-dark.png,
// 04-choose-your-look.png); capture_image_grid.ps1 re-saves them as the README's JPEGs.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Effects;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using RulZImageGrid;

[assembly: DisableDpiAwareness]

static class CaptureImageGrid
{
    const BindingFlags Any = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;

    // The frame (title bar included) of the main window, in pixels at 100%.
    const int FrameW = 1600, FrameH = 1000;

    static readonly string[] ColumnLabels = { "Option A", "Option B", "Option C", "Option D" };
    static readonly string[] RowLabels = { "Batch 1", "Batch 2", "Batch 3" };

    [STAThread]
    static int Main(string[] args)
    {
        string images = Arg(args, "--images"), outDir = Arg(args, "--out"), readme = Arg(args, "--readme");
        if (images == null || outDir == null)
        {
            Console.Error.WriteLine("usage: capture_image_grid --images DIR --out DIR [--readme DIR]");
            return 2;
        }

        // Refuse to run against the real settings folder.
        string data = Environment.GetEnvironmentVariable("RULZ_IMAGEGRID_DATA");
        string real = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "RulZ Image Grid");
        if (string.IsNullOrWhiteSpace(data) || string.Equals(Path.GetFullPath(Settings.Folder).TrimEnd('\\'),
                Path.GetFullPath(real).TrimEnd('\\'), StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine("Set RULZ_IMAGEGRID_DATA to a fresh folder first: the app would use " + real);
            return 2;
        }
        if (File.Exists(Path.Combine(Settings.Folder, "settings.ini")))
        {
            Console.Error.WriteLine("RULZ_IMAGEGRID_DATA must be a fresh folder: " + Settings.Folder + " has a settings.ini");
            return 2;
        }
        Directory.CreateDirectory(outDir);
        if (readme != null) Directory.CreateDirectory(readme);

        var app = new Application { ShutdownMode = ShutdownMode.OnExplicitShutdown };
        // Without this the app's awaits resume on the thread pool and touch the UI from there.
        SynchronizationContext.SetSynchronizationContext(new DispatcherSynchronizationContext());
        app.DispatcherUnhandledException += (s, e) => { Console.Error.WriteLine(e.Exception); Environment.Exit(1); };
        Theme.Init(app, Palette.Dark);

        var win = new MainWindow(new Settings())
        {
            WindowStartupLocation = WindowStartupLocation.Manual,
            Left = -30000,
            Top = -30000,
            Width = FrameW,
            Height = FrameH,
            ShowActivated = false,
            ShowInTaskbar = false,
        };
        win.Show();
        Pump(300);
        FitFrame(win, FrameW, FrameH);

        // In the order make_demo_images.py drew them ("Desert 01", "Bokeh 02", ...): sorted by name, the same kind
        // of picture would sit side by side.
        var files = Directory.GetFiles(images).Where(ImageLoader.IsImage)
            .OrderBy(f => Path.GetFileNameWithoutExtension(f).Split(' ').Last(), StringComparer.OrdinalIgnoreCase)
            .ThenBy(f => Path.GetFileName(f), StringComparer.OrdinalIgnoreCase).ToArray();
        if (files.Length == 0) { Console.Error.WriteLine("No images in " + images); return 2; }
        win.AddFiles(files);
        var items = (List<ImageItem>)Field(win, "items");
        WaitFor(() => items.Count == files.Length && (int)Field(win, "loading") == 0, 60000, "images to load");

        // Four columns, labels filled in from the side panel's own boxes (as if typed).
        var colsBox = (IntBox)Field(win, "columnsBox");
        ((TextBox)Field(colsBox, "box")).Text = "4";
        ((TextBox)Field(win, "colLabelsBox")).Text = string.Join(Environment.NewLine, ColumnLabels);
        ((TextBox)Field(win, "rowLabelsBox")).Text = string.Join(Environment.NewLine, RowLabels);
        Pump(1500); // rebuild, fit and the sharper tiles (loaded 200 ms after a change)

        Save(Capture(win), Path.Combine(outDir, "01-dark.png"));
        if (readme != null) Save(Capture(win), Path.Combine(readme, "01-dark.png"));

        // The export window (what Ctrl+E opens), shown modeless so the harness keeps control, over the main window.
        var grid = (GridSettings)Field(win, "S");
        var export = new ExportWindow(win, grid.ForExport(items.Count), items.ToList())
        {
            WindowStartupLocation = WindowStartupLocation.Manual,
            ShowActivated = false,
        };
        export.Left = win.Left + (win.ActualWidth - export.Width) / 2;
        export.Top = win.Top + (win.ActualHeight - export.Height) / 2;
        export.Show();
        WaitFor(() => !(bool)Field(export, "previewPending") && ((Image)Field(export, "previewImg")).Source != null,
            60000, "the export preview");
        Pump(500);
        var shot = Composite(win, export);
        Save(shot, Path.Combine(outDir, "03-export.png"));
        if (readme != null) Save(shot, Path.Combine(readme, "03-export-dark.png"));
        Console.WriteLine("Export size line: " + ((TextBlock)Field(export, "resText")).Text + " / " +
                          ((TextBlock)Field(export, "detailText")).Text);
        export.Close();
        Pump(300);

        // Light look, switched with the toolbar's own checkbox (it saves to the RULZ_IMAGEGRID_DATA folder).
        ((CheckBox)Field(win, "lightCheck")).IsChecked = true;
        Pump(1000);
        Save(Capture(win), Path.Combine(outDir, "02-light.png"));
        if (readme != null) Save(Capture(win), Path.Combine(readme, "02-light.png"));

        // The first-run "Choose your look" dialog, with Light picked (the way the old README picture showed it).
        var look = (Window)Activator.CreateInstance(typeof(ChooseLookDialog), true);
        RemoveLoadedHandlers(look); // it calls Activate(), which would take the focus from whatever you're doing
        look.WindowStartupLocation = WindowStartupLocation.Manual;
        look.Left = -30000;
        look.Top = -28000;
        look.ShowActivated = false;
        look.ShowInTaskbar = false;
        look.Show();
        typeof(ChooseLookDialog).GetMethod("Pick", Any).Invoke(look, new object[] { Palette.Light });
        Pump(600);
        Save(Capture(look), Path.Combine(outDir, "04-choose-your-look.png"));
        if (readme != null) Save(Capture(look), Path.Combine(readme, "04-choose-your-look.png"));
        look.Close();

        win.Close();
        Pump(200);
        app.Shutdown();
        return 0;
    }

    // ───────────── helpers ─────────────

    static string Arg(string[] args, string name)
    {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }

    static object Field(object o, string name)
    {
        for (var t = o.GetType(); t != null; t = t.BaseType)
        {
            var f = t.GetField(name, Any);
            if (f != null) return f.GetValue(o);
        }
        throw new MissingFieldException(o.GetType().Name, name);
    }

    static void Pump(int ms)
    {
        var end = DateTime.UtcNow.AddMilliseconds(ms);
        do
        {
            var frame = new DispatcherFrame();
            Dispatcher.CurrentDispatcher.BeginInvoke(DispatcherPriority.ContextIdle, new Action(() => frame.Continue = false));
            Dispatcher.PushFrame(frame);
            Thread.Sleep(15);
        } while (DateTime.UtcNow < end);
    }

    static void WaitFor(Func<bool> done, int ms, string what)
    {
        var end = DateTime.UtcNow.AddMilliseconds(ms);
        while (!done())
        {
            if (DateTime.UtcNow > end) throw new TimeoutException("Timed out waiting for " + what);
            Pump(50);
        }
    }

    static void RemoveLoadedHandlers(UIElement el)
    {
        var store = typeof(UIElement).GetProperty("EventHandlersStore", Any).GetValue(el);
        if (store == null) return;
        var handlers = (RoutedEventHandlerInfo[])store.GetType().GetMethod("GetRoutedEventHandlers", Any)
            .Invoke(store, new object[] { FrameworkElement.LoadedEvent });
        if (handlers == null) return;
        foreach (var h in handlers) el.RemoveHandler(FrameworkElement.LoadedEvent, h.Handler);
    }

    // Resizes the window so its visible frame (DWM bounds, without the invisible resize border) is w x h.
    static void FitFrame(Window win, int w, int h)
    {
        var r = FrameRect(Hwnd(win));
        win.Width += w - (r.Right - r.Left);
        win.Height += h - (r.Bottom - r.Top);
        Pump(500);
    }

    // ───────────── capture ─────────────

    [StructLayout(LayoutKind.Sequential)] struct RECT { public int Left, Top, Right, Bottom; }

    [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr hwnd, out RECT r);
    [DllImport("user32.dll")] static extern bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);
    [DllImport("user32.dll")] static extern IntPtr GetDC(IntPtr hwnd);
    [DllImport("user32.dll")] static extern int ReleaseDC(IntPtr hwnd, IntPtr hdc);
    [DllImport("gdi32.dll")] static extern IntPtr CreateCompatibleDC(IntPtr hdc);
    [DllImport("gdi32.dll")] static extern IntPtr CreateCompatibleBitmap(IntPtr hdc, int w, int h);
    [DllImport("gdi32.dll")] static extern IntPtr SelectObject(IntPtr hdc, IntPtr obj);
    [DllImport("gdi32.dll")] static extern bool DeleteObject(IntPtr obj);
    [DllImport("gdi32.dll")] static extern bool DeleteDC(IntPtr hdc);
    [DllImport("dwmapi.dll")] static extern int DwmGetWindowAttribute(IntPtr hwnd, int attr, out RECT r, int size);

    const uint PW_RENDERFULLCONTENT = 2;
    const int DWMWA_EXTENDED_FRAME_BOUNDS = 9;

    static IntPtr Hwnd(Window w) => new WindowInteropHelper(w).Handle;

    static RECT FrameRect(IntPtr hwnd)
    {
        if (DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, out var r, Marshal.SizeOf(typeof(RECT))) == 0) return r;
        GetWindowRect(hwnd, out r);
        return r;
    }

    // PrintWindow(PW_RENDERFULLCONTENT) of one window, cropped to its visible frame.
    static BitmapSource Capture(Window win)
    {
        var hwnd = Hwnd(win);
        GetWindowRect(hwnd, out var wr);
        var fr = FrameRect(hwnd);
        int w = wr.Right - wr.Left, h = wr.Bottom - wr.Top;
        IntPtr screen = GetDC(IntPtr.Zero), dc = CreateCompatibleDC(screen), bmp = CreateCompatibleBitmap(screen, w, h);
        IntPtr old = SelectObject(dc, bmp);
        try
        {
            if (!PrintWindow(hwnd, dc, PW_RENDERFULLCONTENT)) throw new InvalidOperationException("PrintWindow failed");
            SelectObject(dc, old);
            var full = Imaging.CreateBitmapSourceFromHBitmap(bmp, IntPtr.Zero, Int32Rect.Empty, BitmapSizeOptions.FromEmptyOptions());
            var crop = new CroppedBitmap(full, new Int32Rect(fr.Left - wr.Left, fr.Top - wr.Top, fr.Right - fr.Left, fr.Bottom - fr.Top));
            var opaque = new FormatConvertedBitmap(crop, PixelFormats.Bgr32, null, 0); // PrintWindow leaves alpha undefined
            opaque.Freeze();
            return opaque;
        }
        finally
        {
            DeleteObject(bmp);
            DeleteDC(dc);
            ReleaseDC(IntPtr.Zero, screen);
        }
    }

    // The main window with a dialog over it at its real position, and a soft shadow under the dialog the way
    // Windows draws one. Each window is captured on its own.
    static BitmapSource Composite(Window main, Window dialog)
    {
        var a = Capture(main);
        var b = Capture(dialog);
        var ar = FrameRect(Hwnd(main));
        var br = FrameRect(Hwnd(dialog));
        var canvas = new Canvas { Width = a.PixelWidth, Height = a.PixelHeight };
        canvas.Children.Add(new Image { Source = a, Width = a.PixelWidth, Height = a.PixelHeight });
        var top = new Image
        {
            Source = b,
            Width = b.PixelWidth,
            Height = b.PixelHeight,
            Effect = new DropShadowEffect { BlurRadius = 28, ShadowDepth = 6, Direction = 270, Opacity = 0.55, Color = Colors.Black },
        };
        Canvas.SetLeft(top, br.Left - ar.Left);
        Canvas.SetTop(top, br.Top - ar.Top);
        canvas.Children.Add(top);
        canvas.Measure(new Size(a.PixelWidth, a.PixelHeight));
        canvas.Arrange(new Rect(0, 0, a.PixelWidth, a.PixelHeight));
        canvas.UpdateLayout();
        var rtb = new RenderTargetBitmap(a.PixelWidth, a.PixelHeight, 96, 96, PixelFormats.Pbgra32);
        rtb.Render(canvas);
        rtb.Freeze();
        return rtb;
    }

    static void Save(BitmapSource bmp, string path)
    {
        var enc = new PngBitmapEncoder();
        enc.Frames.Add(BitmapFrame.Create(bmp));
        using (var f = File.Create(path)) enc.Save(f);
        Console.WriteLine($"{path}  {bmp.PixelWidth}x{bmp.PixelHeight}");
    }
}
