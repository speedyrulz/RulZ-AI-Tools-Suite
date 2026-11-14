using System;
using System.Windows;

[assembly: System.Reflection.AssemblyTitle("RulZ Image Grid")]
[assembly: System.Reflection.AssemblyProduct("RulZ Image Grid")]
[assembly: System.Reflection.AssemblyVersion("1.2.0.0")]
[assembly: System.Reflection.AssemblyFileVersion("1.2.0.0")]
// csc doesn't add this the way MSBuild does. Without it .NET keeps its pre-4.6.2 behaviour: WPF ignores the
// per-monitor DPI setting in app.manifest (the window keeps its first monitor's scale on another monitor) and
// paths longer than 260 characters can't be opened or saved even where Windows allows them.
[assembly: System.Runtime.Versioning.TargetFramework(".NETFramework,Version=v4.8", FrameworkDisplayName = ".NET Framework 4.8")]

namespace RulZImageGrid
{
    public static class Program
    {
        [STAThread]
        public static void Main(string[] args)
        {
            var app = new Application { ShutdownMode = ShutdownMode.OnExplicitShutdown };
            app.DispatcherUnhandledException += (s, e) =>
            {
                MessageBox.Show(e.Exception.Message, "RulZ Image Grid", MessageBoxButton.OK, MessageBoxImage.Error);
                e.Handled = true;
            };

            // First run asks for Dark or Light; after that the choice is remembered.
            var settings = Settings.Load();
            bool firstRun = !settings.Has("theme");
            Theme.Init(app, settings.Get("theme", "dark") == "light" ? Palette.Light : Palette.Dark);
            if (firstRun)
            {
                var look = ChooseLookDialog.Ask();
                Theme.Apply(look);
                settings.Set("theme", look.IsLight ? "light" : "dark");
                settings.Save();
            }

            var win = new MainWindow(settings);
            if (args.Length > 0) win.Loaded += (s, e) => win.AddFiles(args);
            app.MainWindow = win;
            app.ShutdownMode = ShutdownMode.OnMainWindowClose;
            app.Run(win);
        }
    }
}
