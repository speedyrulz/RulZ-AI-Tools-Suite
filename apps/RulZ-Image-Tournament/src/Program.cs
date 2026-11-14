using System;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("RulZ Image Tournament")]
[assembly: System.Reflection.AssemblyProduct("RulZ Image Tournament")]
[assembly: System.Reflection.AssemblyVersion("1.2.1.0")]
[assembly: System.Reflection.AssemblyFileVersion("1.2.1.0")]
// csc doesn't add this on its own (Visual Studio does). Without it .NET runs the app with the defaults of .NET 4.5:
// paths over 260 characters blocked, and WinForms' accessibility improvements (screen readers, high-contrast
// colours) left off. build.ps1 -Test checks it's in the built .exe.
[assembly: System.Runtime.Versioning.TargetFramework(".NETFramework,Version=v4.8", FrameworkDisplayName = ".NET Framework 4.8")]

namespace RulZ.ImageTournament
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            // Paths longer than 260 characters (deep folders, AI images named after their prompt). The manifest
            // opts in on the Windows side, and the target framework attribute above already makes these .NET's
            // defaults; setting them here as well keeps long paths working if the attribute is ever lost. They
            // must be set before anything touches a path, and .NET reads them once. Everything else is in Run:
            // when Main itself mentioned Settings, compiling Main ran Settings' static initializer (which resolves
            // a path) before these lines, and long paths stayed blocked.
            AppContext.SetSwitch("Switch.System.IO.UseLegacyPathHandling", false);
            AppContext.SetSwitch("Switch.System.IO.BlockLongPaths", false);
            Run(args);
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        static void Run(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.ThreadException += (s, e) =>
                MessageBox.Show(e.Exception.Message, "RulZ Image Tournament", MessageBoxButtons.OK, MessageBoxIcon.Error);

            using (var instance = new SingleInstance(Settings.Folder))
            {
                // Already running: hand it these paths (it comes to the front) instead of opening a second window.
                if (!instance.IsFirst && instance.SendToFirst(args)) return;

                // Paths passed on the command line (or dropped onto the .exe) start a tournament.
                var form = new MainForm(args);
                form.Shown += delegate
                {
                    // Answered only once the window has taken the paths (Invoke waits for it), so a launch that
                    // arrives while this copy is closing runs itself instead of being lost.
                    instance.Listen(paths =>
                    {
                        if (form.IsDisposed || form.Disposing) return false;
                        try { return (bool)form.Invoke(new Func<bool>(() => form.AddFromAnotherLaunch(paths))); }
                        catch (InvalidOperationException) { return false; } // closing or closed (ObjectDisposedException too)
                    });
                };
                Application.Run(form);
            }
        }
    }
}
