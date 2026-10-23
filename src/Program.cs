using System;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("RulZ Image Tournament")]
[assembly: System.Reflection.AssemblyProduct("RulZ Image Tournament")]
[assembly: System.Reflection.AssemblyVersion("1.0.0.0")]
[assembly: System.Reflection.AssemblyFileVersion("1.0.0.0")]

namespace RulZ.ImageTournament
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.ThreadException += (s, e) =>
                MessageBox.Show(e.Exception.Message, "RulZ Image Tournament", MessageBoxButtons.OK, MessageBoxIcon.Error);
            // Paths passed on the command line (or dropped onto the .exe) start a tournament.
            Application.Run(new MainForm(args));
        }
    }
}
