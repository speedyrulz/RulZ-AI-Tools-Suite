using System.Windows.Forms;

namespace RulZ.ImageTournament
{
    // Decides whether a key press belongs to the focused control instead of the app's shortcuts.
    static class KeyRouting
    {
        // True when the focused control uses typing or arrow keys itself (text boxes, drop-downs,
        // number boxes), so shortcuts like the arrow keys must not pick a winner.
        public static bool IsInputControl(Control focused)
        {
            // Containers such as NumericUpDown report themselves as the active control; look inside.
            var container = focused as ContainerControl;
            while (container != null && container.ActiveControl != null && container.ActiveControl != focused)
            {
                focused = container.ActiveControl;
                container = focused as ContainerControl;
            }
            return focused is TextBoxBase || focused is ComboBox || focused is UpDownBase
                || focused is ListControl || focused is DateTimePicker || focused is TrackBar;
        }
    }
}
