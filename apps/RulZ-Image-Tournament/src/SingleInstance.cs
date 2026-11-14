using System;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Security.Cryptography;
using System.Security.Principal;
using System.Text;
using System.Threading;

namespace RulZ.ImageTournament
{
    // One running copy per user, Windows session and data folder. A second launch (Send to, a double-click
    // while the app is open, files dropped on the .exe) hands its paths to the running copy over a named pipe
    // and exits, so two copies never read and overwrite the same tournament.txt. Only the same Windows user
    // can open the pipe. If the running copy doesn't answer, the new one simply runs as before.
    sealed class SingleInstance : IDisposable
    {
        const int ConnectTimeoutMs = 10000;
        const int ASFW_ANY = -1;

        readonly Mutex _mutex;
        readonly string _pipeName;

        public bool IsFirst { get; private set; }

        public SingleInstance(string dataFolder)
        {
            string user = WindowsIdentity.GetCurrent().User.Value;
            int session = Process.GetCurrentProcess().SessionId;
            string key;
            using (var sha = SHA256.Create())
            {
                byte[] hash = sha.ComputeHash(Encoding.UTF8.GetBytes(user + "|" + session + "|" + dataFolder.ToUpperInvariant()));
                key = BitConverter.ToString(hash, 0, 12).Replace("-", "");
            }
            _pipeName = "RulZ.ImageTournament." + key;
            try
            {
                bool createdNew;
                _mutex = new Mutex(true, @"Local\" + _pipeName, out createdNew);
                IsFirst = createdNew;
            }
            catch (UnauthorizedAccessException)
            {
                // A copy running as administrator holds it and a normal launch may not open it. Then this copy
                // tries to hand over its paths, and if that's refused too, it simply runs as well.
                _mutex = null;
            }
        }

        // Sends the paths (made absolute here, since the running copy has its own current folder) to the
        // running copy, which also comes to the front. True only once the running copy has confirmed that it
        // took them; false if it didn't (then this copy runs itself, so the paths aren't lost).
        // The message is a 4-byte length and the paths joined by newlines in UTF-8; the answer is one byte.
        public bool SendToFirst(string[] args)
        {
            var paths = args.Select(Absolute).Where(p => p != null).ToArray();
            try
            {
                using (var pipe = new NamedPipeClientStream(".", _pipeName, PipeDirection.InOut))
                {
                    pipe.Connect(ConnectTimeoutMs);
                    AllowSetForegroundWindow(ASFW_ANY); // let the running copy bring its window forward
                    byte[] data = Encoding.UTF8.GetBytes(string.Join("\n", paths));
                    pipe.Write(BitConverter.GetBytes(data.Length), 0, 4);
                    pipe.Write(data, 0, data.Length);
                    pipe.Flush();
                    var answer = new byte[1];
                    var read = pipe.ReadAsync(answer, 0, 1);
                    return read.Wait(ConnectTimeoutMs) && read.Result == 1 && answer[0] == Accepted;
                }
            }
            catch (TimeoutException) { return false; }
            catch (IOException) { return false; }
            catch (UnauthorizedAccessException) { return false; }
            catch (AggregateException) { return false; } // the read failed: the running copy hung up
        }

        const byte Accepted = 1;
        const int MaxMessage = 1 << 24; // 16 MB of paths is far more than any launch passes

        // Starts taking paths from later launches on a background thread. received runs on that thread and
        // returns whether the paths were handed to the window; only then is the launch told they were taken.
        public void Listen(Func<string[], bool> received)
        {
            new Thread(() => Serve(received)) { IsBackground = true, Name = "Later launches" }.Start();
        }

        void Serve(Func<string[], bool> received)
        {
            var security = new PipeSecurity();
            security.AddAccessRule(new PipeAccessRule(WindowsIdentity.GetCurrent().User, PipeAccessRights.FullControl, AccessControlType.Allow));
            while (true)
            {
                try
                {
                    using (var pipe = new NamedPipeServerStream(_pipeName, PipeDirection.InOut, 1, PipeTransmissionMode.Byte,
                        PipeOptions.None, 0, 0, security))
                    {
                        pipe.WaitForConnection();
                        byte[] size = ReadExactly(pipe, 4);
                        int length = size == null ? -1 : BitConverter.ToInt32(size, 0);
                        if (length < 0 || length > MaxMessage) continue;
                        byte[] data = ReadExactly(pipe, length);
                        if (data == null) continue;
                        string[] paths = Encoding.UTF8.GetString(data).Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
                        if (!received(paths)) continue; // no answer: the launch runs itself
                        pipe.WriteByte(Accepted);
                        pipe.Flush();
                        pipe.WaitForPipeDrain();
                    }
                }
                catch (IOException) { Thread.Sleep(200); }      // a launch that hung up early; keep listening
                catch (UnauthorizedAccessException) { return; } // the name is taken by something else
            }
        }

        static byte[] ReadExactly(Stream s, int count)
        {
            var buffer = new byte[count];
            for (int done = 0; done < count; )
            {
                int n = s.Read(buffer, done, count - done);
                if (n <= 0) return null;
                done += n;
            }
            return buffer;
        }

        static string Absolute(string path)
        {
            try { return Path.GetFullPath(path); }
            catch (ArgumentException) { return null; }
            catch (NotSupportedException) { return null; }
            catch (IOException) { return null; }
            catch (System.Security.SecurityException) { return null; }
        }

        public void Dispose()
        {
            if (_mutex == null) return;
            if (IsFirst) _mutex.ReleaseMutex();
            _mutex.Dispose();
        }

        [DllImport("user32.dll")]
        static extern bool AllowSetForegroundWindow(int processId);
    }
}
