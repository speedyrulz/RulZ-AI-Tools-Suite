using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace RulZ.ImageTournament
{
    // Sends a file to the Recycle Bin, and only there. VB's FileSystem.DeleteFile (and SHFileOperation with
    // FOF_NOCONFIRMATION) silently deletes a file for good when Windows can't recycle it: network shares,
    // most USB sticks, files too big for the bin, or a bin set to "remove files immediately". Here the
    // shell's copy engine tells a progress sink, just before deleting, whether the item will be recycled;
    // if not, the sink cancels and nothing is deleted.
    static class RecycleBin
    {
        const uint FOF_SILENT = 0x4, FOF_NOCONFIRMATION = 0x10, FOF_ALLOWUNDO = 0x40, FOF_NOERRORUI = 0x400;
        const uint TSF_DELETE_RECYCLE_IF_POSSIBLE = 0x80;
        const int S_OK = 0;
        static readonly int Cancelled = unchecked((int)0x800704C7); // HRESULT_FROM_WIN32(ERROR_CANCELLED)

        // True when the file is now in the Recycle Bin; false when Windows would have deleted it for good,
        // in which case nothing happened to it. Throws IOException when the operation failed.
        public static bool Send(IWin32Window owner, string path)
        {
            string full = Path.GetFullPath(path);
            IShellItem item;
            Guid iid = typeof(IShellItem).GUID;
            int hr = SHCreateItemFromParsingName(full, IntPtr.Zero, ref iid, out item);
            if (hr != S_OK) throw new IOException("Windows couldn't find the file to recycle it (error 0x" + hr.ToString("X8") + ").");

            var op = (IFileOperation)new FileOperationRCW();
            var sink = new Sink();
            try
            {
                op.SetOperationFlags(FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI);
                if (owner != null) op.SetOwnerWindow(owner.Handle);
                op.DeleteItem(item, sink);
                hr = op.PerformOperations();
                if (sink.WouldNotRecycle) return false;
                bool aborted;
                op.GetAnyOperationsAborted(out aborted);
                if (hr != S_OK || aborted)
                    throw new IOException(hr == S_OK || hr == Cancelled
                        ? "Windows couldn't send it to the Recycle Bin. It may be open in another program."
                        : "Windows couldn't send it to the Recycle Bin (error 0x" + hr.ToString("X8") + ").");
                return true;
            }
            finally
            {
                Marshal.ReleaseComObject(op);
                Marshal.ReleaseComObject(item);
            }
        }

        // Watches the delete: cancels it if the item is about to be deleted for good instead of recycled.
        [ComVisible(true)]
        sealed class Sink : IFileOperationProgressSink
        {
            public bool WouldNotRecycle;

            public int PreDeleteItem(uint flags, IShellItem item)
            {
                if ((flags & TSF_DELETE_RECYCLE_IF_POSSIBLE) != 0) return S_OK;
                WouldNotRecycle = true;
                return Cancelled;
            }

            public int StartOperations() { return S_OK; }
            public int FinishOperations(int result) { return S_OK; }
            public int PreRenameItem(uint flags, IShellItem item, string newName) { return S_OK; }
            public int PostRenameItem(uint flags, IShellItem item, string newName, int result, IShellItem created) { return S_OK; }
            public int PreMoveItem(uint flags, IShellItem item, IShellItem folder, string newName) { return S_OK; }
            public int PostMoveItem(uint flags, IShellItem item, IShellItem folder, string newName, int result, IShellItem created) { return S_OK; }
            public int PreCopyItem(uint flags, IShellItem item, IShellItem folder, string newName) { return S_OK; }
            public int PostCopyItem(uint flags, IShellItem item, IShellItem folder, string newName, int result, IShellItem created) { return S_OK; }
            public int PostDeleteItem(uint flags, IShellItem item, int result, IShellItem created) { return S_OK; }
            public int PreNewItem(uint flags, IShellItem folder, string newName) { return S_OK; }
            public int PostNewItem(uint flags, IShellItem folder, string newName, string templateName, uint attributes, int result, IShellItem created) { return S_OK; }
            public int UpdateProgress(uint workTotal, uint workSoFar) { return S_OK; }
            public int ResetTimer() { return S_OK; }
            public int PauseTimer() { return S_OK; }
            public int ResumeTimer() { return S_OK; }
        }

        [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
        static extern int SHCreateItemFromParsingName(string path, IntPtr pbc, ref Guid riid, out IShellItem item);

        [ComImport, Guid("3ad05575-8857-4850-9277-11b85bdb8e09")]
        class FileOperationRCW { }

        [ComImport, Guid("947aab5f-0a5c-4c13-b4d6-4bf7836fc9f8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IFileOperation
        {
            void Advise(IFileOperationProgressSink sink, out uint cookie);
            void Unadvise(uint cookie);
            void SetOperationFlags(uint flags);
            void SetProgressMessage([MarshalAs(UnmanagedType.LPWStr)] string message);
            void SetProgressDialog(IntPtr dialog);
            void SetProperties(IntPtr properties);
            void SetOwnerWindow(IntPtr hwnd);
            void ApplyPropertiesToItem(IShellItem item);
            void ApplyPropertiesToItems(IntPtr items);
            void RenameItem(IShellItem item, [MarshalAs(UnmanagedType.LPWStr)] string newName, IFileOperationProgressSink sink);
            void RenameItems(IntPtr items, [MarshalAs(UnmanagedType.LPWStr)] string newName);
            void MoveItem(IShellItem item, IShellItem folder, [MarshalAs(UnmanagedType.LPWStr)] string newName, IFileOperationProgressSink sink);
            void MoveItems(IntPtr items, IShellItem folder);
            void CopyItem(IShellItem item, IShellItem folder, [MarshalAs(UnmanagedType.LPWStr)] string newName, IFileOperationProgressSink sink);
            void CopyItems(IntPtr items, IShellItem folder);
            void DeleteItem(IShellItem item, IFileOperationProgressSink sink);
            void DeleteItems(IntPtr items);
            void NewItem(IShellItem folder, uint attributes, [MarshalAs(UnmanagedType.LPWStr)] string name,
                [MarshalAs(UnmanagedType.LPWStr)] string templateName, IFileOperationProgressSink sink);
            [PreserveSig] int PerformOperations();
            void GetAnyOperationsAborted([MarshalAs(UnmanagedType.Bool)] out bool aborted);
        }

        [ComImport, Guid("04b0f1a7-9490-44bc-96e1-4296a31252e2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IFileOperationProgressSink
        {
            [PreserveSig] int StartOperations();
            [PreserveSig] int FinishOperations(int result);
            [PreserveSig] int PreRenameItem(uint flags, IShellItem item, [MarshalAs(UnmanagedType.LPWStr)] string newName);
            [PreserveSig] int PostRenameItem(uint flags, IShellItem item, [MarshalAs(UnmanagedType.LPWStr)] string newName, int result, IShellItem created);
            [PreserveSig] int PreMoveItem(uint flags, IShellItem item, IShellItem folder, [MarshalAs(UnmanagedType.LPWStr)] string newName);
            [PreserveSig] int PostMoveItem(uint flags, IShellItem item, IShellItem folder, [MarshalAs(UnmanagedType.LPWStr)] string newName, int result, IShellItem created);
            [PreserveSig] int PreCopyItem(uint flags, IShellItem item, IShellItem folder, [MarshalAs(UnmanagedType.LPWStr)] string newName);
            [PreserveSig] int PostCopyItem(uint flags, IShellItem item, IShellItem folder, [MarshalAs(UnmanagedType.LPWStr)] string newName, int result, IShellItem created);
            [PreserveSig] int PreDeleteItem(uint flags, IShellItem item);
            [PreserveSig] int PostDeleteItem(uint flags, IShellItem item, int result, IShellItem created);
            [PreserveSig] int PreNewItem(uint flags, IShellItem folder, [MarshalAs(UnmanagedType.LPWStr)] string newName);
            [PreserveSig] int PostNewItem(uint flags, IShellItem folder, [MarshalAs(UnmanagedType.LPWStr)] string newName,
                [MarshalAs(UnmanagedType.LPWStr)] string templateName, uint attributes, int result, IShellItem created);
            [PreserveSig] int UpdateProgress(uint workTotal, uint workSoFar);
            [PreserveSig] int ResetTimer();
            [PreserveSig] int PauseTimer();
            [PreserveSig] int ResumeTimer();
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
