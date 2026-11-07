// Tests for the parts of RulZ Image Grid that don't need a window: export layout, saving, loading, previews and
// background threads. Run with .\build.ps1 -Test (the release workflow does this before packaging).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using RulZImageGrid;

static class ImageGridTests
{
    static int failures;

    static void Check(bool ok, string what)
    {
        if (!ok) { failures++; Console.WriteLine("FAIL: " + what); }
    }

    [STAThread]
    static int Main()
    {
        string dir = Path.Combine(Path.GetTempPath(), "RulZImageGridTests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            ExportLabelSpace();
            var files = MakeImages(dir, 7, 600, 400);
            Loading(dir, files);
            Saving(dir, files);
            Previews(files);
            BackgroundThreads(files);
        }
        finally
        {
            try { Directory.Delete(dir, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
        Console.WriteLine(failures == 0 ? "All Image Grid tests passed." : failures + " Image Grid test(s) failed.");
        return failures == 0 ? 0 : 1;
    }

    static List<string> MakeImages(string dir, int n, int w, int h)
    {
        var files = new List<string>();
        for (int k = 1; k <= n; k++)
        {
            var px = new byte[w * h * 3];
            for (int i = 0; i < px.Length; i++) px[i] = (byte)(i * k);
            var enc = new JpegBitmapEncoder { QualityLevel = 85 };
            enc.Frames.Add(BitmapFrame.Create(BitmapSource.Create(w, h, 96, 96, PixelFormats.Bgr24, null, px, w * 3)));
            string path = Path.Combine(dir, "img" + k + ".jpg");
            using (var fs = File.Create(path)) enc.Save(fs);
            files.Add(path);
        }
        return files;
    }

    // Label rows and columns with nothing typed take no space in the saved image; the editor keeps room for them.
    static void ExportLabelSpace()
    {
        var sizes = Enumerable.Repeat(new Size(400, 300), 5).ToList(); // 3 columns → 2 rows
        var s = new GridSettings { Columns = 3, Padding = 20 };

        var editor = GridLayout.Compute(s, sizes);
        Check(editor.ColHeaderH > 0 && editor.RowHeaderW > 0, "the editor keeps room for the label placeholders");

        var saved = GridLayout.Compute(s.ForExport(5), sizes);
        Check(saved.X0 == 20 && saved.Y0 == 20, "no labels typed: the saved image has no label space");
        Check(saved.Width == 3 * 400 + 4 * 20, "no labels typed: saved width is tiles plus padding");
        Check(saved.Height == 2 * (300 + saved.CaptionH) + 3 * 20, "no labels typed: saved height is tiles, captions plus padding");

        var col = s.Clone();
        col.ColLabels = new List<string> { "", "B" };
        var c = GridLayout.Compute(col.ForExport(5), sizes);
        Check(c.ColHeaderH > 0 && c.RowHeaderW == 0, "one column label typed: column labels kept, empty row labels dropped");

        var past = s.Clone();
        past.ColLabels = new List<string> { "", "", "", "D" }; // a fourth label with three columns
        past.RowLabels = new List<string> { "", "  ", "C" };   // whitespace, then a label for a third row that doesn't exist
        var p = GridLayout.Compute(past.ForExport(5), sizes);
        Check(p.ColHeaderH == 0 && p.RowHeaderW == 0, "labels past the grid's end or only spaces don't count");

        var hidden = s.Clone();
        hidden.ShowColLabels = false;
        hidden.ColLabels = new List<string> { "A" };
        Check(!hidden.ForExport(5).ShowColLabels, "a label set switched off stays off");

        s.ForExport(5);
        Check(s.ShowColLabels && s.ShowRowLabels, "ForExport leaves the editor's settings alone");
    }

    static void Loading(string dir, List<string> files)
    {
        string broken = Path.Combine(dir, "broken.jpg");
        File.WriteAllBytes(broken, new byte[] { 1, 2, 3, 4, 5 });
        var input = files.AsEnumerable().Reverse().ToList();
        input.Insert(3, broken);

        int calls = 0, highest = 0;
        var result = ImageLoader.ReadInfoMany(input, n =>
        {
            Interlocked.Increment(ref calls);
            int seen;
            while ((seen = Volatile.Read(ref highest)) < n && Interlocked.CompareExchange(ref highest, n, seen) != seen) { }
        }).Result;

        Check(result.ok.Select(i => i.Path).SequenceEqual(input.Where(f => f != broken)), "loaded images keep the order they were given in");
        Check(result.failed.Count == 1 && result.failed[0].StartsWith("broken.jpg: "), "an unreadable file is reported by name");
        Check(calls == input.Count && highest == input.Count, "progress is reported once per file, up to the total");

        var it = result.ok[0];
        Check(it.PixelW == 600 && it.PixelH == 400 && it.Name == "img7", "size and name are read");
        Check(Math.Max(it.Thumb.PixelWidth, it.Thumb.PixelHeight) == ImageLoader.ThumbSide, "the in-memory preview is ThumbSide on its long side");
        Check(it.Thumb.IsFrozen, "the preview can be used from any thread");
        File.Delete(broken);
    }

    static void Saving(string dir, List<string> files)
    {
        var items = files.Take(2).Select(ImageLoader.ReadInfo).ToList();
        var bmp = Renderer.Render(new GridSettings().ForExport(items.Count), items, 0.5, false, null);
        string target = Path.Combine(dir, "grid.png");

        Renderer.Save(bmp, target, false, 90, false);
        Check(File.Exists(target) && !File.Exists(target + ".tmp"), "a new file is saved with no .tmp left behind");
        Check(Decode(target).PixelWidth == bmp.PixelWidth, "the saved PNG has the rendered size");

        Renderer.Save(bmp, target, true, 90, false); // JPG data over the existing file
        Check(!File.Exists(target + ".tmp") && Decode(target).PixelHeight == bmp.PixelHeight, "an existing file is replaced");

        long before = new FileInfo(target).Length;
        File.SetAttributes(target, FileAttributes.ReadOnly);
        bool threw = false;
        try { Renderer.Save(bmp, target, false, 90, false); }
        catch (UnauthorizedAccessException) { threw = true; }
        catch (IOException) { threw = true; }
        File.SetAttributes(target, FileAttributes.Normal);
        Check(threw, "saving over a read-only file reports an error");
        Check(new FileInfo(target).Length == before, "a failed save leaves the old file as it was");
        Check(!File.Exists(target + ".tmp"), "a failed save leaves no .tmp behind");
    }

    static BitmapSource Decode(string path)
    {
        using (var fs = File.OpenRead(path))
            return BitmapDecoder.Create(fs, BitmapCreateOptions.None, BitmapCacheOption.OnLoad).Frames[0];
    }

    static void Previews(List<string> files)
    {
        var items = files.Select(ImageLoader.ReadInfo).ToList();
        var s = new GridSettings();
        var fromDisk = Renderer.Render(s, items, 0.2, false, null);
        var fromThumbs = Renderer.Render(s, items, 0.2, false, null, useThumbs: true);
        Check(fromThumbs.PixelWidth == fromDisk.PixelWidth && fromThumbs.PixelHeight == fromDisk.PixelHeight,
            "a preview drawn from the in-memory previews has the same size");

        var cts = new CancellationTokenSource();
        cts.Cancel();
        bool cancelled = false;
        try { Renderer.Render(s, items, 0.2, false, null, cancel: cts.Token); }
        catch (OperationCanceledException) { cancelled = true; }
        Check(cancelled, "a cancelled render stops");
    }

    // Each background render used to leave its thread's WPF dispatcher behind: about a dozen handles per call.
    static void BackgroundThreads(List<string> files)
    {
        var items = files.Take(3).Select(ImageLoader.ReadInfo).ToList();
        var s = new GridSettings();
        Func<BitmapSource> work = () => Renderer.Render(s, items, 0.1, false, null, useThumbs: true);
        for (int i = 0; i < 5; i++) Renderer.RunSta(work).Wait(); // warm up
        int Handles()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            Thread.Sleep(100);
            var p = Process.GetCurrentProcess();
            p.Refresh();
            return p.HandleCount;
        }
        int before = Handles();
        for (int i = 0; i < 60; i++) Renderer.RunSta(work).Wait();
        int grew = Handles() - before;
        Check(grew < 40, $"60 background renders don't leak handles (grew by {grew})");
    }
}
