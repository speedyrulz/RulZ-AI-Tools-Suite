using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using RulZ.ImageTournament;

static class TournamentTests
{
    static int _failures;

    static void Check(bool ok, string what)
    {
        if (!ok) { _failures++; Console.WriteLine("FAIL: " + what); }
    }

    static List<string> Images(int n)
    {
        return Enumerable.Range(1, n).Select(i => i.ToString()).ToList();
    }

    // Plays to the end, always keeping the left image, and returns "round:leftvright" per match.
    static string PlayLeft(Tournament t)
    {
        var log = new List<string>();
        while (t.HasMatch) { log.Add(t.Round + ":" + t.Left + "v" + t.Right); t.Choose(true); }
        return string.Join(" ", log);
    }

    static int Main()
    {
        var t = new Tournament();
        t.Start(Images(10));
        Check(PlayLeft(t) == "1:1v2 1:3v4 1:5v6 1:7v8 1:9v10 2:1v3 2:5v7 3:9v1 4:5v9", "10 images bracket order");
        Check(t.Champion == "5", "10 images champion");

        t.Start(Images(3));
        Check(PlayLeft(t) == "1:1v2 2:3v1", "3 images: bye faces winner of 1 and 2");

        t.Start(Images(5));
        Check(PlayLeft(t) == "1:1v2 1:3v4 2:5v1 3:3v5", "5 images: bye rotates");

        t.Start(Images(1));
        Check(!t.HasMatch && t.Champion == "1", "single image is champion");
        t.Start(Images(0));
        Check(t.IsEmpty && t.Champion == null, "empty tournament");

        // Every image except the champion loses exactly once; n-1 matches.
        var rng = new Random(1);
        for (int n = 0; n <= 70; n++)
        {
            t.Start(Images(n));
            int expectedTotal = Math.Max(0, n - 1);
            Check(t.TotalMatches == expectedTotal, "total matches up front, n=" + n);
            var losers = new HashSet<string>();
            while (t.HasMatch)
            {
                Check(t.Left != t.Right, "distinct pair");
                string loser = t.Choose(rng.Next(2) == 0);
                Check(losers.Add(loser), "image lost twice, n=" + n);
                Check(!t.Contains(loser), "loser still alive");
            }
            Check(t.MatchesPlayed == expectedTotal, "matches played, n=" + n);
            if (n > 0) Check(t.Champion != null && !losers.Contains(t.Champion), "champion, n=" + n);
        }

        // Save/load mid-tournament continues identically.
        t.Start(Images(13));
        t.Choose(true); t.Choose(false); t.Choose(true);
        var sw = new StringWriter();
        t.Save(sw);
        var loaded = Tournament.Load(new StringReader(sw.ToString()));
        Check(loaded != null && loaded.MatchInRound == t.MatchInRound && loaded.Round == t.Round, "load restores position");
        Check(loaded != null && PlayLeft(loaded) == PlayLeft(t.Clone()), "load continues identically");

        // Clone is independent (used for undo).
        t.Start(Images(4));
        var before = t.Clone();
        t.Choose(false);
        Check(before.Left == "1" && before.MatchesPlayed == 0 && t.Left == "3", "clone independent");

        // Adding mid-round appends to the current round.
        t.Start(Images(3));
        t.Add(new[] { "4" });
        Check(PlayLeft(t) == "1:1v2 1:3v4 2:1v3", "add joins current round");

        // Adding after a champion challenges the champion.
        t.Start(Images(2));
        t.Choose(false);
        t.Add(new[] { "x" });
        Check(t.Left == "2" && t.Right == "x", "add after champion");

        // Removing a missing image keeps the bracket valid.
        t.Start(Images(4));
        t.Choose(true);
        Check(t.Remove("4") && t.Champion == null && t.HasMatch && t.Left == "3" && t.Right == "1", "remove re-pairs");
        Check(t.Contains("1") && !t.Contains("2") && t.Remove("1") && t.Champion == "3", "remove winner");
        Check(!t.Remove("1") && !t.Remove("2") && !t.Contains("1") && t.Contains("3"), "remove gone image is a no-op");

        // Membership is case-insensitive and follows Add, Choose, Remove, Clone and Load.
        t.Start(new[] { @"C:\Pics\A.jpg", @"C:\Pics\B.jpg", @"C:\Pics\C.jpg" });
        Check(t.Contains(@"c:\pics\a.JPG") && !t.Contains(@"C:\Pics\D.jpg") && !t.Contains(null), "contains ignores case");
        t.Add(new[] { @"C:\Pics\D.jpg" });
        Check(t.Contains(@"c:\PICS\d.jpg"), "contains after add");
        string lost = t.Choose(true); // A beats B
        Check(lost == @"C:\Pics\B.jpg" && !t.Contains(@"c:\pics\b.jpg") && t.Contains(@"C:\PICS\A.JPG"), "contains after choose");
        var snapshot = t.Clone();
        Check(t.Remove(@"c:\pics\c.jpg") && !t.Contains(@"C:\Pics\C.jpg"), "contains after remove (other case)");
        Check(snapshot.Contains(@"C:\Pics\C.jpg") && !snapshot.Contains(@"C:\Pics\B.jpg"), "clone keeps its own membership");
        snapshot.Add(new[] { @"C:\Pics\E.jpg" });
        Check(snapshot.Contains(@"C:\Pics\E.jpg") && !t.Contains(@"C:\Pics\E.jpg"), "clone membership independent");
        sw = new StringWriter();
        snapshot.Save(sw);
        loaded = Tournament.Load(new StringReader(sw.ToString()));
        Check(loaded != null && loaded.Contains(@"c:\pics\a.jpg") && loaded.Contains(@"C:\PICS\C.JPG") && loaded.Contains(@"c:\pics\e.jpg")
              && !loaded.Contains(@"C:\Pics\B.jpg"), "contains after load");
        Check(loaded != null && loaded.AliveImages().All(p => loaded.Contains(p)) && loaded.AliveImages().Count() == loaded.Remaining,
              "membership matches alive images after load");
        while (loaded != null && loaded.HasMatch) loaded.Choose(false);
        Check(loaded != null && loaded.Champion != null && loaded.Contains(loaded.Champion)
              && loaded.AliveImages().Count() == 1, "only champion alive at the end");

        // Membership stays in step with the bracket through random play, adds and removes.
        rng = new Random(7);
        t.Start(Images(40));
        int nextId = 1000;
        var gone = new HashSet<string>();
        while (t.HasMatch)
        {
            int roll = rng.Next(10);
            if (roll == 0) t.Add(new[] { (nextId++).ToString() });
            else if (roll == 1) { string victim = t.AliveImages().First(); t.Remove(victim); gone.Add(victim); }
            else gone.Add(t.Choose(rng.Next(2) == 0));
            var alive = new HashSet<string>(t.AliveImages());
            Check(alive.All(t.Contains) && !gone.Any(p => t.Contains(p) != alive.Contains(p)), "membership in step with bracket");
        }

        // Adding many images in small batches stays fast (membership used to be a linear scan).
        var watch = System.Diagnostics.Stopwatch.StartNew();
        t = new Tournament();
        for (int batch = 0; batch < 200; batch++)
        {
            var add = Enumerable.Range(batch * 100, 100).Select(i => @"C:\Big\img" + i + ".png").Where(p => !t.Contains(p)).ToList();
            t.Add(add);
        }
        Check(t.Remaining == 20000 && t.Contains(@"c:\big\IMG19999.PNG"), "large set added");
        Check(watch.ElapsedMilliseconds < 2000, "large set add took " + watch.ElapsedMilliseconds + " ms");

        // Undo snapshots: restoring one gives exactly what a full copy taken at the same moment gives, with
        // images added after the snapshot appended, through random picks, adds, removes and nested undos.
        rng = new Random(11);
        for (int trial = 0; trial < 60; trial++)
        {
            t = new Tournament();
            t.Start(Images(rng.Next(2, 40)));
            var history = new List<KeyValuePair<Tournament.Snapshot, Tournament>>();
            nextId = 5000;
            for (int step = 0; step < 200; step++)
            {
                int roll = rng.Next(20);
                if (roll < 11 && t.HasMatch)
                {
                    history.Add(new KeyValuePair<Tournament.Snapshot, Tournament>(t.TakeSnapshot(), t.Clone()));
                    t.Choose(rng.Next(2) == 0);
                }
                else if (roll < 14) t.Add(Enumerable.Range(0, rng.Next(1, 4)).Select(i => "a" + (nextId++)).ToList());
                else if (roll < 15 && t.Remaining > 1) t.Remove(t.AliveImages().ElementAt(rng.Next(t.Remaining)));
                else if (history.Count > 0)
                {
                    var last = history[history.Count - 1];
                    history.RemoveAt(history.Count - 1);
                    var expected = last.Value;
                    expected.Add(t.AliveImages().Where(p => !expected.Contains(p)).ToList());
                    t.Restore(last.Key);
                    var a = new StringWriter(); t.Save(a);
                    var b = new StringWriter(); expected.Save(b);
                    Check(a.ToString() == b.ToString(), "snapshot restore matches a full copy, trial " + trial + " step " + step);
                    Check(t.AliveImages().All(t.Contains) && t.AliveImages().Count() == t.Remaining
                          && t.AliveImages().Distinct().Count() == t.Remaining, "membership after restore, trial " + trial);
                }
            }
        }

        // Undoing a pick keeps images added after it.
        t.Start(Images(4));
        var snap = t.TakeSnapshot();
        t.Choose(true); // 1 beats 2
        t.Add(new[] { "5", "6" });
        t.Restore(snap);
        Check(t.Left == "1" && t.Right == "2" && t.Contains("2") && t.Contains("5") && t.Contains("6") && t.Remaining == 6,
              "undo keeps images added after the pick");
        Check(PlayLeft(t) == "1:1v2 1:3v4 1:5v6 2:1v3 3:5v1", "undo then play with the added images");

        // Snapshots are O(1): taking one per pick over a large set stays fast and small.
        t = new Tournament();
        t.Start(Enumerable.Range(0, 20000).Select(i => @"C:\Big\img" + i + ".png"));
        watch = System.Diagnostics.Stopwatch.StartNew();
        var snaps = new List<Tournament.Snapshot>();
        for (int i = 0; i < 2000; i++) { snaps.Add(t.TakeSnapshot()); t.Choose(i % 2 == 0); }
        Check(watch.ElapsedMilliseconds < 500, "2000 snapshots over 20000 images took " + watch.ElapsedMilliseconds + " ms");
        t.Restore(snaps[0]);
        Check(t.MatchesPlayed == 0 && t.Remaining == 20000 && t.Left == @"C:\Big\img0.png", "restore the first of many snapshots");

        // Preloading looks into the next round: the next pair is always among Upcoming(2) or the current pair.
        rng = new Random(3);
        for (int n = 2; n <= 40; n++)
        {
            t.Start(Images(n));
            while (t.HasMatch)
            {
                var near = new HashSet<string>(t.Upcoming(2)) { t.Left, t.Right };
                t.Choose(rng.Next(2) == 0);
                if (t.HasMatch) Check(near.Contains(t.Left) && near.Contains(t.Right), "next pair was preloaded, n=" + n);
            }
        }
        t.Start(Images(10));
        for (int i = 0; i < 4; i++) t.Choose(true); // round 1 down to 9 v 10; winners so far 1, 3, 5, 7
        Check(string.Join(",", t.Upcoming(2)) == "1,3", "last match of a round preloads the next round's first match");
        t.Start(Images(5));
        t.Choose(true); // 3 v 4 next, 5 has the bye
        Check(string.Join(",", t.Upcoming(2)) == "5,1", "bye and first winner are preloaded at the end of a round");

        // RemoveAll drops many images in one pass and keeps the bracket valid.
        t.Start(Images(6));
        t.Choose(true); // 1 beats 2
        Check(t.RemoveAll(new[] { "3", "6", "2", "nope" }) == 2 && !t.Contains("3") && !t.Contains("6") && t.Remaining == 3,
              "remove all counts only images in the running");
        Check(PlayLeft(t) == "1:4v5 2:1v4", "remove all keeps the round order");
        t = new Tournament();
        var many = Enumerable.Range(0, 20000).Select(i => @"C:\Gone\img" + i + ".png").ToList();
        t.Start(many);
        watch = System.Diagnostics.Stopwatch.StartNew();
        Check(t.RemoveAll(many.Skip(1)) == 19999 && t.Champion == many[0], "remove all but one leaves a champion");
        Check(watch.ElapsedMilliseconds < 500, "removing 19999 images took " + watch.ElapsedMilliseconds + " ms");

        // A damaged or hand-edited state file is refused rather than resumed wrongly.
        Func<string, Tournament> load = s => Tournament.Load(new StringReader(s));
        string good = "RulZImageTournament/1\nround=1\nplayed=1\npos=2\n[round]\na\nb\nc\nd\n[winners]\na\n";
        Check(load(good) != null && load(good).Left == "c", "well-formed state loads");
        Check(load(good.Replace("pos=2", "pos=1")) == null, "odd position refused");
        Check(load(good.Replace("pos=2", "pos=6")) == null, "position past the round refused");
        Check(load(good.Replace("round=1", "round=0")) == null, "round 0 refused");
        Check(load(good.Replace("played=1", "played=-1")) == null, "negative matches refused");
        Check(load(good.Replace("[winners]\na\n", "[winners]\nc\n")) == null, "an image listed twice is refused");
        Check(load(good.Replace("RulZImageTournament/1", "Something/1")) == null, "wrong header refused");

        // Held-down keys: the repeats Windows sends while a key stays down are recognised.
        Check(KeyRouting.IsRepeat(System.Windows.Forms.Message.Create(IntPtr.Zero, 0x100, IntPtr.Zero, new IntPtr(0x40000001))), "key repeat detected");
        Check(!KeyRouting.IsRepeat(System.Windows.Forms.Message.Create(IntPtr.Zero, 0x100, IntPtr.Zero, new IntPtr(1))), "first key press is not a repeat");

        // Shortcut keys stay with controls that use typing or arrows themselves.
        Check(KeyRouting.IsInputControl(new System.Windows.Forms.TextBox()), "text box keeps keys");
        Check(KeyRouting.IsInputControl(new System.Windows.Forms.ComboBox()), "combo box keeps keys");
        Check(KeyRouting.IsInputControl(new System.Windows.Forms.NumericUpDown()), "number box keeps keys");
        Check(!KeyRouting.IsInputControl(new System.Windows.Forms.Button()), "button doesn't keep keys");
        Check(!KeyRouting.IsInputControl(new System.Windows.Forms.CheckBox()), "check box doesn't keep keys");
        Check(!KeyRouting.IsInputControl(null), "no focused control");

        Console.WriteLine(_failures == 0 ? "All tournament tests passed." : _failures + " failure(s).");
        return _failures == 0 ? 0 : 1;
    }
}
