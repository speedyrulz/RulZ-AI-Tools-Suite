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

        Console.WriteLine(_failures == 0 ? "All tournament tests passed." : _failures + " failure(s).");
        return _failures == 0 ? 0 : 1;
    }
}
