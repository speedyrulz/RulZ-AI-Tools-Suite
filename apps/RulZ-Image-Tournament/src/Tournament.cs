using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace RulZ.ImageTournament
{
    // Single-elimination bracket over image paths.
    // Images are paired in order (1v2, 3v4, ...). When a round has an odd image out it
    // advances automatically and is placed first in the next round, so it faces the winner
    // of the first match and the bye passes to a different image.
    public sealed class Tournament
    {
        const string Header = "RulZImageTournament/1";

        List<string> _round = new List<string>();
        List<string> _winners = new List<string>();
        int _pos; // index of the left image of the current match
        // How many times each path is still in the running (_round from _pos, plus _winners),
        // so Contains stays O(1) for large sets. Paths compare case-insensitively.
        Dictionary<string, int> _alive = NewAliveMap();

        public int Round { get; private set; }
        public int MatchesPlayed { get; private set; }

        public bool HasMatch { get { return _round.Count - _pos >= 2; } }
        public string Left { get { return HasMatch ? _round[_pos] : null; } }
        public string Right { get { return HasMatch ? _round[_pos + 1] : null; } }
        public int Remaining { get { return _round.Count - _pos + _winners.Count; } }
        public bool IsEmpty { get { return Remaining == 0; } }
        public string Champion { get { return !HasMatch && _round.Count == 1 ? _round[0] : null; } }
        public int MatchInRound { get { return _pos / 2 + 1; } }
        public int MatchesInRound { get { return _round.Count / 2; } }
        public int TotalMatches { get { return MatchesPlayed + Math.Max(0, Remaining - 1); } }
        public string ByeThisRound { get { return _round.Count % 2 == 1 ? _round[_round.Count - 1] : null; } }

        public void Start(IEnumerable<string> images)
        {
            _round = images.ToList();
            _winners = new List<string>();
            _pos = 0;
            Round = 1;
            MatchesPlayed = 0;
            RebuildAlive();
            Advance();
        }

        // New images join the end of the current round (or face the champion if one was decided).
        public void Add(IEnumerable<string> images)
        {
            if (IsEmpty) { Start(images); return; }
            foreach (var p in images)
            {
                _round.Add(p);
                MarkAlive(p);
            }
            Advance();
        }

        public bool Contains(string path)
        {
            return path != null && _alive.ContainsKey(path);
        }

        // Records the current match and returns the losing image's path.
        public string Choose(bool keepLeft)
        {
            if (!HasMatch) throw new InvalidOperationException("No match in progress.");
            string winner = keepLeft ? Left : Right;
            string loser = keepLeft ? Right : Left;
            _winners.Add(winner);
            _pos += 2;
            MatchesPlayed++;
            MarkGone(loser);
            Advance();
            return loser;
        }

        // Drops an image that is still in the running (e.g. its file disappeared).
        public bool Remove(string path)
        {
            if (!Contains(path)) return false;
            int i = IndexOf(_round, path, _pos);
            if (i >= 0) _round.RemoveAt(i);
            else
            {
                i = IndexOf(_winners, path, 0);
                if (i < 0) return false;
                _winners.RemoveAt(i);
            }
            MarkGone(path);
            Advance();
            return true;
        }

        public IEnumerable<string> AliveImages()
        {
            return _round.Skip(_pos).Concat(_winners);
        }

        // Images of the matches after the current one in this round, for preloading.
        public IEnumerable<string> Upcoming(int count)
        {
            for (int i = _pos + 2; i < _round.Count && i < _pos + 2 + count; i++)
                yield return _round[i];
        }

        public Tournament Clone()
        {
            return new Tournament
            {
                _round = new List<string>(_round),
                _winners = new List<string>(_winners),
                _pos = _pos,
                _alive = new Dictionary<string, int>(_alive, _alive.Comparer),
                Round = Round,
                MatchesPlayed = MatchesPlayed,
            };
        }

        static Dictionary<string, int> NewAliveMap()
        {
            return new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        }

        void RebuildAlive()
        {
            _alive = NewAliveMap();
            foreach (var p in AliveImages()) MarkAlive(p);
        }

        void MarkAlive(string path)
        {
            if (path == null) return;
            int n;
            _alive.TryGetValue(path, out n);
            _alive[path] = n + 1;
        }

        void MarkGone(string path)
        {
            int n;
            if (path == null || !_alive.TryGetValue(path, out n)) return;
            if (n <= 1) _alive.Remove(path);
            else _alive[path] = n - 1;
        }

        void Advance()
        {
            while (_round.Count - _pos < 2)
            {
                if (_pos == 0 && _winners.Count == 0) return; // zero or one image left
                var next = new List<string>();
                if (_pos < _round.Count) next.Add(_round[_pos]); // the bye goes first
                next.AddRange(_winners);
                _round = next;
                _winners = new List<string>();
                _pos = 0;
                Round++;
            }
        }

        static int IndexOf(List<string> list, string path, int start)
        {
            for (int i = start; i < list.Count; i++)
                if (string.Equals(list[i], path, StringComparison.OrdinalIgnoreCase)) return i;
            return -1;
        }

        public void Save(TextWriter w)
        {
            w.WriteLine(Header);
            w.WriteLine("round=" + Round);
            w.WriteLine("played=" + MatchesPlayed);
            w.WriteLine("pos=" + _pos);
            w.WriteLine("[round]");
            foreach (var p in _round) w.WriteLine(p);
            w.WriteLine("[winners]");
            foreach (var p in _winners) w.WriteLine(p);
        }

        public static Tournament Load(TextReader r)
        {
            if (r.ReadLine() != Header) return null;
            var t = new Tournament();
            List<string> section = null;
            string line;
            while ((line = r.ReadLine()) != null)
            {
                if (line.Length == 0) continue;
                if (line == "[round]") { section = t._round; continue; }
                if (line == "[winners]") { section = t._winners; continue; }
                if (section != null) { section.Add(line); continue; }
                int eq = line.IndexOf('=');
                if (eq < 0) return null;
                int value;
                if (!int.TryParse(line.Substring(eq + 1), out value)) return null;
                switch (line.Substring(0, eq))
                {
                    case "round": t.Round = value; break;
                    case "played": t.MatchesPlayed = value; break;
                    case "pos": t._pos = value; break;
                }
            }
            if (t._pos < 0 || t._pos > t._round.Count) return null;
            t.RebuildAlive();
            t.Advance();
            return t;
        }
    }
}
