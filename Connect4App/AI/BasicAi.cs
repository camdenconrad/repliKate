using System;
using System.Collections.Generic;
using System.Linq;
using Connect4App.Game;

namespace Connect4App.AI;

public class BasicAi
{
    private readonly Random _rng = new();

    public int ChooseMove(Connect4Board board)
    {
        var legal = board.LegalMoves();
        if (legal.Count == 0) return -1;

        // Shuffle legal moves to introduce variability in tactic checks
        Shuffle(legal);

        // Immediate win (randomized due to shuffled order)
        foreach (var c in legal)
        {
            var b2 = board.Clone();
            b2.Drop(c, out _);
            if (b2.CheckWin(out var w) && w != Cell.Empty)
                return c;
        }

        // Block opponent's immediate win: collect all winning opponent columns, choose one at random
        var opponent = board.CurrentPlayer == Cell.P1 ? Cell.P2 : Cell.P1;
        var oppWins = new List<int>();
        foreach (var oc in legal)
        {
            var oppBoard = board.Clone();
            oppBoard.SetCurrentPlayer(opponent);
            oppBoard.Drop(oc, out _);
            if (oppBoard.CheckWin(out var ow) && ow == opponent)
            {
                oppWins.Add(oc);
            }
        }
        if (oppWins.Count > 0)
        {
            return oppWins[_rng.Next(oppWins.Count)];
        }

        // Center preference with randomness among closest columns
        int center = 3;
        int minDist = legal.Min(a => Math.Abs(a - center));
        var best = legal.Where(a => Math.Abs(a - center) == minDist).ToList();
        return best[_rng.Next(best.Count)];
    }

    private void Shuffle<T>(IList<T> list)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = _rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}
