using System;
using System.Collections.Generic;
using System.Linq;

namespace Connect4App.Game;

public enum Cell { Empty = 0, P1 = 1, P2 = 2 }

public class Connect4Board
{
    public const int Columns = 7;
    public const int Rows = 6;

    private readonly Cell[,] _cells = new Cell[Columns, Rows];

    public Cell CurrentPlayer { get; private set; } = Cell.P1;

    public Cell this[int x, int y] => _cells[x, y];

    public void SetCurrentPlayer(Cell player) => CurrentPlayer = player;

    public Connect4Board Clone()
    {
        var b = new Connect4Board { CurrentPlayer = CurrentPlayer };
        Array.Copy(_cells, b._cells, _cells.Length);
        return b;
    }

    public void Clear()
    {
        Array.Clear(_cells, 0, _cells.Length);
        CurrentPlayer = Cell.P1;
    }

    public List<int> LegalMoves()
    {
        var list = new List<int>(Columns);
        for (int c = 0; c < Columns; c++)
        {
            if (_cells[c, Rows - 1] == Cell.Empty) list.Add(c);
        }
        return list;
    }

    public bool IsFull()
    {
        for (int c = 0; c < Columns; c++)
            if (_cells[c, Rows - 1] == Cell.Empty)
                return false;
        return true;
    }

    public bool Drop(int column, out int rowPlaced)
    {
        rowPlaced = -1;
        if (column < 0 || column >= Columns) return false;
        for (int r = 0; r < Rows; r++)
        {
            if (_cells[column, r] == Cell.Empty)
            {
                _cells[column, r] = CurrentPlayer;
                rowPlaced = r;
                CurrentPlayer = CurrentPlayer == Cell.P1 ? Cell.P2 : Cell.P1;
                return true;
            }
        }
        return false;
    }

    public static bool CheckWin(Cell[,] cells, out Cell winner)
    {
        // Horizontal, vertical, and two diagonals
        int cols = cells.GetLength(0);
        int rows = cells.GetLength(1);
        for (int x = 0; x < cols; x++)
        for (int y = 0; y < rows; y++)
        {
            var c = cells[x, y];
            if (c == Cell.Empty) continue;
            // Right
            if (x + 3 < cols && c == cells[x + 1, y] && c == cells[x + 2, y] && c == cells[x + 3, y]) { winner = c; return true; }
            // Up
            if (y + 3 < rows && c == cells[x, y + 1] && c == cells[x, y + 2] && c == cells[x, y + 3]) { winner = c; return true; }
            // Up-right
            if (x + 3 < cols && y + 3 < rows && c == cells[x + 1, y + 1] && c == cells[x + 2, y + 2] && c == cells[x + 3, y + 3]) { winner = c; return true; }
            // Up-left
            if (x - 3 >= 0 && y + 3 < rows && c == cells[x - 1, y + 1] && c == cells[x - 2, y + 2] && c == cells[x - 3, y + 3]) { winner = c; return true; }
        }
        winner = Cell.Empty;
        return false;
    }

    public bool CheckWin(out Cell winner) => CheckWin(_cells, out winner);
}
