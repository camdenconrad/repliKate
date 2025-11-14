using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Connect4App.Game;
using repliKate;

namespace Connect4App.AI;

public class TensorAi
{
    private readonly TensorSequenceTree _tree;
    private readonly int _boardSize = Connect4Board.Columns * Connect4Board.Rows * 3; // one-hot per cell (empty, p1, p2)
    private const int MoveSize = Connect4Board.Columns; // 7
    private const int TypeSize = 2; // [state, move]
    private readonly int _tensorSize;

    public List<Tensor> CurrentSequence { get; } = new();

    public TensorAi(TensorSequenceTree? tree = null)
    {
        _tensorSize = _boardSize + MoveSize + TypeSize;
        _tree = tree ?? new TensorSequenceTree(maxContextWindow: 50, similarityThreshold: 0.97f);
    }

    public int ChooseMove(Connect4Board board)
    {
        // Build context: ensure the latest state tensor is included
        var context = CurrentSequence.ToArray();
        var stateTensor = EncodeState(board);
        var extended = context.Concat(new[] { stateTensor }).ToArray();

        // Ask prediction
        var predicted = _tree.PredictNext(extended, count: 1, useBlending: true);
        int chosen = ExtractLegalMove(predicted?.FirstOrDefault(), board);
        if (chosen == -1)
        {
            // Fallback to top candidates list
            var tops = _tree.GetTopPredictions(extended, topN: 5);
            foreach (var (tensor, _) in tops)
            {
                chosen = ExtractLegalMove(tensor, board);
                if (chosen != -1) break;
            }
        }
        return chosen;
    }

    private int ExtractLegalMove(Tensor? t, Connect4Board board)
    {
        if (t == null) return -1;
        int offset = _boardSize;
        int bestIdx = -1;
        float bestVal = float.NegativeInfinity;
        var legal = board.LegalMoves();
        for (int i = 0; i < MoveSize; i++)
        {
            float v = t.Data[offset + i];
            if (legal.Contains(i) && v > bestVal)
            {
                bestVal = v;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    public Tensor EncodeState(Connect4Board board)
    {
        var t = new Tensor(_tensorSize);
        int idx = 0;
        for (int y = 0; y < Connect4Board.Rows; y++)
        for (int x = 0; x < Connect4Board.Columns; x++)
        {
            var c = board[x, y];
            // one-hot per cell: empty, p1, p2
            t.Data[idx + 0] = c == Cell.Empty ? 1f : 0f;
            t.Data[idx + 1] = c == Cell.P1 ? 1f : 0f;
            t.Data[idx + 2] = c == Cell.P2 ? 1f : 0f;
            idx += 3;
        }
        // moves area is zeros for state
        idx = _boardSize;
        for (int i = 0; i < MoveSize; i++) t.Data[idx + i] = 0f;
        // type flag: [1,0] for state
        idx = _boardSize + MoveSize;
        t.Data[idx + 0] = 1f;
        t.Data[idx + 1] = 0f;
        return t;
    }

    public Tensor EncodeMove(int column)
    {
        var t = new Tensor(_tensorSize);
        // board part zeros (we could also copy recent board, but keep zeros to distinguish)
        // move one-hot
        int idx = _boardSize;
        for (int i = 0; i < MoveSize; i++) t.Data[idx + i] = i == column ? 1f : 0f;
        // type flag: [0,1] for move
        idx = _boardSize + MoveSize;
        t.Data[idx + 0] = 0f;
        t.Data[idx + 1] = 1f;
        return t;
    }

    public void RecordState(Connect4Board board)
    {
        CurrentSequence.Add(EncodeState(board));
    }

    public void RecordMove(int column)
    {
        CurrentSequence.Add(EncodeMove(column));
    }

    public void LearnFromCurrentSequence()
    {
        if (CurrentSequence.Count > 1)
        {
            _tree.Learn(CurrentSequence.ToArray());
        }
    }

    public void ClearSequence() => CurrentSequence.Clear();

    public TensorSequenceTree GetTree() => _tree;

    public int TensorSize => _tensorSize;
}
