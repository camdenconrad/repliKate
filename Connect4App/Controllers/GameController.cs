using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Security.Cryptography;
using Avalonia.Threading;
using Connect4App.AI;
using Connect4App.Game;
using repliKate;

namespace Connect4App.Controllers;

public enum MatchMode { TensorVsBasic = 0, TensorVsTensor = 1 }

public class GameController
{
    public Cell LastWinner { get; private set; } = Cell.Empty;
    public string LastResultText { get; private set; } = "";
    public Connect4Board Board { get; } = new();
    public BasicAi Basic { get; } = new();
    public TensorAi TensorAi { get; private set; }

    public int TensorWins { get; private set; }
    public int BasicWins { get; private set; }
    public int Draws { get; private set; }

    public MatchMode Mode { get; private set; } = MatchMode.TensorVsBasic;

    public bool IsRunning => _timer.IsEnabled;

    private readonly DispatcherTimer _timer;
    private int _gamesCompleted = 0;
    private readonly TimeSpan _defaultInterval = TimeSpan.FromMilliseconds(250);

    private readonly string _modelPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".connect4_tensor_model.gz");

    public event Action? BoardUpdated;
    public event Action? StatsUpdated;
    public event Action? WinnerUpdated;
    public event Action<string>? StatusMessage;

    private readonly Random _rng = new();

    public GameController()
    {
        // Load model if exists
        TensorSequenceTree? tree = null;
        try
        {
            if (File.Exists(_modelPath))
            {
                tree = TensorSequenceTree.LoadFromFile(_modelPath);
                StatusMessage?.Invoke($"Loaded model: {_modelPath}");
            }
        }
        catch (Exception ex)
        {
            StatusMessage?.Invoke($"Failed to load model: {ex.Message}");
        }

        TensorAi = new TensorAi(tree);

        _timer = new DispatcherTimer { Interval = _defaultInterval };
        _timer.Tick += (_, _) => OnTick();
    }

    public void SetSpeed(double movesPerSecond)
    {
        // Allow very high speeds; at the top end, drive the timer to ~1ms (practical UI-thread limit)
        movesPerSecond = Math.Clamp(movesPerSecond, 0.1, 1000.0);
        if (movesPerSecond >= 999.9)
        {
            _timer.Interval = TimeSpan.FromMilliseconds(1);
        }
        else
        {
            _timer.Interval = TimeSpan.FromMilliseconds(Math.Max(1.0, 1000.0 / movesPerSecond));
        }
    }

    public void SetMode(MatchMode mode)
    {
        if (Mode == mode) return;
        Mode = mode;
        // reset stats to avoid mixing across modes
        TensorWins = 0;
        BasicWins = 0;
        Draws = 0;
        _gamesCompleted = 0;
        LastWinner = Cell.Empty;
        LastResultText = "";
        StatsUpdated?.Invoke();
        WinnerUpdated?.Invoke();
        StatusMessage?.Invoke($"Mode changed to: {(Mode == MatchMode.TensorVsBasic ? "Tensor vs Basic" : "Tensor vs Tensor")}");
        ResetGame();
    }

    private string GetPlayerLabel(Cell player)
    {
        if (player == Cell.P1) return "Tensor (P1)";
        // P2 label depends on mode
        return Mode == MatchMode.TensorVsBasic ? "Basic (P2)" : "Tensor (P2)";
    }

    public void Start()
    {
        if (!_timer.IsEnabled) _timer.Start();
        StatusMessage?.Invoke("Auto-play started");
    }

    public void Stop()
    {
        if (_timer.IsEnabled) _timer.Stop();
        StatusMessage?.Invoke("Auto-play stopped");
    }

    public void ResetGame()
    {
        Stop();
        Board.Clear();
        // Randomize starting player each new game
        var starter = RandomNumberGenerator.GetInt32(0, 2) == 0 ? Cell.P1 : Cell.P2;
        Board.SetCurrentPlayer(starter);
        TensorAi.ClearSequence();
        TensorAi.RecordState(Board);
        BoardUpdated?.Invoke();
        StatusMessage?.Invoke($"Game reset — { GetPlayerLabel(starter) } starts");
    }

    private void OnTick()
    {
        PlayAIMove();
    }

    public void PlayAIMove()
    {
        if (Board.CheckWin(out _) || Board.IsFull())
        {
            EndGameAndTrain();
            return;
        }

        int move = -1;
        if (Board.CurrentPlayer == Cell.P1)
        {
            // Tensor AI is Player 1
            move = TensorAi.ChooseMove(Board);
            if (move == -1)
            {
                // fallback to simple heuristic if no prediction
                move = new BasicAi().ChooseMove(Board);
            }
        }
        else
        {
            move = Basic.ChooseMove(Board);
        }

        if (move == -1)
        {
            // No legal moves - treat as draw
            EndGameAndTrain();
            return;
        }

        // Record intended move for training (record both players to learn enemy patterns)
        TensorAi.RecordMove(move);

        Board.Drop(move, out _);
        // After move, record new state for sequence
        TensorAi.RecordState(Board);

        if (Board.CheckWin(out var winner) || Board.IsFull())
        {
            EndGameAndTrain(winner);
            return;
        }

        BoardUpdated?.Invoke();
    }

    private void EndGameAndTrain(Cell winner = Cell.Empty)
    {
        LastWinner = winner;
        if (winner == Cell.P1)
        {
            TensorWins++;
            LastResultText = "Winner: Tensor (P1)";
        }
        else if (winner == Cell.P2)
        {
            if (Mode == MatchMode.TensorVsBasic)
            {
                BasicWins++;
            }
            else
            {
                TensorWins++; // In Tensor vs Tensor, count P2 wins for Tensor as well
            }
            LastResultText = $"Winner: {GetPlayerLabel(Cell.P2)}";
        }
        else
        {
            Draws++;
            LastResultText = "Draw";
        }

        try
        {
            TensorAi.LearnFromCurrentSequence();
            _gamesCompleted++;
            // Only save every 15 games
            if (_gamesCompleted % 100 == 0)
            {
                SaveModel();
            }
        }
        catch (Exception ex)
        {
            StatusMessage?.Invoke($"Training error: {ex.Message}");
        }

        WinnerUpdated?.Invoke();
        StatsUpdated?.Invoke();
        BoardUpdated?.Invoke();
        // Start a new game automatically if running
        Board.Clear();
        // Randomize starting player for the next game
        var starter = RandomNumberGenerator.GetInt32(0, 2) == 0 ? Cell.P1 : Cell.P2;
        Board.SetCurrentPlayer(starter);
        TensorAi.ClearSequence();
        TensorAi.RecordState(Board);
        StatusMessage?.Invoke($"New game — { (starter == Cell.P1 ? "Tensor (P1)" : "Basic (P2)") } starts");
    }

    public void SaveModel()
    {
        try
        {
            TensorAi.GetTree().SaveToFile(_modelPath);
            StatusMessage?.Invoke($"Model saved: {_modelPath}");
        }
        catch (Exception ex)
        {
            StatusMessage?.Invoke($"Failed to save model: {ex.Message}");
        }
    }
}

internal static class TensorAiExtensions
{
    public static void RecordMoveIfTensorPlayer(this TensorAi ai, Connect4Board board, int column)
    {
        if (board.CurrentPlayer == Cell.P1)
        {
            ai.RecordMove(column);
        }
    }
}
