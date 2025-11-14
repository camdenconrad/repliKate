using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using Avalonia.Media;
using Connect4App.Controllers;
using Connect4App.Game;

namespace Connect4App.ViewModels;

public class CellVm
{
    public required IBrush Color { get; init; }
    public required string Tooltip { get; init; }
    public double Opacity { get; init; } = 1.0;
}

public class MainViewModel : INotifyPropertyChanged
{
    public ObservableCollection<ObservableCollection<CellVm>> Rows { get; } = new();

    private readonly GameController _controller = new();

    private string _status = "Ready";
    public string Status { get => _status; set { _status = value; OnPropertyChanged(); } }

    private string _stats = "Tensor: 0 | Basic: 0 | Draws: 0";
    public string Stats { get => _stats; set { _stats = value; OnPropertyChanged(); } }

    private string _winnerText = "—";
    public string WinnerText { get => _winnerText; set { _winnerText = value; OnPropertyChanged(); } }

    private IBrush _winnerBrush = Brushes.Transparent;
    public IBrush WinnerBrush { get => _winnerBrush; set { _winnerBrush = value; OnPropertyChanged(); } }

    private double _speed = 4.0;
    public double Speed
    {
        get => _speed;
        set
        {
            if (Math.Abs(_speed - value) > 0.001)
            {
                _speed = value;
                _controller.SetSpeed(_speed);
                OnPropertyChanged();
            }
        }
    }

    public ICommand StartCommand { get; }
    public ICommand StopCommand { get; }
    public ICommand StepCommand { get; }
    public ICommand ResetCommand { get; }
    public ICommand SaveModelCommand { get; }

    public MainViewModel()
    {
        // Initialize board rows (top row last for display: we will show from top to bottom)
        for (int r = Connect4Board.Rows - 1; r >= 0; r--)
        {
            var row = new ObservableCollection<CellVm>();
            for (int c = 0; c < Connect4Board.Columns; c++)
                row.Add(new CellVm { Color = Brushes.LightGray, Tooltip = $"({c},{r})" });
            Rows.Add(row);
        }

        _controller.BoardUpdated += UpdateBoard;
        _controller.StatsUpdated += UpdateStats;
        _controller.WinnerUpdated += UpdateWinner;
        _controller.StatusMessage += msg => Status = msg;

        StartCommand = new RelayCommand(_ => { _controller.Start(); }, _ => true);
        StopCommand = new RelayCommand(_ => { _controller.Stop(); }, _ => true);
        StepCommand = new RelayCommand(_ => { _controller.PlayAIMove(); }, _ => true);
        ResetCommand = new RelayCommand(_ => { _controller.ResetGame(); UpdateBoard(); }, _ => true);
        SaveModelCommand = new RelayCommand(_ => { _controller.SaveModel(); }, _ => true);

        // prime first state for sequence and UI
        _controller.ResetGame();
        UpdateBoard();
        UpdateStats();
        _controller.SetSpeed(_speed);
    }

    private void UpdateBoard()
    {
        var board = _controller.Board;
        for (int displayRow = 0; displayRow < Connect4Board.Rows; displayRow++)
        {
            int y = Connect4Board.Rows - 1 - displayRow; // translate UI row to logical row
            for (int x = 0; x < Connect4Board.Columns; x++)
            {
                var cell = board[x, y];
                IBrush brush = cell switch
                {
                    Cell.P1 => Brushes.Crimson,
                    Cell.P2 => Brushes.Goldenrod,
                    _ => Brushes.LightGray
                };
                double opacity = cell == Cell.Empty ? 0.35 : 1.0;
                Rows[displayRow][x] = new CellVm
                {
                    Color = brush,
                    Tooltip = $"({x},{y}) {(cell == Cell.P1 ? "P1" : cell == Cell.P2 ? "P2" : "Empty")}",
                    Opacity = opacity
                };
            }
        }
        OnPropertyChanged(nameof(Rows));
    }

    private void UpdateStats()
    {
        Stats = $"Tensor: {_controller.TensorWins} | Basic: {_controller.BasicWins} | Draws: {_controller.Draws}";
    }

    private void UpdateWinner()
    {
        WinnerText = string.IsNullOrWhiteSpace(_controller.LastResultText) ? "" : _controller.LastResultText;
        WinnerBrush = _controller.LastWinner switch
        {
            Cell.P1 => Brushes.Crimson,
            Cell.P2 => Brushes.Goldenrod,
            _ => Brushes.Gray
        };
    }

    public event PropertyChangedEventHandler? PropertyChanged;
    protected void OnPropertyChanged([CallerMemberName] string? name = null) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

public class RelayCommand : ICommand
{
    private readonly Action<object?> _execute;
    private readonly Predicate<object?>? _canExecute;

    public RelayCommand(Action<object?> execute, Predicate<object?>? canExecute = null)
    {
        _execute = execute;
        _canExecute = canExecute;
    }

    public bool CanExecute(object? parameter) => _canExecute?.Invoke(parameter) ?? true;

    public void Execute(object? parameter) => _execute(parameter);

    public event EventHandler? CanExecuteChanged
    {
        add { }
        remove { }
    }
}
