using Avalonia.Controls;
using Connect4App.ViewModels;

namespace Connect4App;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        DataContext = new MainViewModel();
    }
}
