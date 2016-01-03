using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Microsoft.Win32;
using System.IO;

namespace TestWpfSVM
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void ArimaGA_Click(object sender, RoutedEventArgs e)
        {
            new ArimaGA().Show();
        }

        private void NetGA_Click(object sender, RoutedEventArgs e)
        {
            new NetGAxaml().Show();
        }

        private void SVM_Click(object sender, RoutedEventArgs e)
        {
            new SVMWindow().Show();
        }

        private void button1_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                TimeSeriGenerator<float> timeSeriGenerator = new TimeSeriGenerator<float>();
                if (timeSeriGenerator.load(Int32.Parse(NumberOfInpTextBox.Text)))
                {
                    timeSeriGenerator.generate().write2FileWithBrowse();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message + "\n" + ex.StackTrace, "ERROR", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
        
        private void Arima_SVM_Click(object sender, RoutedEventArgs e)
        {
            new HybridArimaSVM().Show();
        }

        private void Complex_Model_Click(object sender, RoutedEventArgs e)
        {
            new HybridComplexModel().Show();
        }

        private void Best_Hyrbid_Click(object sender, RoutedEventArgs e)
        {
            new BestHybrid().Show();
        }
    }
}
