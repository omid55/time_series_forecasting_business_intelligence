using System;
using System.Collections.Generic;
using System.IO;
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

namespace TestWpfSVM
{
    /// <summary>
    /// Interaction logic for NetGAxaml.xaml
    /// </summary>
    public partial class NetGAxaml : Window
    {
        private MLApp.MLApp matlab;

        public NetGAxaml()
        {
            InitializeComponent();

            matlab = new MLApp.MLApp();
            matlab.Visible = 0;
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            matlab.Quit();
        }

        private void button1_Click(object sender, RoutedEventArgs e)
        {
            string filePath = "NetGA\\MyGA.m";
            StreamReader sr = new StreamReader(filePath);
            string content = sr.ReadToEnd();
            StreamWriter sw=new StreamWriter("NetGA_Output.txt");
            string result = matlab.Execute(content);
            sw.Write(result);
            sw.Flush();
            sw.Close();
        }
    }
}
