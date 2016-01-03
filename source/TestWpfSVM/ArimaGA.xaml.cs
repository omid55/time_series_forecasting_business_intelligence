using System;
using System.Diagnostics;
using System.Reflection;
using System.Windows;
using System.Windows.Input;
using System.IO;
using Microsoft.Win32;
using Extreme.Mathematics;
using Extreme.Statistics;
using Extreme.Statistics.TimeSeriesAnalysis;
using Microsoft.CSharp;
using Vector = Extreme.Mathematics.Vector;

namespace TestWpfSVM
{
    /// <summary>
    /// Interaction logic for ExecCommandWindow.xaml
    /// </summary>
    public partial class ArimaGA : Window
    {
        private MLApp.MLApp matlab;
        private const int NUMBER_OF_TEST_CASES = 5;
        private StreamWriter logger;
        public int bestP, bestD, bestQ;
        public double[] train, test;

        public ArimaGA()
        {
            InitializeComponent();

            matlab = new MLApp.MLApp();
            matlab.Visible = 0;
        }

        public void loadFromMFile()
        {
            try
            {
                OpenFileDialog ofd = new OpenFileDialog();
                ofd.InitialDirectory = Environment.CurrentDirectory;
                ofd.Filter = "M Files (*.m)|*.m|All Files (*.*)|*.*";
                if (ofd.ShowDialog().Value)
                {
                    string filePath = ofd.FileName;
                    StreamReader sr = new StreamReader(filePath);
                    string content = sr.ReadToEnd();
                    MessageBox.Show(matlab.Execute(content));

                    MessageBox.Show("Your File Loaded Successfully .");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("An Error Occurred In loadFromMFile Function , \n\n" + ex.Message, "ERROR",
                                MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        public void execMatlabCode(string commands)
        {
            ResultListBox.Items.Add(matlab.Execute(commands));
        }

        public void load(string filename)
        {
            using (
                Stream stream =
                    Assembly.GetExecutingAssembly().GetManifestResourceStream("TestWpfSVM.MatlabCodes." +
                                                                              filename))
            {
                using (StreamReader sr = new StreamReader(stream))
                {
                    string content = sr.ReadToEnd();
                    sr.Close();
                    execMatlabCode(content);
                }
            }
        }

        private void ExecuteButton_Click(object sender, RoutedEventArgs e)
        {
            string result = matlab.Execute(CommandTextBox.Text);
            ResultListBox.Items.Add(result);
        }

        private void CommandTextBox_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                ExecuteButton_Click(null, null);
            }
        }

        private void button1_Click(object sender, RoutedEventArgs e)
        {
            loadFromMFile();
        }

        public string getMyVariable(string name)
        {
            string s = matlab.Execute(name);
            s = s.Substring(11);
            return s.Remove(s.Length - 2);
        }

        private void button2_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                double[] data = LoadData();

                StartArima(data);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message + "\n" + ex.StackTrace, "ERROR", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        public ArimaModel GetBestModel(double[] data)
        {
            StartArima(data);

            NumericalVariable newInput = new NumericalVariable("NewInput", train);
            ArimaModel arimaModel = new ArimaModel(newInput, this.bestP, this.bestD, this.bestQ);
            arimaModel.Compute();

            return arimaModel;
        }

        public void StartArima(double[] data)
        {
            train = new double[data.Length - NUMBER_OF_TEST_CASES];
            test = new double[NUMBER_OF_TEST_CASES];
            for (int i = 0; i < train.Length; i++)
            {
                train[i] = data[i];
            }
            for (int i = train.Length, j = 0; i < data.Length; i++, j++)
            {
                test[j] = data[i];
            }

            logger = new StreamWriter("ArimaGALog.txt");

            load("MyGA1.m");
            int n = (int)matlab.GetVariable("n", "base");
            bool endCondition = true;
            StreamWriter sw = new StreamWriter("ArimaGA_Output.txt");
            do
            {
                string command = "\n y=[";
                for (int i = 1; i <= n; i++)
                {
                    double pp = double.Parse(getMyVariable("p(" + i + ",1)"));
                    double dd = double.Parse(getMyVariable("p(" + i + ",2)"));
                    double qq = double.Parse(getMyVariable("p(" + i + ",3)"));
                    int p = (int)Math.Round(pp);
                    int d = (int)Math.Round(dd);
                    int q = (int)Math.Round(qq);
                    double error = arimaModelFunction(p, d, q, CompareComboBox.SelectedIndex);
                    logger.WriteLine("{0} {1} {2} ->  {3}", p, d, q, error);
                    command += error + ",";
                }
                logger.WriteLine("\n\n\n");
                command = command.Remove(command.Length - 1) + "]; \n";
                string matlabCode = justLoadAndGetContent("MyGA2.m") + command + justLoadAndGetContent("MyGA3.m");
                string result = matlab.Execute(matlabCode);
                sw.WriteLine(result);
                matlab.Execute(
                    "EndCondition = m<1000 && abs(maxvalue(m)-maxvalue(m-(m0-1)))>0.001*maxvalue(m) && (abs(maxvalue(m))>1e-10 && abs(maxvalue(m-(m0-1)))>1e-10) && m<10000 && abs(maxvalue(m)-meanvalue(m))>1e-5 || m<20");
                endCondition = (bool)matlab.GetVariable("EndCondition", "base");
            } while (endCondition);

            bestP = (int)Math.Round(float.Parse(getMyVariable("NowP")));
            bestD = (int)Math.Round(float.Parse(getMyVariable("NowD")));
            bestQ = (int)Math.Round(float.Parse(getMyVariable("NowQ")));
            double mse = -1 * arimaModelFunction(bestP, bestD, bestQ, 0);
            double errorPercent = -1 * arimaModelFunction(bestP, bestD, bestQ, 1);
            sw.WriteLine("\n\n\n Best P & D & Q are => with this MSE & Error%\n\n {0} {1} {2} => {3} {4}", bestP,
                         bestD, bestQ, mse, errorPercent);
            sw.Flush();
            sw.Close();
            MessageBox.Show("Done", "GA with Arima", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        public string justLoadAndGetContent(string embedFileName)
        {
            using (
                Stream stream =
                    Assembly.GetExecutingAssembly().GetManifestResourceStream("TestWpfSVM.MatlabCodes." +
                                                                              embedFileName))
            {
                using (StreamReader sr = new StreamReader(stream))
                {
                    string content = sr.ReadToEnd();
                    sr.Close();
                    return content;
                }
            }
        }

        private double[] LoadData()
        {
            double[] data = null;
            try
            {
                int number = Int32.Parse(ColumnNumberTextBox.Text);
                var timeSeriGenerator = new TimeSeriGenerator<double>();
                timeSeriGenerator.load(number);
                data = timeSeriGenerator.TimeSeri;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "ERROR", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            return data;
        }

        public double arimaModelFunction(int p, int d, int q,int errorMode)
        {
            NumericalVariable trainData = new NumericalVariable("trainData", train);

            // ARMA models (no differencing) are constructed from
            // the variable containing the time series data, and the
            // AR and MA orders. The following constructs an ARMA(2,1)
            // model:
            ArimaModel model = new ArimaModel(trainData, p, d, q);

            // The Compute methods fits the model.
            model.Compute();

            // or to predict a specified number of values:
            Vector nextValues = model.Forecast(NUMBER_OF_TEST_CASES);

            switch (errorMode)
            {
                case 0:     // MSE
                    return -1 * MyErrorParameters.MSE(nextValues.ToArray(), test);

                case 1:     // ERROR %
                    return -1 * MyErrorParameters.ERROR_Percent(nextValues.ToArray(), test);
            }
            return -9999999999;     // a very bad fitness when error occured
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            matlab.Quit();
            logger.Flush();
            logger.Close();
        }
    }
}