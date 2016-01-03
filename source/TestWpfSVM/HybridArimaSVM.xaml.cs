using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Emgu.CV.ML;
using Emgu.CV.Structure;
using Extreme.Statistics;
using Extreme.Statistics.TimeSeriesAnalysis;
using Microsoft.Win32;
using Emgu.CV;

namespace TestWpfSVM
{
    /// <summary>
    /// Interaction logic for HybridArimaSVM.xaml
    /// </summary>
    public partial class HybridArimaSVM : Window
    {
        private double[] train;
        private float[] test;
        private float[] testHybrid;
        private const int NUMBER_OF_HYBRID_TEST = 5;
        private double[] trainArima;
        private double[] testArima;

        private MLApp.MLApp matlab;
        private SVM svmModel = null;
        private ArimaModel arimaModel = null;

        private TimeSeriGenerator<float> timeSeriGenerator;

        private static string outputFileName = "Hybrid_Result_Arima_SVM_Output.txt";
        private StreamWriter outputWriter;


        public HybridArimaSVM()
        {
            InitializeComponent();

            matlab=new MLApp.MLApp();
            matlab.Visible = 0;
        }

        private void StartButton_Click(object sender, RoutedEventArgs e)
        {
            Start();
        }

        private void Start()
        {
            outputWriter = new StreamWriter(outputFileName);

            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Title = "Please Open Your Train File :";
            ofd.InitialDirectory = Environment.CurrentDirectory;
            ofd.Filter = "Text Files (*.txt)|*.txt|All Files (*.*)|*.*";
            if (ofd.ShowDialog().Value)
            {
                LoadData(ofd.FileName);
            }

            StartArima(train);

            StartSVM(getResidual().ToArray());


            // now our hybrid model is created (Arima + SVM)
            List<float> results = new List<float>();
            var arimaPred = arimaModel.Forecast(test.Length + NUMBER_OF_HYBRID_TEST);
            Queue<float> q = timeSeriGenerator.getLastInputForTesting();
            for (int i = 0; i < NUMBER_OF_HYBRID_TEST; i++)
            {
                var valuePred = arimaPred[test.Length + i];
                float residualPred = svmModel.Predict(new Matrix<float>(q.ToArray()));
                q.Dequeue();
                q.Enqueue(residualPred);
                float resultValue = (float) valuePred + residualPred;
                results.Add(resultValue);
            }
            foreach (var result in results)
            {
                outputWriter.WriteLine(result);
            }
            outputWriter.WriteLine("\n\n");
            outputWriter.WriteLine("MSE =>  " + MyErrorParameters.MSE(results.ToArray(), testHybrid));
            outputWriter.WriteLine("ERROR% =>  " + MyErrorParameters.ERROR_Percent(results.ToArray(), testHybrid));

            outputWriter.Flush();
            outputWriter.Close();

            MessageBox.Show(
                String.Format("Hybrid Model Created File {0} For Output Successfully Now , Please Check It Out .",
                              outputFileName), "Hybrid Arima SVM Done", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        public List<float> getResidual()   // with Arima Model
        {
            List<float > res = new List<float>();
            int size = test.Length;
            var predictedValue = arimaModel.Forecast(size);
            for (int i = 0; i < size; i++)
            {
                float diff = test[i] - (float)predictedValue[i];
                res.Add(diff);
            }
            return res;
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
                    matlab.Execute(content);
                }
            }
        }

        public string getMyVariable(string name)
        {
            string s = matlab.Execute(name);
            s = s.Substring(11);
            return s.Remove(s.Length - 2);
        }

        public void StartArima(double[] data)
        {
            try
            {
                trainArima = new double[data.Length - NUMBER_OF_HYBRID_TEST];
                testArima = new double[NUMBER_OF_HYBRID_TEST];
                for (int i = 0; i < trainArima.Length; i++)
                {
                    trainArima[i] = data[i];
                }
                for (int i = trainArima.Length, j = 0; i < data.Length; i++, j++)
                {
                    testArima[j] = data[i];
                }

                load("MyGA1.m");
                int n = (int)matlab.GetVariable("n", "base");
                bool endCondition = true;
                StreamWriter sw = new StreamWriter("Hybrid_ArimaSVM_Output.txt");
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
                        double error = arimaModelFunction(p, d, q);
                        command += error + ",";
                    }
                    command = command.Remove(command.Length - 1) + "]; \n";
                    string matlabCode = justLoadAndGetContent("MyGA2.m") + command + justLoadAndGetContent("MyGA3.m");
                    string result = matlab.Execute(matlabCode);
                    sw.WriteLine(result);
                    matlab.Execute(
                        "EndCondition = m<1000 && abs(maxvalue(m)-maxvalue(m-(m0-1)))>0.001*maxvalue(m) && (abs(maxvalue(m))>1e-10 && abs(maxvalue(m-(m0-1)))>1e-10) && m<10000 && abs(maxvalue(m)-meanvalue(m))>1e-5 || m<20");
                    endCondition = (bool)matlab.GetVariable("EndCondition", "base");
                } while (endCondition);
                sw.Flush();
                sw.Close();
                MessageBox.Show("Model Created Successfully", "Best SVM with Grid & Best Arima With GA", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message + "\n" + ex.StackTrace, "ERROR1", MessageBoxButton.OK, MessageBoxImage.Error);
            }
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

        public double arimaModelFunction(int p, int d, int q)
        {
            NumericalVariable timeSerii = new NumericalVariable("timeSerii", train);

            // ARMA models (no differencing) are constructed from
            // the variable containing the time series data, and the
            // AR and MA orders.
            arimaModel = new ArimaModel(timeSerii, p, d, q);

            // The Compute methods fits the model.
            arimaModel.Compute();

            return getErrorOfArimaModel();
        }

        private double getErrorOfArimaModel()
        {
            // or to predict a specified number of values:
            Extreme.Mathematics.Vector nextValues = arimaModel.Forecast(NUMBER_OF_HYBRID_TEST);

            switch (CompareComboBox.SelectedIndex)
            {
                case 0: // MSE
                    double mse = 0;
                    for (int i = 0; i < NUMBER_OF_HYBRID_TEST; i++)
                    {
                        mse += Math.Pow(nextValues[i] - testArima[i], 2);
                    }
                    mse /= NUMBER_OF_HYBRID_TEST;
                    return -1 * mse;
                case 1: // ERROR %

                    double errorPercent = 0;
                    double sumTargets = 0;
                    for (int i = 0; i < NUMBER_OF_HYBRID_TEST; i++)
                    {
                        errorPercent += Math.Abs(nextValues[i] - testArima[i]);
                        sumTargets += testArima[i];
                    }
                    errorPercent /= sumTargets;
                    return -1 * errorPercent;
            }
            return -9999999999; // a very bad fitness when error occured
        }

        public static Dimension getSizeOfFile(string filePath)
        {
            StreamReader sr = new StreamReader(filePath);
            sr.ReadLine(); // ignore labels in first line
            int m = 0;
            int n = 0;
            while (!sr.EndOfStream)
            {
                m++;
                string line = sr.ReadLine();
                if (n == 0 && line != null)
                {
                    string[] res = line.Split('\t');
                    n = res.Length - 1;
                }
            }
            return new Dimension(m, n);
        }

        public void StartSVM(float[] data)
        {
            timeSeriGenerator = new TimeSeriGenerator<float>();
            MyTimeSeri<float> myTimeSeri = timeSeriGenerator.generateWithThisData(data, 6);

            Matrix<float> trainData=new Matrix<float>(myTimeSeri.inputs);
            Matrix<float> trainClasses = new Matrix<float>(myTimeSeri.targets);

            svmModel = new SVM();
            SVMParams p = new SVMParams();
            p.KernelType = Emgu.CV.ML.MlEnum.SVM_KERNEL_TYPE.POLY;
            p.SVMType = Emgu.CV.ML.MlEnum.SVM_TYPE.EPS_SVR; // for regression
            p.C = 1;
            p.TermCrit = new MCvTermCriteria(100, 0.00001);
            p.Gamma = 1;
            p.Degree = 1;
            p.P = 1;
            p.Nu = 0.1;

            bool trained = svmModel.TrainAuto(trainData, trainClasses, null, null, p.MCvSVMParams, 10);

        }

        public void LoadData(string filePath)
        {
            try
            {
                Dimension dim = getSizeOfFile(filePath);
                int size = (dim.m - NUMBER_OF_HYBRID_TEST)/2;

                train = new double[size];
                test = new float[size];
                testHybrid = new float[NUMBER_OF_HYBRID_TEST];

                StreamReader sr = new StreamReader(filePath);
                sr.ReadLine(); // ignore labels in first line
                for (int l = 0, k = 0, j = 0; l < dim.m; l++)
                {
                    string line = sr.ReadLine();
                    if (line != null)
                    {
                        string[] res = line.Split('\t');
                        int lastIndex = res.Length - 1;

                        if (l < size)
                        {
                            train[l] = double.Parse(res[lastIndex]);
                        }
                        else if (l < 2*size)
                        {
                            test[k++] = float.Parse(res[lastIndex]);
                        }
                        else
                        {
                            testHybrid[j++] = float.Parse(res[lastIndex]);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "ERROR2", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }
}
