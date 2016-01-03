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
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.IO;
using Emgu.CV;
using Emgu.CV.ML;
using Emgu.CV.Structure;
using Microsoft.Win32;
using Emgu.CV.ML.MlEnum;

namespace TestWpfSVM
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class SVMWindow : Window
    {
        private StreamWriter writer;
        private float[] targets;


        public SVMWindow()
        {
            InitializeComponent();
        }

        public void Start()
        {
            writer = new StreamWriter("SVM_Output.txt");

            #region Loading the training data and classes and test data and test classes

            Matrix<float> trainData = null;
            Matrix<float> trainClasses = null;
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Title = "Please Open Your Train File :";
            ofd.InitialDirectory = Environment.CurrentDirectory;
            ofd.Filter = "Text Files (*.txt)|*.txt|All Files (*.*)|*.*";
            if (ofd.ShowDialog().Value)
            {
                LoadData(ofd.FileName, ref trainData, ref trainClasses);
            }

            Matrix<float> testData = null;
            Matrix<float> testClasses = null;
            ofd.Title = "Now Please Open Your Test File :";
            if (ofd.ShowDialog().Value)
            {
                LoadData(ofd.FileName, ref testData, ref testClasses);
            }

            #endregion


            #region creating and training the svm model

            double minError = 9999999;
            SVM_KERNEL_TYPE bestKernelType = SVM_KERNEL_TYPE.LINEAR;
            double bestEps = 0.1;

            foreach (Emgu.CV.ML.MlEnum.SVM_KERNEL_TYPE tp in Enum.GetValues(typeof(Emgu.CV.ML.MlEnum.SVM_KERNEL_TYPE)))
            {
                for (double eps = 0.1; eps >= 0.00001; eps *= 0.1)
                {
                    using (SVM model = new SVM())
                    {
                        SVMParams p = new SVMParams();
                        p.KernelType = tp;
                        p.SVMType = Emgu.CV.ML.MlEnum.SVM_TYPE.EPS_SVR; // for regression
                        p.C = 1;
                        p.TermCrit = new MCvTermCriteria(100, eps);
                        p.Gamma = 1;
                        p.Degree = 1;
                        p.P = 1;
                        p.Nu = 0.1;

                        //these just work with rounded trainClasses data
                        //bool trained = model.Train(trainData, trainClasses, null, null, p);
                        bool trained = model.TrainAuto(trainData, trainClasses, null, null, p.MCvSVMParams, 10);


                        double error = getSumError(model, testData, testClasses,false);
                        if (trained && minError > error)
                        {
                            minError = error;

                            bestEps = eps;
                            bestKernelType = tp;
                        }
                    }
                }
            }

            using (SVM model = new SVM())
            {
                SVMParams p = new SVMParams();
                p.KernelType = bestKernelType;
                p.SVMType = Emgu.CV.ML.MlEnum.SVM_TYPE.EPS_SVR; // for regression
                p.C = 1;
                p.TermCrit = new MCvTermCriteria(100, bestEps);
                p.Gamma = 1;
                p.Degree = 1;
                p.P = 1;
                p.Nu = 0.1;

                //these just work with rounded trainClasses data
                //bool trained = model.Train(trainData, trainClasses, null, null, p);
                bool trained = model.TrainAuto(trainData, trainClasses, null, null, p.MCvSVMParams, 10);

                double error = getSumError(model, testData, testClasses,true);
            }

            writer.Flush();
            writer.Close();
            MessageBox.Show("Done", "SVM", MessageBoxButton.OK, MessageBoxImage.Information);

            #endregion
        }

        public double getSumError(SVM model, Matrix<float> testInputs, Matrix<float> testOutputs, bool write2File)
        {
            double sum = 0;
            List<float> results = new List<float>();
            for (int i = 0; i < testOutputs.Rows; i++)
            {
                float res = model.Predict(testInputs.GetRow(i));
                double diff2 = Math.Pow((testOutputs.Data[i, 0] - res), 2);
                if (write2File)
                {
                    results.Add(res);
                    writer.WriteLine(res);
                }
                sum += diff2;
            }
            if (write2File)
            {
                double mse = MyErrorParameters.MSE(results.ToArray(),targets);
                double errorPercent = MyErrorParameters.ERROR_Percent(results.ToArray(), targets);
                writer.WriteLine("\n\n\nMSE & ERROR% are =>\n\n{0} {1}",mse,errorPercent);
            }
            return sum / testOutputs.Rows;
        }

        public static Dimension getSizeOfFile(string filePath)
        {
            StreamReader sr = new StreamReader(filePath);
            string line = sr.ReadLine(); // ignore labels in first line
            int m = 0;
            int n = 0;
            while (!String.IsNullOrEmpty(line) && !sr.EndOfStream)
            {
                m++;
                line = sr.ReadLine();
                if (n == 0 && line != null)
                {
                    string[] res = line.Split('\t');
                    while (n < res.Length && !String.IsNullOrEmpty(res[n]))
                    {
                        n++;
                    }
                    n--;
                }
            }
            return new Dimension(m, n);
        }

        public void LoadData(string filePath, ref Matrix<float> inputData, ref Matrix<float> outputData)
        {
            float[,] inputs = null;
            float[,] outputs = null;
            try
            {
                Dimension dim = getSizeOfFile(filePath);
                inputs = new float[dim.m, dim.n];
                outputs = new float[dim.m, 1];
                targets = new float[dim.m];
                int cnt = 0;

                StreamReader sr = new StreamReader(filePath);
                sr.ReadLine(); // ignore labels in first line
                while (!sr.EndOfStream)
                {
                    string line = sr.ReadLine();
                    if (line != null)
                    {
                        string[] res = line.Split('\t');
                        for (int i = 0; i < dim.n; i++)
                        {
                            inputs[cnt, i] = float.Parse(res[i]);
                        }
                        outputs[cnt, 0] = targets[cnt] = float.Parse(res[dim.n]);
                    }
                    cnt++;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "ERROR", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            inputData = new Matrix<float>(inputs);
            outputData = new Matrix<float>(outputs);
        }

        public static void getResidual(SVM model, Matrix<float> trainData, Matrix<float> trainClasses)
        {
            for (int i = 0; i < trainClasses.Rows; i++)
            {
                double diff = trainClasses.Data[i, 0] - model.Predict(trainData.GetRow(i));
                Console.WriteLine(diff);
            }
        }

        private void button1_Click(object sender, RoutedEventArgs e)
        {
            Start();
        }
    }
}
