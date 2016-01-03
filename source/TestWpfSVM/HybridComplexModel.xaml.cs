using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System;
using System.Windows;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Emgu.CV;
using Emgu.CV.ML;
using Emgu.CV.ML.MlEnum;
using Emgu.CV.Structure;
using Extreme.Mathematics.Curves;
using Extreme.Statistics;
using Extreme.Statistics.TimeSeriesAnalysis;
using Microsoft.Win32;
using TestWpfSVM.TimeSeriClasses;
using Vector = Extreme.Mathematics.Vector;
using System.Threading;

namespace TestWpfSVM
{
    /// <summary>
    /// Interaction logic for HybridComplexModel.xaml
    /// </summary>
    public partial class HybridComplexModel : Window
    {
        private MLApp.MLApp matlab;
        private const string OUTPUT_FILE_NAME = "Complex_Hybrid_Result_Output.txt";
        private StreamWriter svmWriter;
        private StreamWriter arimaLogger;
        private StreamWriter hybridWriter;

        private double[] trainArima;
        private double[] testArima;

        private SVM svmModel = null;
        private SVM svmModelHybrid = null;
        private ArimaModel arimaModel = null;
        public MyCategorizedTimeSeri<float> myCategorizedTimeSeri;

        public int numberOfTests;
        public int numberOfForecastTests;

        public int maxGAIteretionInArima = 40;   // 1000 must be fine   << CHECK HERE >>
        public int maxGAIteretionInHybrid = 20;    //30   


        public HybridComplexModel()
        {
            InitializeComponent();

            matlab = new MLApp.MLApp();
            matlab.Visible = 0;
        }

        public void Start()
        {
            try
            {
                svmWriter = new StreamWriter("Complex_Hybrid_SVMOutput.txt");
                arimaLogger = new StreamWriter("Complex_Hybrid_ArimaGALog.txt");
                hybridWriter = new StreamWriter(OUTPUT_FILE_NAME);

                #region Loading the training data and classes and test data and test classes

                TimeSeriGenerator<float> timeSeriGenerator = new TimeSeriGenerator<float>();
                int numInp = 0;
                this.Dispatcher.Invoke(new Action(() => numInp = Int32.Parse(NumberOfInpTextBox.Text)));
                timeSeriGenerator.load(numInp);
                Dispatcher.Invoke(new Action(() => ActivityProgressBar.IsIndeterminate = true));
                Dispatcher.Invoke(new Action(() => numberOfTests = Int32.Parse(OptimumTestTextBox.Text)));
                Dispatcher.Invoke(new Action(() => numberOfForecastTests = Int32.Parse(ForecastTestTextBox.Text)));
                myCategorizedTimeSeri = timeSeriGenerator.generate(numberOfTests, numberOfForecastTests);

                #endregion


                #region creating and training the svm model

                double minError = 9999999;
                SVM_KERNEL_TYPE bestKernelType = SVM_KERNEL_TYPE.LINEAR;
                double bestEps = 0.1;

                SVMParams p;
                Matrix<float> trainData = new Matrix<float>(myCategorizedTimeSeri.TrainInputs);
                Matrix<float> trainClasses = new Matrix<float>(myCategorizedTimeSeri.TrainTargets);
                Matrix<float> testData = new Matrix<float>(myCategorizedTimeSeri.TestInputs);
                Matrix<float> testClasses = new Matrix<float>(myCategorizedTimeSeri.TestTargets);

                foreach (SVM_KERNEL_TYPE tp in Enum.GetValues(typeof (SVM_KERNEL_TYPE)))
                {
                    for (double eps = 0.1; eps >= 0.00001; eps *= 0.1)
                    {
                        using (SVM model = new SVM())
                        {
                            p = new SVMParams();
                            p.KernelType = tp;
                            p.SVMType = SVM_TYPE.EPS_SVR; // for regression
                            p.C = 1;
                            p.TermCrit = new MCvTermCriteria(100, eps);
                            p.Gamma = 1;
                            p.Degree = 1;
                            p.P = 1;
                            p.Nu = 0.1;

                            bool trained = model.TrainAuto(trainData, trainClasses, null, null, p.MCvSVMParams, 10);

                            double error = getSumError(model, testData, testClasses);
                            if (trained && minError > error)
                            {
                                minError = error;

                                bestEps = eps;
                                bestKernelType = tp;
                            }
                        }
                    }
                }

                Matrix<float> trainDataWithGATest = new Matrix<float>(myCategorizedTimeSeri.getTrainWithTestInputs());
                Matrix<float> trainClassesWithGATest = new Matrix<float>(myCategorizedTimeSeri.getTrainWithTestTargets());

                svmModel = new SVM();
                p = new SVMParams();
                p.KernelType = bestKernelType;
                p.SVMType = Emgu.CV.ML.MlEnum.SVM_TYPE.EPS_SVR; // for regression
                p.C = 1;
                p.TermCrit = new MCvTermCriteria(100, bestEps);
                p.Gamma = 1;
                p.Degree = 1;
                p.P = 1;
                p.Nu = 0.1;

                bool _trained = svmModel.TrainAuto(trainDataWithGATest, trainClassesWithGATest, null, null,
                                                   p.MCvSVMParams, 10);

                List<float> Et = getResidual(trainDataWithGATest, trainClassesWithGATest);
                svmWriter.Flush();
                svmWriter.Close();

                int bestD = StartArima(Et.ToArray());

                List<float> Zt = new List<float>();
                float mu = Et.Average();
                if (bestD == 0)
                {
                    for (int i = 0; i < Et.Count; i++)
                    {
                        Zt.Add(Et[i] - mu);
                    }
                }
                else if (bestD == 1)
                {
                    Zt.Add(0);
                    for (int i = 1; i < Et.Count; i++)
                    {
                        Zt.Add(Et[i] - Et[i - 1] - mu);
                    }
                }
                else //else if (bestD == 2)    << CHECK HERE >>
                {
                    Zt.Add(0);
                    Zt.Add(0);
                    for (int i = 2; i < Et.Count; i++)
                    {
                        Zt.Add(Et[i] - 2*Et[i - 1] + Et[i - 2] - mu);
                    }
                }

                Pair<int> bestAB = CreateComplexHybridModel(Et.ToArray(), Zt.ToArray());
                MessageBox.Show(bestAB.First + " , " + bestAB.Second, "INJAAAAAAAAAAAAAAAAA", MessageBoxButton.OK,
                                MessageBoxImage.Asterisk);

                // now our complex hybrid model is created

                double minErr = SVMComplexModelForBestModel(bestAB.First, bestAB.Second, Et.ToArray(), Zt.ToArray());
                MessageBox.Show("MinError In Training =>  "+minErr);

                double mse = 0;
                double errorPercent = 0;
                double sumTargets = 0;

                List<float> results = new List<float>();
                Matrix<float> testIn = new Matrix<float>(myCategorizedTimeSeri.ForecastTestInputs);
                Queue<float> EtQueue = new Queue<float>();
                Queue<float> ZtQueue = new Queue<float>();
                for (int i = 0; i < bestAB.First; i++)
                {
                    EtQueue.Enqueue(Et[Et.Count - bestAB.First + i]);
                }
                for (int i = 0; i < bestAB.Second; i++)
                {
                    ZtQueue.Enqueue(Zt[Zt.Count - bestAB.Second + i]);
                }
                for (int i = 0; i < numberOfForecastTests; i++)
                {
                    float Lt = svmModel.Predict(testIn.GetRow(i));
                    float[] inpTest = new float[bestAB.First + bestAB.Second + 1];
                    float[] EQArray = EtQueue.ToArray();
                    float[] ZQArray = ZtQueue.ToArray();
                    int l = 0;
                    for (int j = 0; j < bestAB.First; j++, l++)
                    {
                        inpTest[l] = EQArray[j];
                    }
                    inpTest[l++] = Lt;
                    for (int j = 0; j < bestAB.Second; j++, l++)
                    {
                        inpTest[l] = ZQArray[j];
                    }
                    float result = svmModelHybrid.Predict(new Matrix<float>(inpTest));
                    results.Add(result);
                    hybridWriter.WriteLine(result);
                    float target = myCategorizedTimeSeri.TestTargets[i];

                    //mse += Math.Pow(target - result, 2);
                    //errorPercent += Math.Abs(target - result);
                    //sumTargets += Math.Abs(target);

                    // preparing for next use in this for loop
                    float resi = target - Lt;    // float resi = target - result;   << CHECK HERE IMPORTANT >>
                    Et.Add(resi);
                    EtQueue.Dequeue();
                    EtQueue.Enqueue(resi);
                    ZtQueue.Dequeue();
                    mu = Et.Average();
                    if (bestD == 0)
                    {
                        ZtQueue.Enqueue(EQArray[EQArray.Length - 1] - mu);
                    }
                    else if (bestD == 1)
                    {
                        ZtQueue.Enqueue(EQArray[EQArray.Length - 1] - EQArray[EQArray.Length - 2] - mu);
                    }
                    else //else if (bestD == 2)    << CHECK HERE >>
                    {
                        ZtQueue.Enqueue(EQArray[EQArray.Length - 1] - 2*EQArray[EQArray.Length - 2] +
                                        EQArray[EQArray.Length - 3] - mu);
                    }
                }
                //mse /= numberOfForecastTests;
                //hybridWriter.WriteLine("\n\nMSE =>  " + mse);
                //errorPercent /= sumTargets;
                //hybridWriter.WriteLine("\n\nERROR% =>  " + errorPercent*100);

                double _mse = MyErrorParameters.MSE(results.ToArray(), myCategorizedTimeSeri.ForecastTestTargets);
                double _errorPercent = MyErrorParameters.ERROR_Percent(results.ToArray(), myCategorizedTimeSeri.ForecastTestTargets);
                hybridWriter.WriteLine("\n\n\nMSE & ERROR% are =>\n\n{0} {1}",_mse,_errorPercent);

                hybridWriter.Flush();
                hybridWriter.Close();

                MessageBox.Show(
                    String.Format(
                        "Complex Hybrid Model Created File {0} For Output Successfully Now , Please Check It Out .",
                        OUTPUT_FILE_NAME), "Hybrid SVM Arima Done", MessageBoxButton.OK,
                    MessageBoxImage.Information);

                #endregion
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message + "\n\n" + ex.StackTrace, "ERROR", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        public Pair<int> CreateComplexHybridModel(float[] Et, float[] Zt)
        {
            try
            {
                int bestA = 0;
                int bestB = 0;

                load("MyGA_1.m");
                int n = (int)matlab.GetVariable("n", "base");
                bool endCondition = true;
                StreamWriter gaLogger = new StreamWriter("Complex_Hybrid_Logger.txt");
                StreamWriter sw = new StreamWriter("Complex_Result_Model_Output.txt");
                do
                {
                    Dispatcher.Invoke(new Action(() => IterationLabel.Content = getMyVariable("m")));

                    string command = "\n y=[";
                    for (int i = 1; i <= n; i++)
                    {
                        float aa = 0, bb = 0;
                        aa = float.Parse(getMyVariable("p(" + i + ",1)"));
                        bb = float.Parse(getMyVariable("p(" + i + ",2)"));
                        int a = (int) Math.Round(aa);
                        int b = (int) Math.Round(bb);
                        double error = SVMComplexModel(a, b, Et, Zt);
                        gaLogger.WriteLine("{0} {1} ->  {2}", a, b, error);
                        gaLogger.Flush();
                        command += error + ",";
                    }
                    gaLogger.WriteLine("\n\n\n");
                    command = command.Remove(command.Length - 1) + "]; \n";
                    string matlabCode = justLoadAndGetContent("MyGA2.m") + command + justLoadAndGetContent("MyGA_3.m");
                    string result = matlab.Execute(matlabCode);

                    bestA = (int)Math.Round(float.Parse(getMyVariable("NowA")));
                    bestB = (int)Math.Round(float.Parse(getMyVariable("NowB")));

                    sw.WriteLine(result);
                    Dispatcher.Invoke(new Action(() => IterationLabel.Content = getMyVariable("m")));
                    matlab.Execute(
                        "EndCondition = m<" + maxGAIteretionInHybrid + " && abs(maxvalue(m)-maxvalue(m-(m0-1)))>0.001*maxvalue(m) && (abs(maxvalue(m))>1e-10 && abs(maxvalue(m-(m0-1)))>1e-10) && m<" + maxGAIteretionInHybrid + " && abs(maxvalue(m)-meanvalue(m))>1e-5 || m<20");
                    endCondition = (bool)matlab.GetVariable("EndCondition", "base");
                } while (endCondition);
                gaLogger.Flush();
                gaLogger.Close();
                sw.Flush();
                sw.Close();
                Dispatcher.Invoke(new Action(() => ActivityProgressBar.IsIndeterminate = false));
                MessageBox.Show("Finaly Our Complex Hybrid Model Created Successfully .", "Done", MessageBoxButton.OK,
                                MessageBoxImage.Information);

                return new Pair<int>(bestA,bestB);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message + "\n" + ex.StackTrace, "ERROR1", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            return null;
        }

        private double SVMComplexModel(int y, int z, float[] Et, float[] Zt)
        {
            double error = -9999999;

            if (Math.Max(y, z) < myCategorizedTimeSeri.TrainInputs.GetLength(1))
            {
                return error;
            }

            int numOfInp = y + z + 1;
            int rows = Et.Length - Math.Max(y, z);
            float[,] inps = new float[rows,numOfInp];
            float[] targs = new float[rows];
            int l = 0;
            for (int i = 0; i < rows; i++)
            {
                if (y > z)
                {
                    for (int j = 0; j < y; j++)
                    {
                        inps[i, j] = Et[l + j];
                    }
                    inps[i, y] = svmModel.Predict(new Matrix<float>(myCategorizedTimeSeri.getInputsForTarget(l + y)));
                    for (int j = 0; j < z; j++)
                    {
                        inps[i, y + j + 1] = Zt[l + y - z + j];
                    }
                    targs[i] = myCategorizedTimeSeri.TimeSeri[l + y];
                }
                else
                {
                    for (int j = 0; j < y; j++)
                    {
                        inps[i, j] = Et[l + z - y + j];
                    }
                    inps[i, y] = svmModel.Predict(new Matrix<float>(myCategorizedTimeSeri.getInputsForTarget(l + z)));
                    for (int j = 0; j < z; j++)
                    {
                        inps[i, j + y + 1] = Zt[l + j];
                    }
                    targs[i] = myCategorizedTimeSeri.TimeSeri[l + z];
                }
                l++;
            }

            float[,] trainInputs = new float[rows - numberOfTests,numOfInp];
            float[] trainTargets = new float[rows - numberOfTests];
            float[,] testInputs = new float[numberOfTests,numOfInp];
            float[] testTargets = new float[numberOfTests];
            int t = 0;
            for (; t < rows - numberOfTests; t++)
            {
                for (int j = 0; j < numOfInp; j++)
                {
                    trainInputs[t, j] = inps[t, j];
                }
                trainTargets[t] = targs[t];
            }
            for (int i = 0; t < rows; i++, t++)
            {
                for (int j = 0; j < numOfInp; j++)
                {
                    testInputs[i, j] = inps[t, j];
                }
                testTargets[i] = targs[t];
            }


            double minError = 9999999;
            SVM_KERNEL_TYPE bestKernelType = SVM_KERNEL_TYPE.LINEAR;
            double bestEps = 0.1;
            SVMParams p;
            Matrix<float> trainData = new Matrix<float>(trainInputs);
            Matrix<float> trainClasses = new Matrix<float>(trainTargets);
            Matrix<float> testData = new Matrix<float>(testInputs);
            Matrix<float> testClasses = new Matrix<float>(testTargets);

            // UNCOMMENT THIS FOR BETER MODEL   << CHECK HERE >>
            //foreach (SVM_KERNEL_TYPE tp in Enum.GetValues(typeof(SVM_KERNEL_TYPE)))
            //{
            //    for (double eps = 0.1; eps >= 0.00001; eps *= 0.1)
            //    {
            //        using (SVM model = new SVM())
            //        {
            //            p = new SVMParams();
            //            p.KernelType = tp;
            //            p.SVMType = SVM_TYPE.EPS_SVR; // for regression
            //            p.C = 1;
            //            p.TermCrit = new MCvTermCriteria(100, eps);
            //            p.Gamma = 1;
            //            p.Degree = 1;
            //            p.P = 1;
            //            p.Nu = 0.1;

            //            bool trained = model.TrainAuto(trainData, trainClasses, null, null, p.MCvSVMParams, 10);

            //            double err = getSumError(model, testData, testClasses);
            //            if (trained && minError > err)
            //            {
            //                minError = err;

            //                bestEps = eps;
            //                bestKernelType = tp;
            //            }
            //        }
            //    }
            //}

            //svmModelHybrid = new SVM();
            //p = new SVMParams();
            //p.KernelType = bestKernelType;
            //p.SVMType = Emgu.CV.ML.MlEnum.SVM_TYPE.EPS_SVR; // for regression
            //p.C = 1;
            //p.TermCrit = new MCvTermCriteria(100, bestEps);
            //p.Gamma = 1;
            //p.Degree = 1;
            //p.P = 1;
            //p.Nu = 0.1;
            // UNCOMMENT HERE


            svmModelHybrid = new SVM();
            p = new SVMParams();
            p.KernelType = Emgu.CV.ML.MlEnum.SVM_KERNEL_TYPE.POLY;
            p.SVMType = Emgu.CV.ML.MlEnum.SVM_TYPE.EPS_SVR; // for regression
            p.C = 1;
            p.TermCrit = new MCvTermCriteria(100, 0.00001);
            p.Gamma = 1;
            p.Degree = 1;
            p.P = 1;
            p.Nu = 0.1;


            bool _trained = svmModelHybrid.TrainAuto(trainData, trainClasses, null, null, p.MCvSVMParams, 10);
            error = -1*getSumError(svmModelHybrid, testData, testClasses);

            return error;
        }

        private double SVMComplexModelForBestModel(int y, int z, float[] Et, float[] Zt)
        {
            double error = -9999999;

            if (Math.Max(y, z) < myCategorizedTimeSeri.TrainInputs.GetLength(1))
            {
                return error;
            }

            int numOfInp = y + z + 1;
            int rows = Et.Length - Math.Max(y, z);
            float[,] inps = new float[rows, numOfInp];
            float[] targs = new float[rows];
            int l = 0;
            for (int i = 0; i < rows; i++)
            {
                if (y > z)
                {
                    for (int j = 0; j < y; j++)
                    {
                        inps[i, j] = Et[l + j];
                    }
                    inps[i, y] = svmModel.Predict(new Matrix<float>(myCategorizedTimeSeri.getInputsForTarget(l + y)));
                    for (int j = 0; j < z; j++)
                    {
                        inps[i, y + j + 1] = Zt[l + y - z + j];
                    }
                    targs[i] = myCategorizedTimeSeri.TimeSeri[l + y];
                }
                else
                {
                    for (int j = 0; j < y; j++)
                    {
                        inps[i, j] = Et[l + z - y + j];
                    }
                    inps[i, y] = svmModel.Predict(new Matrix<float>(myCategorizedTimeSeri.getInputsForTarget(l + z)));
                    for (int j = 0; j < z; j++)
                    {
                        inps[i, j + y + 1] = Zt[l + j];
                    }
                    targs[i] = myCategorizedTimeSeri.TimeSeri[l + z];
                }
                l++;
            }

            float[,] trainInputs = new float[rows - numberOfTests, numOfInp];
            float[] trainTargets = new float[rows - numberOfTests];
            float[,] testInputs = new float[numberOfTests, numOfInp];
            float[] testTargets = new float[numberOfTests];
            int t = 0;
            for (; t < rows - numberOfTests; t++)
            {
                for (int j = 0; j < numOfInp; j++)
                {
                    trainInputs[t, j] = inps[t, j];
                }
                trainTargets[t] = targs[t];
            }
            for (int i = 0; t < rows; i++, t++)
            {
                for (int j = 0; j < numOfInp; j++)
                {
                    testInputs[i, j] = inps[t, j];
                }
                testTargets[i] = targs[t];
            }


            double minError = 9999999;
            SVM_KERNEL_TYPE bestKernelType = SVM_KERNEL_TYPE.LINEAR;
            double bestEps = 0.1;
            SVMParams p;
            Matrix<float> trainData = new Matrix<float>(trainInputs);
            Matrix<float> trainClasses = new Matrix<float>(trainTargets);
            Matrix<float> testData = new Matrix<float>(testInputs);
            Matrix<float> testClasses = new Matrix<float>(testTargets);

            foreach (SVM_KERNEL_TYPE tp in Enum.GetValues(typeof(SVM_KERNEL_TYPE)))
            {
                for (double eps = 0.1; eps >= 0.00001; eps *= 0.1)
                {
                    using (SVM model = new SVM())
                    {
                        p = new SVMParams();
                        p.KernelType = tp;
                        p.SVMType = SVM_TYPE.EPS_SVR; // for regression
                        p.C = 1;
                        p.TermCrit = new MCvTermCriteria(100, eps);
                        p.Gamma = 1;
                        p.Degree = 1;
                        p.P = 1;
                        p.Nu = 0.1;

                        bool trained = model.TrainAuto(trainData, trainClasses, null, null, p.MCvSVMParams, 10);

                        double err = getSumError(model, testData, testClasses);
                        if (trained && minError > err)
                        {
                            minError = err;

                            bestEps = eps;
                            bestKernelType = tp;
                        }
                    }
                }
            }

            svmModelHybrid = new SVM();
            p = new SVMParams();
            p.KernelType = bestKernelType;
            p.SVMType = Emgu.CV.ML.MlEnum.SVM_TYPE.EPS_SVR; // for regression
            p.C = 1;
            p.TermCrit = new MCvTermCriteria(100, bestEps);
            p.Gamma = 1;
            p.Degree = 1;
            p.P = 1;
            p.Nu = 0.1;


            bool _trained = svmModelHybrid.TrainAuto(trainData, trainClasses, null, null, p.MCvSVMParams, 10);
            error = -1 * getSumError(svmModelHybrid, testData, testClasses);

            return error;
        }

        public string getMyVariable(string name)
        {
            string s = matlab.Execute(name);
            s = s.Substring(11);
            return s.Remove(s.Length - 2);
        }

        public void execMatlabCode(string commands)
        {
            Dispatcher.Invoke(new Action(() => ResultListBox.Items.Add(matlab.Execute(commands))));
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

        public int StartArima(float[] data)   // returns best D
        {
            int bestD = 0;
            try
            {
                trainArima = new double[data.Length - numberOfTests];
                testArima = new double[numberOfTests];
                for (int i = 0; i < trainArima.Length; i++)
                {
                    trainArima[i] = data[i];
                }
                for (int i = trainArima.Length, j = 0; i < data.Length; i++, j++)
                {
                    testArima[j] = data[i];
                }

                load("MyGA1.m");
                int n = (int) matlab.GetVariable("n", "base");
                bool endCondition = true;
                StreamWriter sw = new StreamWriter("Complex_Hybrid_Output.txt");
                do
                {
                    Dispatcher.Invoke(new Action(() => IterationLabel.Content = getMyVariable("m")));

                    string command = "\n y=[";
                    for (int i = 1; i <= n; i++)
                    {
                        double pp = double.Parse(getMyVariable("p(" + i + ",1)"));
                        double dd = double.Parse(getMyVariable("p(" + i + ",2)"));
                        double qq = double.Parse(getMyVariable("p(" + i + ",3)"));
                        int p = (int) Math.Round(pp);
                        int d = (int) Math.Round(dd);
                        int q = (int) Math.Round(qq);
                        double error = arimaModelFunction(p, d, q);
                        arimaLogger.WriteLine("{0} {1} {2} ->  {3}", p, d, q, error);
                        command += error + ",";
                    }
                    arimaLogger.WriteLine("\n\n\n");
                    command = command.Remove(command.Length - 1) + "]; \n";
                    string matlabCode = justLoadAndGetContent("MyGA2.m") + command + justLoadAndGetContent("MyGA3.m");
                    string result = matlab.Execute(matlabCode);
                    sw.WriteLine(result);
                    matlab.Execute(
                        "EndCondition = m<" + maxGAIteretionInArima + " && abs(maxvalue(m)-maxvalue(m-(m0-1)))>0.001*maxvalue(m) && (abs(maxvalue(m))>1e-10 && abs(maxvalue(m-(m0-1)))>1e-10) && m<" + maxGAIteretionInArima + " && abs(maxvalue(m)-meanvalue(m))>1e-5 || m<20");
                    endCondition = (bool) matlab.GetVariable("EndCondition", "base");
                } while (endCondition);
                sw.Flush();
                sw.Close();
                bestD = (int)Math.Round(float.Parse(getMyVariable("NowD")));
                //MessageBox.Show("Arima Model Created Successfully", "Best SVM with Grid & Best Arima With GA", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message + "\n" + ex.StackTrace, "ERROR1", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            
            return bestD;
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
            NumericalVariable timeSerii = new NumericalVariable("timeSerii", trainArima);

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
            Vector nextValues = arimaModel.Forecast(numberOfTests);
            int selected = 0;
            Dispatcher.Invoke(new Action(() => selected = CompareComboBox.SelectedIndex));
            switch (selected)
            {
                case 0: // MSE
                    double mse = 0;
                    for (int i = 0; i < numberOfTests; i++)
                    {
                        mse += Math.Pow(nextValues[i] - testArima[i], 2);
                    }
                    mse /= numberOfTests;
                    return -1 * mse;
                case 1: // ERROR %

                    double errorPercent = 0;
                    double sumTargets = 0;
                    for (int i = 0; i < numberOfTests; i++)
                    {
                        errorPercent += Math.Abs(nextValues[i] - testArima[i]);
                        sumTargets += Math.Abs(testArima[i]);
                    }
                    errorPercent /= sumTargets;
                    return -1 * errorPercent*100;
            }
            return -9999999999; // a very bad fitness when error occured
        }

        public List<float> getResidual(Matrix<float> trainData, Matrix<float> trainClasses)
        {
            List<float > res = new List<float>();
            for (int i = 0; i < trainClasses.Rows; i++)
            {
                float predictedValue =  svmModel.Predict(trainData.GetRow(i));
                svmWriter.WriteLine(predictedValue);
                float diff = trainClasses.Data[i, 0] - predictedValue;
                res.Add(diff);
            }
            return res;
        }

        public double getSumError(SVM model, Matrix<float> testInputs, Matrix<float> testOutputs)
        {
            double sum = 0;
            for (int i = 0; i < testOutputs.Rows; i++)
            {
                double diff2 = Math.Pow((testOutputs.Data[i, 0] - model.Predict(testInputs.GetRow(i))), 2);
                sum += diff2;
            }
            return sum / testOutputs.Rows;
        }

        private void StartButton_Click(object sender, RoutedEventArgs e)
        {
            Thread thread=new Thread(new ThreadStart(Start));
            thread.Start();
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            matlab.Quit();
            arimaLogger.Flush();
            arimaLogger.Close();
        }
    }
}