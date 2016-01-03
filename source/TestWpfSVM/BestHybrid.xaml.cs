using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Emgu.CV.ML;
using Emgu.CV.ML.MlEnum;
using Emgu.CV.Structure;
using Extreme.Statistics;
using Extreme.Statistics.TimeSeriesAnalysis;
using TestWpfSVM.TimeSeriClasses;
using Vector = Extreme.Mathematics.Vector;
using Emgu.CV;

namespace TestWpfSVM
{
    /// <summary>
    /// Interaction logic for BestHybrid.xaml
    /// </summary>
    public partial class BestHybrid : Window
    {
        private MLApp.MLApp matlab;
        public int numberOfForecastTests;
        public int numberOfTests;

        private int maxGAIteretionInArima = 30;      //50;
        public int maxGAIteretionInHybrid = 25;    //30;

        private TimeSeriGenerator<double> timeSeriGenerator;
        private SVM svmModelHybrid = null;

        private double[] trainArima;
        private double[] testArima;

        private StreamWriter arimaLogger;
        private ArimaModel arimaModel = null;

        private int myBestP, myBestQ, myBestD;

        private string OUTPUT_FILE_NAME = "Best_Result_Hybrid_Output.txt";


        public BestHybrid()
        {
            InitializeComponent();

            matlab = new MLApp.MLApp();
            matlab.Visible = 0;
        }

        void Start()
        {
            var Et = new List<double>();
            var Zt = new List<double>();
            var Lt = new List<double>();

            timeSeriGenerator = new TimeSeriGenerator<double>();
            arimaLogger = new StreamWriter("Best_Hybrid_ArimaGALog.txt");
            int numInp = 0;
            this.Dispatcher.Invoke(new Action(() => numInp = Int32.Parse(NumberOfInpTextBox.Text)));
            timeSeriGenerator.load(numInp);
            Dispatcher.Invoke(new Action(() => ActivityProgressBar.IsIndeterminate = true));
            Dispatcher.Invoke(new Action(() => numberOfTests = Int32.Parse(OptimumTestTextBox.Text)));
            Dispatcher.Invoke(new Action(() => numberOfForecastTests = Int32.Parse(ForecastTestTextBox.Text)));

            MyTimeSeriForBestHybrid<double> myTimeSeriForBestHybrid =
                timeSeriGenerator.generateForBestHybrid(numberOfForecastTests);


            //maxGAIteretionInArima = 1000;
            //var train = new double[timeSeriGenerator.TimeSeri.Length - 5];
            //var test = new double[5];
            //for (int i = 0; i < train.Length; i++)
            //{
            //    train[i] = timeSeriGenerator.TimeSeri[i];
            //}
            //for (int i = train.Length, j = 0; i < timeSeriGenerator.TimeSeri.Length; i++, j++)
            //{
            //    test[j] = timeSeriGenerator.TimeSeri[i];
            //}

            //ArimaGA aga=new ArimaGA();
            //aga.StartArima(train);

            //NumericalVariable timeSeriii = new NumericalVariable("timeSeriii", train);
            //arimaModel = new ArimaModel(timeSeriii, aga.bestP, aga.bestD, aga.bestQ);
            //arimaModel.Compute();

            //var fv2 = arimaModel.Forecast(numberOfForecastTests);
            //double ea2 = MyErrorParameters.ERROR_Percent(fv2.ToArray(), test);)



            for (int i = 0; i < myTimeSeriForBestHybrid.part2.Count; i++)
            {
                StartArima(myTimeSeriForBestHybrid.part1.ToArray());

                //// converting to double[]
                //double[] db = new double[myTimeSeriForBestHybrid.part1.Count];
                //for (int j = 0; j < db.Length; j++)
                //{
                //    db[j] = myTimeSeriForBestHybrid.part1.ToArray()[j];
                //}

                NumericalVariable timeSerii = new NumericalVariable("timeSerii", myTimeSeriForBestHybrid.part1.ToArray());
                arimaModel = new ArimaModel(timeSerii, myBestP, myBestD, myBestQ);
                arimaModel.Compute();

                var res = arimaModel.Forecast(1);
                float lt = (float)res[0];
                Lt.Add(lt);
                double target = myTimeSeriForBestHybrid.part2[i];
                double e = lt - target;
                Et.Add(e);

                myTimeSeriForBestHybrid.part1.Add(target);
                double mu = myTimeSeriForBestHybrid.part1.Average();

                if (myBestD == 0)
                {
                    Zt.Add(myTimeSeriForBestHybrid.part1[myTimeSeriForBestHybrid.part1.Count - 1] - mu);
                }
                else if (myBestD == 1)
                {
                    Zt.Add(myTimeSeriForBestHybrid.part1[myTimeSeriForBestHybrid.part1.Count - 1] -
                           myTimeSeriForBestHybrid.part1[myTimeSeriForBestHybrid.part1.Count - 2] - mu);
                }
                else
                {
                    Zt.Add(myTimeSeriForBestHybrid.part1[myTimeSeriForBestHybrid.part1.Count - 1] -
                           2 * myTimeSeriForBestHybrid.part1[myTimeSeriForBestHybrid.part1.Count - 2] +
                           myTimeSeriForBestHybrid.part1[myTimeSeriForBestHybrid.part1.Count - 3] - mu);
                }
            }

            ArimaModel EtArimaModel = new ArimaGA().GetBestModel(Et.ToArray());
            ArimaModel ZtArimaModel = new ArimaGA().GetBestModel(Zt.ToArray());
            int a = 0;
            
            SVM svm = new SVM();
            

            //TimeSeriGenerator<double> gen = new TimeSeriGenerator<double>();
            //gen.NumberOfInputVariables = Int32.Parse(NumberOfInpTextBox.Text);
            //gen.TimeSeri = Et.ToArray();
            //var EtTimeSeries = gen.generate();

            //gen = new TimeSeriGenerator<double>();
            //gen.NumberOfInputVariables = Int32.Parse(NumberOfInpTextBox.Text);
            //gen.TimeSeri = Zt.ToArray();
            //var ZtTimeSeries = gen.generate();
            //// biaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa


            Pair<int> bestAB = CreateComplexHybridModel(Et.ToArray(), Lt.ToArray(), Zt.ToArray());
            double minErr = SVMComplexModelForBestModel(bestAB.First, bestAB.Second, Et.ToArray(), Lt.ToArray(),
                                                        Zt.ToArray());

            MessageBox.Show(bestAB.First + " , " + bestAB.Second + "\nMinError In Training Is =>  " + minErr, "Now Best M & N Found", MessageBoxButton.OK,
                            MessageBoxImage.Asterisk);


            // --------------------------------- now our complex hybrid model is created -----------------------------------------------------------------

            double mse = 0;
            double errorPercent = 0;
            double sumTargets = 0;

            if (myTimeSeriForBestHybrid.part1.Count != timeSeriGenerator.TimeSeri.Length - numberOfForecastTests)
            {
                MessageBox.Show("Input For Arima Model Is Not Completed", "ERROR", MessageBoxButton.OK, MessageBoxImage.Error);
            }


            // << CHECK HERE >>  (FOR CHECKING PURPOSE ONLY , COMMENT HERE LATER)

            var forecastedVector = arimaModel.Forecast(numberOfForecastTests);
            double eoa = MyErrorParameters.ERROR_Percent(forecastedVector.ToArray(), myTimeSeriForBestHybrid.testCases.ToArray());

            MessageBox.Show("Error Of Arima Is =>  " + eoa, "Arima Error", MessageBoxButton.OK,
                MessageBoxImage.Information);

            //maxGAIteretionInArima = 1000;
            //StartArima(myTimeSeriForBestHybrid.part1.ToArray());
            //double[] dbb = new double[myTimeSeriForBestHybrid.part1.Count];
            //for (int j = 0; j < dbb.Length; j++)
            //{
            //    dbb[j] = myTimeSeriForBestHybrid.part1.ToArray()[j];
            //}
            //NumericalVariable timeSeriTest = new NumericalVariable("timeSerii", dbb);
            //arimaModel = new ArimaModel(timeSeriTest, myBestP, myBestD, myBestQ);
            //arimaModel.Compute();

            StreamWriter hybridWriter = new StreamWriter(OUTPUT_FILE_NAME);
            List<double> results = new List<double>();

            //double errorOfArima = MyErrorParameters.ERROR_Percent(forcastedVector.ToArray(), myTimeSeriForBestHybrid.testCases.ToArray());
            //MessageBox.Show("Error Of Arima Is =>  " + errorOfArima, "Arima Error", MessageBoxButton.OK,
            //                MessageBoxImage.Information);


            // ---------------------------------------------------------------
            int numOfInp = bestAB.First + bestAB.Second + 1;
            int rows = Et.Count - Math.Max(bestAB.First, bestAB.Second);
            float[,] inps = new float[rows, numOfInp];
            double[] targs = new double[rows];
            int y = bestAB.First;
            int z = bestAB.Second;
            int ll = 0;
            for (int o = 0; o < rows; o++)
            {
                if (y > z)
                {
                    for (int j = 0; j < y; j++)
                    {
                        inps[o, j] = (float)Et[ll + j];
                    }
                    inps[o, y] = (float)Lt[ll + y];
                    for (int j = 0; j < z; j++)
                    {
                        inps[o, y + j + 1] = (float)Zt[ll + y - z + j];
                    }
                    targs[o] = timeSeriGenerator.TimeSeri[ll + y];
                }
                else
                {
                    for (int j = 0; j < y; j++)
                    {
                        inps[o, j] = (float)Et[ll + z - y + j];
                    }
                    inps[o, y] = (float)Lt[ll + z];
                    for (int j = 0; j < z; j++)
                    {
                        inps[o, j + y + 1] = (float)Zt[ll + j];
                    }
                    targs[o] = timeSeriGenerator.TimeSeri[ll + z];
                }
                ll++;
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
                trainTargets[t] = (float)targs[t];
            }
            for (int o = 0; t < rows; o++, t++)
            {
                for (int j = 0; j < numOfInp; j++)
                {
                    testInputs[o, j] = inps[t, j];
                }
                testTargets[o] = (float)targs[t];
            }

            svmModelHybrid = new SVM();

            SVM_KERNEL_TYPE bestKernelType = SVM_KERNEL_TYPE.RBF;
            double bestEps = 0.001;
            SVMParams p;
            Matrix<float> trainData = new Matrix<float>(trainInputs);
            Matrix<float> trainClasses = new Matrix<float>(trainTargets);
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

            // ---------------------------------------------------------------


            for (int i = 0; i < numberOfForecastTests; i++)
            {
                float[,] inpTest = new float[bestAB.First + bestAB.Second + 1,1];
                int l = 0;
                for (int j = 0; j < bestAB.First; j++, l++)
                {
                    inpTest[l,0] = (float)Et[Et.Count - bestAB.First + j];
                }
                inpTest[l++,0] = (float)forecastedVector[i];
                for (int j = 0; j < bestAB.Second; j++, l++)
                {
                    inpTest[l,0] = (float)Zt[Zt.Count - bestAB.Second + j];
                }


                // injaaaaaaaaaaaaaaaaaaaa



                float result = svmModelHybrid.Predict(new Matrix<float>(inpTest));
                results.Add(result);
                hybridWriter.WriteLine(result);
                double target = myTimeSeriForBestHybrid.testCases[i];


                // preparing for next use in this for loop
                double resi = target - (float)forecastedVector[i];    // float resi = target - result;   << CHECK HERE IMPORTANT >>
                Et.Add(resi);

                myTimeSeriForBestHybrid.part1.Add(target);
                double mu = myTimeSeriForBestHybrid.part1.Average();
                if (myBestD == 0)
                {
                    Zt.Add(myTimeSeriForBestHybrid.part1[myTimeSeriForBestHybrid.part1.Count - 1] - mu);
                }
                else if (myBestD == 1)
                {
                    Zt.Add(myTimeSeriForBestHybrid.part1[myTimeSeriForBestHybrid.part1.Count - 1] -
                           myTimeSeriForBestHybrid.part1[myTimeSeriForBestHybrid.part1.Count - 2] - mu);
                }
                else //else if (bestD == 2)    << CHECK HERE >>
                {
                    Zt.Add(myTimeSeriForBestHybrid.part1[myTimeSeriForBestHybrid.part1.Count - 1] -
                           2 * myTimeSeriForBestHybrid.part1[myTimeSeriForBestHybrid.part1.Count - 2] +
                           myTimeSeriForBestHybrid.part1[myTimeSeriForBestHybrid.part1.Count - 3] - mu);
                }
            }

            double _mse = MyErrorParameters.MSE(results.ToArray(), myTimeSeriForBestHybrid.testCases.ToArray());
            double _errorPercent = MyErrorParameters.ERROR_Percent(results.ToArray(), myTimeSeriForBestHybrid.testCases.ToArray());
            hybridWriter.WriteLine("\n\n\nMSE & ERROR% are =>\n\n{0} {1}", _mse, _errorPercent);

            hybridWriter.Flush();
            hybridWriter.Close();

            MessageBox.Show(
                String.Format(
                    "Complex Hybrid Model Created File {0} For Output Successfully Now , Please Check It Out .",
                    OUTPUT_FILE_NAME), "Hybrid SVM Arima Done", MessageBoxButton.OK,
                MessageBoxImage.Information);
        }

        private double SVMComplexModel(int y, int z, double[] Et, double[] Lt, double[] Zt)
        {
            double error = -9999999;

            int numOfInp = y + z + 1;
            int rows = Et.Length - Math.Max(y, z);
            float[,] inps = new float[rows, numOfInp];
            double[] targs = new double[rows];
            int l = 0;
            for (int i = 0; i < rows; i++)
            {
                if (y > z)
                {
                    for (int j = 0; j < y; j++)
                    {
                        inps[i, j] = (float)Et[l + j];
                    }
                    inps[i, y] = (float)Lt[l + y];
                    for (int j = 0; j < z; j++)
                    {
                        inps[i, y + j + 1] = (float)Zt[l + y - z + j];
                    }
                    targs[i] = timeSeriGenerator.TimeSeri[l + y];
                }
                else
                {
                    for (int j = 0; j < y; j++)
                    {
                        inps[i, j] = (float)Et[l + z - y + j];
                    }
                    inps[i, y] = (float)Lt[l + z];
                    for (int j = 0; j < z; j++)
                    {
                        inps[i, j + y + 1] = (float)Zt[l + j];
                    }
                    targs[i] = timeSeriGenerator.TimeSeri[l + z];
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
                trainTargets[t] = (float)targs[t];
            }
            for (int i = 0; t < rows; i++, t++)
            {
                for (int j = 0; j < numOfInp; j++)
                {
                    testInputs[i, j] = inps[t, j];
                }
                testTargets[i] = (float)targs[t];
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

            //            float[] predictedVa = new float[testData.Rows];
            //            for (int i = 0; i < testData.Rows; i++)
            //            {
            //                predictedVa[i] = model.Predict(testData.GetRow(i));
            //            }
            //            double err = MyErrorParameters.ERROR_Percent(predictedVa, testTargets);

            //            if (trained && minError > err)
            //            {
            //                minError = err;

            //                bestEps = eps;
            //                bestKernelType = tp;
            //            }
            //        }
            //    }
            //}

            //var svmMod = new SVM();
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

            // COMMENT HERE
            var svmMod = new SVM();
            p = new SVMParams();
            p.KernelType = Emgu.CV.ML.MlEnum.SVM_KERNEL_TYPE.POLY;
            p.SVMType = Emgu.CV.ML.MlEnum.SVM_TYPE.EPS_SVR; // for regression
            p.C = 1;
            p.TermCrit = new MCvTermCriteria(100, 0.00001);
            p.Gamma = 1;
            p.Degree = 1;
            p.P = 1;
            p.Nu = 0.1;
            // COMMENT HERE

            bool _trained = svmMod.TrainAuto(trainData, trainClasses, null, null, p.MCvSVMParams, 10);

            float[] predictedV = new float[testData.Rows];
            for (int i = 0; i < testData.Rows; i++)
            {
                predictedV[i] = svmMod.Predict(testData.GetRow(i));
            }
            error = -1 * MyErrorParameters.ERROR_Percent(predictedV, testTargets);

            return error;
        }

        private double SVMComplexModelForBestModel(int y, int z, double[] Et, double[] Lt, double[] Zt)
        {
            double error = -9999999;

            int numOfInp = y + z + 1;
            int rows = Et.Length - Math.Max(y, z);
            float[,] inps = new float[rows, numOfInp];
            double[] targs = new double[rows];
            int l = 0;
            for (int i = 0; i < rows; i++)
            {
                if (y > z)
                {
                    for (int j = 0; j < y; j++)
                    {
                        inps[i, j] = (float)Et[l + j];
                    }
                    inps[i, y] = (float)Lt[l + y];
                    for (int j = 0; j < z; j++)
                    {
                        inps[i, y + j + 1] = (float)Zt[l + y - z + j];
                    }
                    targs[i] = timeSeriGenerator.TimeSeri[l + y];
                }
                else
                {
                    for (int j = 0; j < y; j++)
                    {
                        inps[i, j] = (float)Et[l + z - y + j];
                    }
                    inps[i, y] = (float)Lt[l + z];
                    for (int j = 0; j < z; j++)
                    {
                        inps[i, j + y + 1] = (float)Zt[l + j];
                    }
                    targs[i] = timeSeriGenerator.TimeSeri[l + z];
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
                trainTargets[t] = (float)targs[t];
            }
            for (int i = 0; t < rows; i++, t++)
            {
                for (int j = 0; j < numOfInp; j++)
                {
                    testInputs[i, j] = inps[t, j];
                }
                testTargets[i] = (float)targs[t];
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

                        float[] predictedVa = new float[testData.Rows];
                        for (int i = 0; i < testData.Rows; i++)
                        {
                            predictedVa[i] = model.Predict(testData.GetRow(i));
                        }
                        double err = MyErrorParameters.ERROR_Percent(predictedVa, testTargets);

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

            float[] predictedV = new float[testData.Rows];
            for (int i = 0; i < testData.Rows; i++)
            {
                predictedV[i] = svmModelHybrid.Predict(testData.GetRow(i));
            }
            error = -1 * MyErrorParameters.ERROR_Percent(predictedV, testTargets);

            return error;
        }

        public Pair<int> CreateComplexHybridModel(double[] Et, double[] Lt, double[] Zt)
        {
            try
            {
                int bestA = 0;
                int bestB = 0;

                load("MyGA_1.m");
                int n = (int)matlab.GetVariable("n", "base");
                bool endCondition = true;
                StreamWriter gaLogger = new StreamWriter("Best_Hybrid_Logger.txt");
                StreamWriter sw = new StreamWriter("Best_Result_Hybrid_Model_Output.txt");
                do
                {
                    Dispatcher.Invoke(new Action(() => IterationLabel.Content = getMyVariable("m")));

                    string command = "\n y=[";
                    for (int i = 1; i <= n; i++)
                    {
                        float aa = 0, bb = 0;
                        aa = float.Parse(getMyVariable("p(" + i + ",1)"));
                        bb = float.Parse(getMyVariable("p(" + i + ",2)"));
                        int a = (int)Math.Round(aa);
                        int b = (int)Math.Round(bb);
                        //if(a==3 && b==2)
                        //{
                        //    int c = 0;
                        //}
                        double error = SVMComplexModel(a, b, Et, Lt, Zt);
                        gaLogger.WriteLine("{0} {1} ->  {2}", a, b, error);
                        gaLogger.Flush();
                        command += error + ",";
                    }
                    gaLogger.WriteLine("\n\n\n");
                    gaLogger.Flush();
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

                return new Pair<int>(bestA, bestB);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message + "\n" + ex.StackTrace, "ERROR1", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            return null;
        }

        private void StartButton_Click(object sender, RoutedEventArgs e)
        {
            Thread thread = new Thread(new ThreadStart(Start));
            //thread.ApartmentState = ApartmentState.STA;
            thread.Start();
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            matlab.Quit();
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

        public double arimaModelFunction(int p, int d, int q, int errorMode)
        {
            NumericalVariable timeSerii = new NumericalVariable("timeSerii", trainArima);
            // ARMA models (no differencing) are constructed from
            // the variable containing the time series data, and the
            // AR and MA orders.
            arimaModel = new ArimaModel(timeSerii, p, d, q);

            // The Compute methods fits the model.
            arimaModel.Compute();

            return getErrorOfArimaModel(errorMode);
        }

        private double getErrorOfArimaModel(int selected)
        {
            // or to predict a specified number of values:
            Vector nextValues = arimaModel.Forecast(numberOfTests);

            switch (selected)
            {
                case 0: // MSE
                    return -1 * MyErrorParameters.MSE(nextValues.ToArray(), testArima);

                case 1: // ERROR %
                    return -1 * MyErrorParameters.ERROR_Percent(nextValues.ToArray(), testArima);
            }
            return -9999999999; // a very bad fitness when error occured
        }

        public void StartArima(double[] data)   // returns best D
        {
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

                int selected = 0;
                Dispatcher.Invoke(new Action(() => selected = CompareComboBox.SelectedIndex));
                load("MyGA1.m");
                int n = (int)matlab.GetVariable("n", "base");
                bool endCondition = true;
                StreamWriter sw = new StreamWriter("Best_Hybrid_Arima_Output.txt");
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

                        double error = arimaModelFunction(p, d, q,selected);
                        arimaLogger.WriteLine("{0} {1} {2} ->  {3}", p, d, q, error);
                        command += error + ",";
                    }
                    arimaLogger.WriteLine("\n\n\n");
                    command = command.Remove(command.Length - 1) + "]; \n";
                    string matlabCode = justLoadAndGetContent("MyGA2.m") + command + justLoadAndGetContent("MyGA3.m");
                    string result = matlab.Execute(matlabCode);
                    sw.WriteLine(result);
                    matlab.Execute(
                        "EndCondition = m<" + maxGAIteretionInArima + " && abs(maxvalue(m)-maxvalue(m-(m0-1)))>0.001*maxvalue(m) && (abs(maxvalue(m))>1e-10 && abs(maxvalue(m-(m0-1)))>1e-10) && m<10000 && abs(maxvalue(m)-meanvalue(m))>1e-5 || m<20");
                    endCondition = (bool)matlab.GetVariable("EndCondition", "base");
                } while (endCondition);

                myBestP = (int)Math.Round(float.Parse(getMyVariable("NowP")));
                myBestD = (int)Math.Round(float.Parse(getMyVariable("NowD")));
                myBestQ = (int)Math.Round(float.Parse(getMyVariable("NowQ")));
                double mse = -1 * arimaModelFunction(myBestP, myBestD, myBestQ, 0);
                double errorPercent = -1 * arimaModelFunction(myBestP, myBestD, myBestQ, 1);
                sw.WriteLine("\n\n\n Best P & D & Q are => with this MSE & Error%\n\n {0} {1} {2} => {3} {4}", myBestP,
                             myBestD, myBestQ, mse, errorPercent);
                sw.Flush();
                sw.Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message + "\n" + ex.StackTrace, "ERROR", MessageBoxButton.OK, MessageBoxImage.Error);
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
    }
}
