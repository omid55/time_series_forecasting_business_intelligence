using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Win32;
using TestWpfSVM.TimeSeriClasses;

namespace TestWpfSVM
{
    public class TimeSeriGenerator<T>
    {
        #region Properties

        public T[] TimeSeri { get; set; }
        public int NumberOfInputVariables { get; set; }

        #endregion


        #region Methods

        public TimeSeriGenerator()
        {
        }

        public bool load(int numberOfInputVariables)
        {
            try
            {
                this.NumberOfInputVariables = numberOfInputVariables;
                OpenFileDialog ofd = new OpenFileDialog();
                ofd.Title = "Please Open Your Time Seri Array File :";
                ofd.InitialDirectory = Environment.CurrentDirectory;
                ofd.Filter = "Text Files (*.txt)|*.txt|All Files (*.*)|*.*";
                List<T> loaded = new List<T>();
                bool? res = ofd.ShowDialog();
                if (res.Value)
                {
                    StreamReader sr = new StreamReader(ofd.FileName);
                    string line = sr.ReadLine();
                    while (!sr.EndOfStream && String.IsNullOrEmpty(line) || !('0' <= line[0] && line[0] <= '9'))
                    {
                        line = sr.ReadLine();
                    }
                    string str = line.Split('\t')[0];
                    var parseMethod = typeof (T).GetMethod("Parse", new Type[] {typeof (string)});
                    loaded.Add((T)parseMethod.Invoke(null, new object[] {str}));
                    while (!sr.EndOfStream)
                    {
                        try
                        {
                            loaded.Add((T) parseMethod.Invoke(null, new object[] {sr.ReadLine().Split('\t')[0]}));
                        }
                        catch{}
                    }
                    TimeSeri = loaded.ToArray();
                    return true;
                }
                return false;
            }
            catch(Exception ex)
            {
                return false;
            }
        }

        // please before call this fuction set 2 properties and then call generate method
        public MyTimeSeri<T> generate()
        {
            int rows = TimeSeri.Length - NumberOfInputVariables;
            T[,] input = new T[rows, NumberOfInputVariables];
            T[] target=new T[rows];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < NumberOfInputVariables; j++)
                {
                    input[i, j] = TimeSeri[i + j];
                }
                target[i] = TimeSeri[i + NumberOfInputVariables];
            }

            return new MyTimeSeri<T>(input, target);
        }

        // please before call this fuction set 2 properties and then call generate method
        public MyCategorizedTimeSeri<T> generate(int numberOfTests, int numberOfForecastTests)
        {
            MyTimeSeri<T> tmp = generate();
            MyCategorizedTimeSeri<T> myCategorizedTimeSeri = new MyCategorizedTimeSeri<T>();
            int rows = tmp.inputs.GetLength(0) - numberOfTests - numberOfForecastTests;
            myCategorizedTimeSeri.TimeSeri = TimeSeri;
            myCategorizedTimeSeri.TrainInputs = new T[rows,NumberOfInputVariables];
            myCategorizedTimeSeri.TrainTargets = new T[rows];
            myCategorizedTimeSeri.TestInputs = new T[numberOfTests,NumberOfInputVariables];
            myCategorizedTimeSeri.TestTargets = new T[numberOfTests];
            myCategorizedTimeSeri.ForecastTestInputs = new T[numberOfForecastTests,NumberOfInputVariables];
            myCategorizedTimeSeri.ForecastTestTargets = new T[numberOfForecastTests];
            int l = 0;
            for (; l < rows; l++)
            {
                for (int j = 0; j < NumberOfInputVariables; j++)
                {
                    myCategorizedTimeSeri.TrainInputs[l, j] = tmp.inputs[l, j];
                }
                myCategorizedTimeSeri.TrainTargets[l] = tmp.targets[l];
            }
            for (int i = 0; i < numberOfTests; i++, l++)
            {
                for (int j = 0; j < NumberOfInputVariables; j++)
                {
                    myCategorizedTimeSeri.TestInputs[i, j] = tmp.inputs[l, j];
                }
                myCategorizedTimeSeri.TestTargets[i] = tmp.targets[l];
            }
            for (int i = 0; i < numberOfForecastTests; i++, l++)
            {
                for (int j = 0; j < NumberOfInputVariables; j++)
                {
                    myCategorizedTimeSeri.ForecastTestInputs[i, j] = tmp.inputs[l, j];
                }
                myCategorizedTimeSeri.ForecastTestTargets[i] = tmp.targets[l];
            }
            return myCategorizedTimeSeri;
        }

        // please before call this fuction set 2 properties and then call generate method
        public MyTimeSeriForBestHybrid<T> generateForBestHybrid(int numberOfForecastTests)
        {
            MyTimeSeriForBestHybrid<T> tmp = new MyTimeSeriForBestHybrid<T>();
            int leng = TimeSeri.Length - numberOfForecastTests;
            int l = 0;
            for (; l < leng / 2; l++)
            {
                tmp.part1.Add(TimeSeri[l]);
            }
            for (; l < leng; l++)
            {
                tmp.part2.Add(TimeSeri[l]);
            }
            for (; l < TimeSeri.Length; l++)
            {
                tmp.testCases.Add(TimeSeri[l]);
            }
            return tmp;
        }

        public MyTimeSeri<T> generateWithThisData(T[] data,int numberOfVar)
        {
            TimeSeri = data;
            NumberOfInputVariables = numberOfVar;
            return generate();
        }

        public Queue<T> getLastInputForTesting()
        {
            Queue<T> q = new Queue<T>(NumberOfInputVariables);
            for (int i = TimeSeri.Length-NumberOfInputVariables; i < TimeSeri.Length; i++)
            {
                q.Enqueue(TimeSeri[i]);
            }
            return q;
        }

        #endregion
    }
}
