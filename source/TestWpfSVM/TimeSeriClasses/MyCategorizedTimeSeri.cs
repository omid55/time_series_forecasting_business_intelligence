using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TestWpfSVM.TimeSeriClasses
{
    public class MyCategorizedTimeSeri<T>
    {        
        #region Properties

        public T[] TimeSeri { get; set; }

        public T[,] TrainInputs { get; set; }
        public T[] TrainTargets { get; set; }

        public T[,] TestInputs { get; set; }
        public T[] TestTargets { get; set; }

        public T[,] ForecastTestInputs { get; set; }
        public T[] ForecastTestTargets { get; set; }

        #endregion


        #region Methods

        public MyCategorizedTimeSeri()
        {}

        public T[,] getTrainWithTestInputs()
        {
            T[,] result = new T[TrainInputs.GetLength(0)+TestInputs.GetLength(0),TrainInputs.GetLength(1)];
            int l = 0;
            for (; l < TrainInputs.GetLength(0); l++)
            {
                for (int j = 0; j < TrainInputs.GetLength(1); j++)
                {
                    result[l, j] = TrainInputs[l, j];
                }
            }
            for (int i = 0; i < TestInputs.GetLength(0); i++,l++)
            {
                for (int j = 0; j < TestInputs.GetLength(1); j++)
                {
                    result[l, j] = TestInputs[i, j];
                }
            }
            return result;
        }

        public T[] getTrainWithTestTargets()
        {
            T[] result = new T[TrainTargets.Length+TestTargets.Length];
            int l = 0;
            for (; l < TrainTargets.Length; l++)
            {
                result[l] = TrainTargets[l];
            }
            for (int i = 0; i < TestTargets.Length; i++, l++)
            {
                result[l] = TrainTargets[i];
            }
            return result;
        }

        public T[] getInputsForTarget(int targetIndex)
        {
            T[] inps = new T[TrainInputs.GetLength(1)];
            for (int i = 0; i < inps.Length; i++)
            {
                inps[inps.Length-1-i] = TimeSeri[targetIndex - i - 1];
            }
            return inps;
        }

        #endregion
    }
}
