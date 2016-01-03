using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TestWpfSVM
{
    public class MyErrorParameters
    {
        public static double MSE(double[] values, double[] targets)
        {
            double mse = 0;
            for (int i = 0; i < values.Length; i++)
            {
                mse += Math.Pow(targets[i] - values[i], 2);
            }
            mse /= values.Length;
            return mse;
        }

        public static double MSE(float[] values, float[] targets)
        {
            double mse = 0;
            for (int i = 0; i < values.Length; i++)
            {
                mse += Math.Pow(targets[i] - values[i], 2);
            }
            mse /= values.Length;
            return mse;
        }

        public static double ERROR_Percent(double[] values, double[] targets)
        {
            double errorPercent = 0;
            double sumTargets = 0;
            for (int i = 0; i < values.Length; i++)
            {
                errorPercent += Math.Abs(targets[i] - values[i]);
                sumTargets += Math.Abs(targets[i]);
            }
            errorPercent /= sumTargets;
            return errorPercent*100;
        }

        public static double ERROR_Percent(float[] values, float[] targets)
        {
            double errorPercent = 0;
            double sumTargets = 0;
            for (int i = 0; i < values.Length; i++)
            {
                errorPercent += Math.Abs(targets[i] - values[i]);
                sumTargets += Math.Abs(targets[i]);
            }
            errorPercent /= sumTargets;
            return errorPercent * 100;
        }
    }
}
