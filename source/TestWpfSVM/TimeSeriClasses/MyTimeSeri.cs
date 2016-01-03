using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Windows;
using Microsoft.Win32;

namespace TestWpfSVM
{
    public class MyTimeSeri<T>
    {
        #region Properties

        public T[,] inputs { get; set; }
        public T[] targets { get; set; }

        #endregion


        #region Methods

        public MyTimeSeri()
        {
        }

        public MyTimeSeri(T[,] inp, T[] targ)
        {
            inputs = inp;
            targets = targ;
        }

        public void write2File(string filePath, bool withHeader = true)
        {
            StreamWriter sw = new StreamWriter(filePath);
            int rows = inputs.GetLength(0);
            int cols = inputs.GetLength(1) + 1;    // because we want to add targets to that

            for (int i = 0; i < cols - 1; i++)
            {
                sw.Write("input{0}\t", (i + 1));
            }
            sw.WriteLine("output");

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols - 1; j++)
                {
                    sw.Write(inputs[i, j] + "\t");
                }
                sw.WriteLine(targets[i]);
            }

            sw.Flush();
            sw.Close();
        }

        public void write2FileWithBrowse()
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.InitialDirectory = Environment.CurrentDirectory;
            sfd.Title = "Please Save Your Time Seri Matrix File :";
            sfd.Filter = "Text Files (*.txt)|*.txt|All Files (*.*)|*.*";
            if (sfd.ShowDialog().Value)
            {
                string saveFilePath = sfd.FileName;
                this.write2File(saveFilePath);

                MessageBox.Show("Your Time Seri Matrix Was Saved Successfully In :\n\n" + sfd.FileName,
                                "Save Time Seri", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }

        #endregion
    }
}
