using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TestWpfSVM.TimeSeriClasses
{
    public class MyTimeSeriForBestHybrid<T>
    {
        public List<T> part1 { get; set; }
        public List<T> part2 { get; set; }
        public List<T> testCases { get; set; }

        public MyTimeSeriForBestHybrid()
        {
            part1=new List<T>();
            part2=new List<T>();
            testCases = new List<T>();
        }
    }
}
