using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TestWpfSVM
{
    public class Dimension
    {
        public int m { get; set; }
        public int n { get; set; }

        public Dimension()
        {}

        public Dimension(int m,int n)
        {
            this.m = m;
            this.n = n;
        }
    }
}
