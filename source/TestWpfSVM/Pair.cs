using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TestWpfSVM
{
    public class Pair<T>
    {
        #region Fields

        private T _first;
        private T _second;

        #endregion


        #region FieldProperty

        public T First
        {
            get { return _first; }
            set { _first = value; }
        }

        public T Second
        {
            get { return _second; }
            set { _second = value; }
        }

        #endregion


        #region Methods

        public Pair()
        {
        }

        public Pair(T a,T b)
        {
            _first = a;
            _second = b;
        }

        #endregion
    }
}
