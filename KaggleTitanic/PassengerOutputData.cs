using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace KaggleTitanic
{
    public class PassengerOutputData
    {
        [ColumnName("Score")]
        public float Survived;
    }
}
