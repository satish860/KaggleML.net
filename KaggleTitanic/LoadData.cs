using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace KaggleTitanic
{
    public class LoadData
    {
        public TextLoader LoadTrainData(string filePath)
        {
            return new TextLoader(filePath)
                 .CreateFrom<PassengerData>(useHeader:false,separator: ',');
        }
    }
}
