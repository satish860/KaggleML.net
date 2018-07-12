using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace KaggleTitanic
{
    public class PassengerData
    {

        [Column("0")]
        public string PassengerId;

        [Column("1")]
        public bool Survived;

        [Column("2")]
        public string Pclass;

        [Column("3")]
        public string Name;

        [Column("4")]
        public string Sex;

        [Column("5")]
        public string Age;

        [Column("6")]
        public string SibSp;

        [Column("7")]
        public string Parch;

        [Column("7")]
        public string Ticket;

        [Column("8")]
        public string Cabin;

        [Column("9")]
        public string Embarked;
    }
}
