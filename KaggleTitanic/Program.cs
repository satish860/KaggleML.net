using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using System.Threading.Tasks;

namespace KaggleTitanic
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var trainFilePath =
                Path.Combine(@"D:\MachineLearningNet\KaggleTitanic\datasets\", "train.csv");
            var testFilePath =
                Path.Combine(@"D:\MachineLearningNet\KaggleTitanic\datasets\", "test.csv");
            var ModelPath =
               Path.Combine(@"D:\MachineLearningNet\KaggleTitanic\datasets\", "PassengerModel.Zip");
            LearningPipeline pipeline = new LearningPipeline
           {
               new LoadData().LoadTrainData(trainFilePath),
               new ColumnCopier(("Survived","Label")),
               new CategoricalOneHotVectorizer("Pclass"
               ,"Name"
               ,"Sex"
               ,"Age"
               ,"SibSp"
               ,"Parch"
               ,"Ticket"
               ,"Cabin"
               ,"Embarked"),
               new ColumnConcatenator("Features",
               "Pclass",
               "Name",
               "Sex",
               "Age",
               "SibSp",
               "Parch",
               "Ticket","Cabin","Embarked"),

               new FastTreeBinaryClassifier()
           };


            Console.WriteLine("=============== Training model ===============");
            // The pipeline is trained on the dataset that has been loaded and transformed.
            var model = pipeline.Train<PassengerData, PassengerOutputData>();


            // Saving the model as a .zip file.
            await model.WriteAsync(ModelPath);

            var streamReader = File.OpenText(testFilePath);
            var csvReader = new CsvReader(streamReader);
            while (csvReader.Read())
            {
                var automap = csvReader.Configuration.AutoMap<PassengerData>();
                csvReader.Configuration.HasHeaderRecord = true;
                var record = csvReader.GetRecord<dynamic>();
                PassengerData passengerData = new PassengerData()
                {
                    Age = record.Age,
                    Cabin = record.Cabin,
                    Embarked = record.Embarked,
                    Name = record.Name,
                    Parch = record.Parch,
                    PassengerId = record.PassengerId,
                    Pclass = record.Pclass,
                    Sex = record.Sex,
                    SibSp = record.SibSp,
                    Ticket = record.Ticket
                };
                var output = model.Predict(passengerData);
                Console.WriteLine($"Output for Passenger id {passengerData.PassengerId} is {output.Survived}");
            }



            Console.WriteLine("=============== End training ===============");
            Console.WriteLine("The model is saved to {0}", ModelPath);
        }
    }
}
