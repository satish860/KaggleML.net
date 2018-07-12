using System;
using System.IO;
using Xunit;
using System.Linq;
using FluentAssertions;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML;

namespace KaggleTitanic.Tests
{
    public class LoadTrainDataTests
    {
        [Fact]
        public void ShouldBeAbleToLoadTheTestData()
        {
            LoadData data = new LoadData();
            var filePath =
                Path.Combine(@"D:\MachineLearningNet\KaggleTitanic\datasets\", "train.csv");
            var textloader = data.LoadTrainData(filePath);
            using (var environment = new TlcEnvironment())
            {
                Experiment experiment = environment.CreateExperiment();
                ILearningPipelineDataStep output = textloader.ApplyStep(null, experiment)
                    as ILearningPipelineDataStep;
                experiment.Compile();
                textloader.SetInput(environment, experiment);
                experiment.Run();

                
                output.Data.Should().NotBeNull();
            }
        }
    }
}
