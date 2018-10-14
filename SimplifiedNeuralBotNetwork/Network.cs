using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplifiedNeuralBotNetwork
{
    public class Network
    {
        public int ID { get; set; }

        public double InputLayerSize { get; set; }
        public double HiddenLayerSize { get; set; }
        public double OutputLayerSize { get; set; }

        public double[] InputValues { get; set; }

        public double[] HiddenValues { get; set; }
        public double[] HiddenWeights { get; set; }

        public double[] OutputValues { get; set; }
        public double[] OutputWeights { get; set; }

        public double Fitness { get; set; }

        Random rand;

        public Network(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, Random rand, int id)
        {
            InputLayerSize = inputLayerSize;
            HiddenLayerSize = hiddenLayerSize;
            OutputLayerSize = outputLayerSize;

            this.rand = rand;

            InputValues = new double[inputLayerSize];

            HiddenValues = new double[hiddenLayerSize];
            HiddenWeights = new double[hiddenLayerSize * inputLayerSize];

            OutputValues = new double[outputLayerSize];
            OutputWeights = new double[outputLayerSize * hiddenLayerSize];

            ID = id;

            InitializeWeights(HiddenWeights);
            InitializeWeights(OutputWeights);
        }

        private void InitializeWeights(double[] weightArray)
        {
            for (int i = 0; i < weightArray.Length; ++i)
            {
                weightArray[i] = rand.NextDouble();
            }
        }

        public void Propagate()
        {
            for (int i = 0; i < HiddenValues.Length; ++i)
            {
                double currentHiddenValue = 0;
                for (int j = 0; j < InputValues.Length; ++j)
                {
                    currentHiddenValue += InputValues[j] * HiddenWeights[i];
                }
                HiddenValues[i] = currentHiddenValue;
            }

            for (int i = 0; i < OutputValues.Length; ++i)
            {
                double currentOutputValue = 0;
                for (int j = 0; j < HiddenValues.Length; ++j)
                {
                    currentOutputValue += HiddenValues[j] * OutputWeights[i];
                }
                OutputValues[i] = currentOutputValue;
            }
        }

        public void CalculateFitness(double[] expectedValues)
        {
            double totalError = 0.0;
            for (int i = 0; i < expectedValues.Length; ++i)
            {
                totalError += Math.Pow(expectedValues[i] - OutputValues[i], 2);
            }

            Fitness = 1 / totalError;
        }
    }
}
