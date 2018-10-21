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

        public int InputLayerSize { get; set; }
        public int HiddenLayerSize { get; set; }
        public int OutputLayerSize { get; set; }

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
            PropagateArray(InputValues, HiddenValues, HiddenWeights, true);
            PropagateArray(HiddenValues, OutputValues, OutputWeights, true);
        }

        private void PropagateArray(double[] leftArrayValues, double[] rightArrayValues, double[] rightArrayWeights, bool useSigmoid)
        {
            int currentWeight = 0;
            for (int i = 0; i < rightArrayValues.Length; ++i)
            {
                double currentHiddenValue = 0;
                for (int j = 0; j < leftArrayValues.Length; ++j)
                {
                    currentHiddenValue += leftArrayValues[j] * rightArrayWeights[currentWeight];
                    ++currentWeight;
                }
                if (useSigmoid)
                {
                    rightArrayValues[i] = 1.0 / (1 + currentHiddenValue);
                }
                else
                {
                    rightArrayValues[i] = currentHiddenValue;
                }
            }
        }

        public double CalculateFitness(double[] expectedValues)
        {
            double numberOfCorrectAwnsers = 0.0;
            for (int i = 0; i < expectedValues.Length; ++i)
            {
                numberOfCorrectAwnsers += Math.Round(OutputValues[i], 0, MidpointRounding.AwayFromZero) == expectedValues[i] ? 1 : 0;
            }
            return numberOfCorrectAwnsers /*/ expectedValues.Length*/;

            //double totalError = 0.0;
            //for (int i = 0; i < expectedValues.Length; ++i)
            //{
            //    totalError += Math.Pow(expectedValues[i] - OutputValues[i], 2);
            //}
            ////return Fitness = 1.0 / totalError;
            //return Fitness = expectedValues.Length / totalError;
        }
    }
}
