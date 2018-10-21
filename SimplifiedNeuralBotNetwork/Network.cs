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

        public Node[] InputNodes { get; set; }

        public Node[] HiddenNodes { get; set; }

        public Node[] OutputNodes { get; set; }

        public double Fitness { get; set; }

        Random rand;

        public Network(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, Random rand, int id)
        {
            InputLayerSize = inputLayerSize;
            HiddenLayerSize = hiddenLayerSize;
            OutputLayerSize = outputLayerSize;

            this.rand = rand;

            InputNodes = new Node[inputLayerSize];
            for(int i = 0; i < InputNodes.Length; ++i)
            {
                InputNodes[i] = new Node();
            }

            HiddenNodes = new Node[hiddenLayerSize];
            for (int i = 0; i < HiddenNodes.Length; i++)
            {
                HiddenNodes[i] = new Node
                {
                    Weights = new double[inputLayerSize]
                };
            }

            OutputNodes = new Node[outputLayerSize];
            for (int i = 0; i < OutputNodes.Length; i++)
            {
                OutputNodes[i] = new Node
                {
                    Weights = new double[hiddenLayerSize]
                };
            }

            ID = id;

            InitializeWeights(HiddenNodes);
            InitializeWeights(OutputNodes);
        }

        private void InitializeWeights(Node[] nodeArray)
        {
            foreach(Node node in nodeArray)
            {
                for (int i = 0; i < node.Weights.Length; i++)
                {
                    node.Weights[i] = rand.NextDouble();
                }
            }
        }

        public void Propagate()
        {
            PropagateArray(InputNodes, HiddenNodes, true);
            PropagateArray(HiddenNodes, OutputNodes, true);
        }

        private void PropagateArray(Node[] leftLayer, Node[] rightLayer, bool useSigmoid)
        {
            for(int i = 0; i < rightLayer.Length; ++i)
            {
                double currentWeightedValue = 0.0;
                for(int j = 0; j < leftLayer.Length; ++j)
                {
                    currentWeightedValue += leftLayer[j].Value * rightLayer[i].Weights[j];
                }
                if (useSigmoid)
                {
                    rightLayer[i].Value = 1.0 / (1 + currentWeightedValue);
                }
                else
                {
                    rightLayer[i].Value = currentWeightedValue;
                }
            }
        }

        public double CalculateFitness(double[] expectedValues)
        {
            double numberOfCorrectAwnsers = 0.0;
            for(int i = 0; i < expectedValues.Length; ++i)
            {
                numberOfCorrectAwnsers += Math.Round(OutputNodes[i].Value, 0, MidpointRounding.AwayFromZero) == expectedValues[i] ? 1 : 0;
            }
            return numberOfCorrectAwnsers;
        }

        public void SetInputs(double[] inputArray)
        {
            if(inputArray.Length != InputLayerSize)
            {
                throw new ArgumentException("Data-Size and Input-Size do not match");
            }

            for(int i = 0; i < inputArray.Length; ++i)
            {
                InputNodes[i].Value = inputArray[i];
            }
        }

        public double[] GetInputs()
        {
            return GetValues(InputNodes);
        }

        public double[] GetOutputs()
        {
            return GetValues(OutputNodes);
        }

        private double[] GetValues(Node[] layer)
        {
            double[] values = new double[layer.Length];
            for (int i = 0; i < layer.Length; ++i)
            {
                values[i] = layer[i].Value;
            }
            return values;
        }
    }
}
