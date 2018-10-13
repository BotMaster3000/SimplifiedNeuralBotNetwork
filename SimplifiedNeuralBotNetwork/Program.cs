using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplifiedNeuralBotNetwork
{
    class Program
    {
        const int inputLayerSize = 1;
        const int hiddenLayerSize = 2;
        const int outputLayerSize = 1;

        const int numberOfTrainingNumbers = 100000;
        const int networkAmount = 1000;
        const int numberOfNetworksToKeep = 100;
        const double mutationRate = 0.001;

        const int totalNumberNetworksToDisplayEachGeneration = 10;


        static Random rand = new Random();

        static List<double[]> inputList = new List<double[]>();
        static List<double[]> expectedList = new List<double[]>();

        static int CurrentDataSet = 0;

        static List<Network> networkList = new List<Network>();

        static int currentGeneration = 0;

        static int idCounter = 0;

        static void Main(string[] args)
        {
            for (int i = 0; i < networkAmount; ++i)
            {
                Network network = new Network(inputLayerSize, hiddenLayerSize, outputLayerSize, rand, idCounter);
                networkList.Add(network);
                ++idCounter;
            }

            InitializeLists();

            for (; currentGeneration < 10000; ++currentGeneration)
            {

                CurrentDataSet = rand.Next(0, inputList.Count);

                foreach (Network network in networkList)
                {
                    network.InputValues = inputList[CurrentDataSet];
                    network.Propagate();

                    network.CalculateFitness(expectedList[CurrentDataSet]);
                }

                for (int i = 0; i < networkList.Count; ++i)
                {
                    for (int j = i + 1; j < networkList.Count; ++j)
                    {
                        if (networkList[i].Fitness < networkList[j].Fitness)
                        {
                            Network tempNetwork = networkList[i];
                            networkList[i] = networkList[j];
                            networkList[j] = tempNetwork;
                        }
                    }
                }

                Console.WriteLine("CurrentGeneration: " + currentGeneration + " | Input: " + inputList[CurrentDataSet][0]);
                for (int i = 0; i < totalNumberNetworksToDisplayEachGeneration && i < networkList.Count; i++)
                {
                    DisplayResult(networkList[i]);
                }

                List<int> networkIdPool = new List<int>();
                for (int i = 0; i < numberOfNetworksToKeep; ++i)
                {
                    for (int j = i; j < numberOfNetworksToKeep; ++j)
                    {
                        networkIdPool.Add(i);
                    }
                }
                for (int i = numberOfNetworksToKeep; i < networkList.Count; ++i)
                {
                    int networkIndexToClone = networkIdPool[rand.Next(0, networkIdPool.Count)];
                    double[] currentHiddenWeights = (double[])networkList[networkIndexToClone].HiddenWeights.Clone();
                    double[] currentOutputWeights = (double[])networkList[networkIndexToClone].OutputWeights.Clone();

                    for (int j = 0; j < currentHiddenWeights.Length; j++)
                    {
                        double currentMutationFactor = rand.NextDouble() * mutationRate;
                        if (rand.Next(0, 2) == 0)
                        {
                            currentMutationFactor = -currentMutationFactor;
                        }

                        currentHiddenWeights[j] += currentMutationFactor;
                    }

                    for (int j = 0; j < currentOutputWeights.Length; j++)
                    {
                        double currentMutationFactor = rand.NextDouble() * mutationRate;
                        if (rand.Next(0, 2) == 0)
                        {
                            currentMutationFactor = -currentMutationFactor;
                        }

                        currentOutputWeights[j] += currentMutationFactor;
                    }

                    networkList[i].HiddenWeights = currentHiddenWeights;
                    networkList[i].OutputWeights = currentOutputWeights;
                    networkList[i].ID = idCounter; // New ID since its basically a new network
                    ++idCounter;
                }
            }
        }

        private static void InitializeLists()
        {
            for (int i = 1; i <= numberOfTrainingNumbers; ++i)
            {
                inputList.Add(new double[] { i });
                expectedList.Add(new double[] { i % 2 });
            }
        }


        public static void DisplayResult(Network network)
        {
            for (int i = 0; i < network.OutputValues.Length; ++i)
            {
                Console.WriteLine("ID: {0} Result: {1} | Expected: {2} | Fitness: {3}", network.ID, network.OutputValues[i], expectedList[CurrentDataSet][0], network.Fitness);
            }
            Console.WriteLine();
        }

    }
}
