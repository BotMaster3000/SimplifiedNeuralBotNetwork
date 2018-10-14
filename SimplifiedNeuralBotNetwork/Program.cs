using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplifiedNeuralBotNetwork
{
    internal class Program
    {
        private const int INPUT_LAYERSIZE = 1;
        private const int HIDDEN_LAYERSIZE = 2;
        private const int OUTPUT_LAYERSIZE = 1;

        private const int NUMBER_OF_TRAINING_NUMBERS = 100000;
        private const int NETWORK_AMOUNT = 1000;
        private const int NUMBER_OF_NETWORKS_TO_KEEP = 100;
        private const double MUTATION_RATE = 0.001;

        private const int TOTAL_GENERATIONS_TO_CALCULATE = 10000;
        private static int currentGeneration;

        private const int TOTAL_NUMBER_OF_NETWORKS_TO_DISPLAY_EACH_GENERATION = 10;

        private static readonly Random rand = new Random();

        private static List<double[]> inputList = new List<double[]>();
        private static List<double[]> expectedList = new List<double[]>();

        private static int CurrentDataSet;

        private static List<Network> networkList = new List<Network>();

        private static int idCounter;

        private static void Main()
        {
            NetworkCreator networkCreator = new NetworkCreator(INPUT_LAYERSIZE, HIDDEN_LAYERSIZE, OUTPUT_LAYERSIZE, NETWORK_AMOUNT, rand);
            networkList = networkCreator.NetworkList;

            InitializeLists();
            IterateNetworks();
        }

        private static void InitializeLists()
        {
            TrainingSetGenerator.GenerateModulusDataSet(NUMBER_OF_TRAINING_NUMBERS);
            inputList = TrainingSetGenerator.GetInputList();
            expectedList = TrainingSetGenerator.GetExpectedList();
        }

        private static void IterateNetworks()
        {
            for (; currentGeneration < TOTAL_GENERATIONS_TO_CALCULATE; ++currentGeneration)
            {
                CurrentDataSet = rand.Next(0, inputList.Count);
                CycleNetworks();
                SortNetworks();
                DisplayResults();
                RebreedNetworks();
            }
        }

        private static void CycleNetworks()
        {
            foreach (Network network in networkList)
            {
                network.InputValues = inputList[CurrentDataSet];
                network.Propagate();

                network.CalculateFitness(expectedList[CurrentDataSet]);
            }
        }

        private static void SortNetworks()
        {
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
        }

        private static void DisplayResults()
        {
            Console.WriteLine("CurrentGeneration: " + currentGeneration + " | Input: " + inputList[CurrentDataSet][0]);
            for (int i = 0; i < TOTAL_NUMBER_OF_NETWORKS_TO_DISPLAY_EACH_GENERATION && i < networkList.Count; i++)
            {
                DisplayResult(networkList[i]);
            }
        }

        private static void DisplayResult(Network network)
        {
            for (int i = 0; i < network.OutputValues.Length; ++i)
            {
                Console.WriteLine("ID: {0} Result: {1} | Expected: {2} | Fitness: {3}", network.ID, network.OutputValues[i], expectedList[CurrentDataSet][0], network.Fitness);
            }
            Console.WriteLine();
        }

        private static void RebreedNetworks()
        {
            List<int> networkIdPool = new List<int>();
            for (int i = 0; i < NUMBER_OF_NETWORKS_TO_KEEP; ++i)
            {
                for (int j = i; j < NUMBER_OF_NETWORKS_TO_KEEP; ++j)
                {
                    networkIdPool.Add(i);
                }
            }
            for (int i = NUMBER_OF_NETWORKS_TO_KEEP; i < networkList.Count; ++i)
            {
                int networkIndexToClone = networkIdPool[rand.Next(0, networkIdPool.Count)];
                double[] currentHiddenWeights = (double[])networkList[networkIndexToClone].HiddenWeights.Clone();
                double[] currentOutputWeights = (double[])networkList[networkIndexToClone].OutputWeights.Clone();

                for (int j = 0; j < currentHiddenWeights.Length; j++)
                {
                    double currentMutationFactor = rand.NextDouble() * MUTATION_RATE;
                    if (rand.Next(0, 2) == 0)
                    {
                        currentMutationFactor = -currentMutationFactor;
                    }

                    currentHiddenWeights[j] += currentMutationFactor;
                }

                for (int j = 0; j < currentOutputWeights.Length; j++)
                {
                    double currentMutationFactor = rand.NextDouble() * MUTATION_RATE;
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
}
