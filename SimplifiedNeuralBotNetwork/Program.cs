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

        private const int TOTAL_GENERATIONS_TO_CALCULATE = 1000;

        private const int TOTAL_NUMBER_OF_NETWORKS_TO_DISPLAY_EACH_GENERATION = 10;

        private static readonly Random rand = new Random();

        private static List<double[]> inputList = new List<double[]>();
        private static List<double[]> expectedList = new List<double[]>();

        private static List<Network> networkList = new List<Network>();

        private static void Main()
        {
            NetworkCreator networkCreator = new NetworkCreator(INPUT_LAYERSIZE, HIDDEN_LAYERSIZE, OUTPUT_LAYERSIZE, NETWORK_AMOUNT, rand);
            networkList = networkCreator.NetworkList;

            InitializeLists();

            GeneticAlgorithmHandler algorithmHandler = new GeneticAlgorithmHandler(rand, networkList)
            {
                InputList = inputList,
                ExpectedList = expectedList,
                MutationRate = MUTATION_RATE,
                NumberOfNetworksToKeep = NUMBER_OF_NETWORKS_TO_KEEP,
            };


            DisplayResults(algorithmHandler.CurrentDataSet, algorithmHandler.CurrentGeneration);
            for (int i = 0; i < TOTAL_GENERATIONS_TO_CALCULATE; ++i)
            {
                algorithmHandler.IterateNetworks(1);
                DisplayResults(algorithmHandler.CurrentDataSet, algorithmHandler.CurrentGeneration);
            }
        }

        private static void InitializeLists()
        {
            TrainingSetGenerator.GenerateModulusDataSet(NUMBER_OF_TRAINING_NUMBERS);
            inputList = TrainingSetGenerator.GetInputList();
            expectedList = TrainingSetGenerator.GetExpectedList();
        }

        private static void DisplayResults(int currentDataSet, int currentGeneration)
        {
            Console.WriteLine("CurrentGeneration: " + currentGeneration + " | Input: " + inputList[currentDataSet][0]);
            for (int i = 0; i < TOTAL_NUMBER_OF_NETWORKS_TO_DISPLAY_EACH_GENERATION && i < networkList.Count; i++)
            {
                DisplayResult(networkList[i], currentDataSet);
            }
        }

        private static void DisplayResult(Network network, int currentDataSet)
        {
            for (int i = 0; i < network.OutputValues.Length; ++i)
            {
                Console.WriteLine("ID: {0} Result: {1} | Expected: {2} | Fitness: {3}", network.ID, network.OutputValues[i], expectedList[currentDataSet][0], network.Fitness);
            }
            Console.WriteLine();
        }
    }
}
