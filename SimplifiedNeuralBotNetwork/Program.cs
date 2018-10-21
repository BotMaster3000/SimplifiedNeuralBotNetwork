using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplifiedNeuralBotNetwork
{
    internal class Program
    {
        private static int inputLayerSize = 1;
        private const int HIDDEN_LAYERSIZE = 6;
        private static int outputLayerSize = 1;

        private const int NUMBER_OF_TRAINING_NUMBERS = 1000;
        private const int NETWORK_AMOUNT = 1000;
        private const int NUMBER_OF_NETWORKS_TO_KEEP = 10;
        private const double MUTATION_CHANCE = 0.01;
        private const double MUTATION_RATE = 0.5;
        private const int NUMBER_OF_DATASETS_PER_CYCLE = 255;

        private const int TOTAL_GENERATIONS_TO_CALCULATE = 500;

        private const int TOTAL_NUMBER_OF_NETWORKS_TO_DISPLAY_EACH_GENERATION = 10;

        private static readonly Random rand = new Random();

        private static List<double[]> inputList = new List<double[]>();
        private static List<double[]> expectedList = new List<double[]>();

        private static List<Network> networkList = new List<Network>();

        private static void Main()
        {
            InitializeLists();

            NetworkCreator networkCreator = new NetworkCreator(inputLayerSize, HIDDEN_LAYERSIZE, outputLayerSize, NETWORK_AMOUNT, rand);
            networkList = networkCreator.NetworkList;

            GeneticAlgorithmHandler algorithmHandler = new GeneticAlgorithmHandler(rand, networkList)
            {
                InputList = inputList,
                ExpectedList = expectedList,
                MutationChance = MUTATION_CHANCE,
                MutationRate = MUTATION_RATE,
                NumberOfNetworksToKeep = NUMBER_OF_NETWORKS_TO_KEEP,
                NumberOfDataSetsPerCycle = NUMBER_OF_DATASETS_PER_CYCLE,
            };

            NetworkOutputDisplayer.TotalNumberOfNetworksToDisplayEachGeneration = TOTAL_NUMBER_OF_NETWORKS_TO_DISPLAY_EACH_GENERATION;
            NetworkOutputDisplayer.DisplayResults(algorithmHandler);
            for (int i = 0; i < TOTAL_GENERATIONS_TO_CALCULATE; ++i)
            {
                algorithmHandler.IterateNetworks(1);
                NetworkOutputDisplayer.DisplayResults(algorithmHandler);
            }

            Network bestNetwork = algorithmHandler.NetworkList[0];
            while (true)
            {
                Console.WriteLine("Enter a number");
                double[] inputValues = new double[] { Convert.ToDouble(Console.ReadLine()) };
                double[] expectedValues = new double[] { inputValues[0] % 2 == 0 ? 1 : 0, inputValues[0] % 2 == 1 ? 1 : 0 };
                bestNetwork.SetInputs(inputValues);
                bestNetwork.Propagate();
                NetworkOutputDisplayer.DisplayResult(bestNetwork, expectedValues);
            }
        }

        private static void InitializeLists()
        {
            TrainingSetGenerator.GenerateBinaryNumberDataSet();
            //TrainingSetGenerator.GenerateModulusDataSet(NUMBER_OF_TRAINING_NUMBERS);
            //TrainingSetGenerator.GenerateBitDetermineDataSet();
            inputList = TrainingSetGenerator.GetInputList();
            expectedList = TrainingSetGenerator.GetExpectedList();

            inputLayerSize = inputList[0].Length;
            outputLayerSize = expectedList[0].Length;
        }
    }
}
