using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplifiedNeuralBotNetwork
{
    public static class NetworkOutputDisplayer
    {
        public static int TotalNumberOfNetworksToDisplayEachGeneration { get; set; } = 10;

        public static void DisplayResults(GeneticAlgorithmHandler algorithmHandler)
        {
            Console.ForegroundColor = ConsoleColor.DarkRed;
            Console.WriteLine("CurrentGeneration: " + algorithmHandler.CurrentGeneration);
            Console.ForegroundColor = ConsoleColor.Gray;

            for (int i = 0; i < TotalNumberOfNetworksToDisplayEachGeneration && i < algorithmHandler.NetworkList.Count; i++)
            {
                DisplayResult(algorithmHandler.NetworkList[i], algorithmHandler.ExpectedList[algorithmHandler.CurrentDataSet]);
            }
        }

        public static void DisplayResult(Network network, double[] expectedValues)
        {
            Console.WriteLine("ID: {0} | Fitness: {1}", network.ID, network.Fitness);
            Console.Write("Inputs: ");
            DisplayArray(network.InputValues);

            Console.Write("Outputs: ");
            DisplayArray(network.OutputValues);

            Console.Write("Expected: ");
            DisplayArray(expectedValues);

            Console.WriteLine();
        }

        private static void DisplayArray(double[] valueArray)
        {
            foreach(double value in valueArray)
            {
                Console.Write(value + " | ");
            }
            Console.WriteLine();
        }
    }
}
