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
                int lastCurrentDataSet = 0;
                if(algorithmHandler.CurrentDataSets?.Count > 0)
                {
                    lastCurrentDataSet = algorithmHandler.CurrentDataSets.Last();
                }
                double[] expectedArray = algorithmHandler.ExpectedList[lastCurrentDataSet];
                DisplayResult(algorithmHandler.NetworkList[i], expectedArray);
            }
        }

        public static void DisplayResult(Network network, double[] expectedValues)
        {
            Console.WriteLine("ID: {0} | Average Fitness: {1}", network.ID, network.Fitness);
            Console.Write("Last Inputs: ");
            DisplayArray(network.GetInputs());

            Console.Write("Last Outputs: ");
            DisplayArray(network.GetOutputs());

            Console.Write("Last Expected: ");
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
