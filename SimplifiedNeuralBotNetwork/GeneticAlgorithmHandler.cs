using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplifiedNeuralBotNetwork
{
    public class GeneticAlgorithmHandler
    {
        private readonly Random rand;

        public int CurrentGeneration { get; private set; }
        public int CurrentDataSet { get; set; }

        public List<Network> NetworkList { get; set; }

        public List<double[]> InputList { get; set; }
        public List<double[]> ExpectedList { get; set; }

        public int NumberOfNetworksToKeep { get; set; }
        public double MutationRate { get; set; }

        public int IdCounter { get; set; }

        public GeneticAlgorithmHandler(Random rand, List<Network> networkList)
        {
            this.rand = rand;
            NetworkList = networkList;
        }

        public void IterateNetworks(int totalGenerationsToCalcualte)
        {
            for (int i = 0; i < totalGenerationsToCalcualte; ++i)
            {
                CurrentDataSet = rand.Next(0, InputList.Count);
                CycleNetworks();
                SortNetworks();
                RebreedNetworks();
                ++CurrentGeneration;
            }
        }

        private void CycleNetworks()
        {
            foreach (Network network in NetworkList)
            {
                network.InputValues = InputList[CurrentDataSet];
                network.Propagate();

                network.CalculateFitness(ExpectedList[CurrentDataSet]);
            }
        }

        private void SortNetworks()
        {
            for (int i = 0; i < NetworkList.Count; ++i)
            {
                for (int j = i + 1; j < NetworkList.Count; ++j)
                {
                    if (NetworkList[i].Fitness < NetworkList[j].Fitness)
                    {
                        Network tempNetwork = NetworkList[i];
                        NetworkList[i] = NetworkList[j];
                        NetworkList[j] = tempNetwork;
                    }
                }
            }
        }

        private void RebreedNetworks()
        {
            List<int> networkIdPool = new List<int>();
            for (int i = 0; i < NumberOfNetworksToKeep; ++i)
            {
                for (int j = i; j < NumberOfNetworksToKeep; ++j)
                {
                    networkIdPool.Add(i);
                }
            }
            for (int i = NumberOfNetworksToKeep; i < NetworkList.Count; ++i)
            {
                int networkIndexToClone = networkIdPool[rand.Next(0, networkIdPool.Count)];
                double[] currentHiddenWeights = (double[])NetworkList[networkIndexToClone].HiddenWeights.Clone();
                double[] currentOutputWeights = (double[])NetworkList[networkIndexToClone].OutputWeights.Clone();

                for (int j = 0; j < currentHiddenWeights.Length; j++)
                {
                    double currentMutationFactor = rand.NextDouble() * MutationRate;
                    if (rand.Next(0, 2) == 0)
                    {
                        currentMutationFactor = -currentMutationFactor;
                    }

                    currentHiddenWeights[j] += currentMutationFactor;
                }

                for (int j = 0; j < currentOutputWeights.Length; j++)
                {
                    double currentMutationFactor = rand.NextDouble() * MutationRate;
                    if (rand.Next(0, 2) == 0)
                    {
                        currentMutationFactor = -currentMutationFactor;
                    }

                    currentOutputWeights[j] += currentMutationFactor;
                }

                NetworkList[i].HiddenWeights = currentHiddenWeights;
                NetworkList[i].OutputWeights = currentOutputWeights;
                NetworkList[i].ID = IdCounter; // New ID since its basically a new network
                ++IdCounter;
            }
        }
    }
}
