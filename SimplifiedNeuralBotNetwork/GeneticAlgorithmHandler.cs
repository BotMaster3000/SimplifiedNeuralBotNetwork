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
        public List<int> CurrentDataSets { get; set; } = new List<int>();

        public List<Network> NetworkList { get; set; }

        public List<double[]> InputList { get; set; }
        public List<double[]> ExpectedList { get; set; }

        public int NumberOfNetworksToKeep { get; set; }
        public double MutationRate { get; set; }

        public int NumberOfDataSetsPerCycle { get; set; } = 1;

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
                SetCurrentDataSets();
                CycleNetworks();
                SortNetworks();
                RebreedNetworks();
                ++CurrentGeneration;
            }
        }

        private void SetCurrentDataSets()
        {
            CurrentDataSets.Clear();

            int currentDataSetToAdd = rand.Next(0, InputList.Count);

            for (int i = 0; i < NumberOfDataSetsPerCycle; ++i)
            {
                if (!CurrentDataSets.Contains(currentDataSetToAdd))
                {
                    CurrentDataSets.Add(currentDataSetToAdd);
                }
            }
        }

        private void CycleNetworks()
        {
            foreach (Network network in NetworkList)
            {
                double totalFitness = 0.0;
                foreach (int currentDataSet in CurrentDataSets)
                {
                    network.InputValues = InputList[currentDataSet];
                    network.Propagate();

                    totalFitness += network.CalculateFitness(ExpectedList[currentDataSet]);
                }

                network.Fitness = totalFitness / network.OutputLayerSize;
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
