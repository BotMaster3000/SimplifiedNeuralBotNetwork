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

        private List<double[]> inputList;
        public List<double[]> InputList
        {
            get { return inputList; }
            set { inputList = NormalizeData(value); }
        }
        public List<double[]> ExpectedList { get; set; }

        public int NumberOfNetworksToKeep { get; set; }
        public double MutationChance { get; set; }
        public double MutationRate { get; set; }

        public int NumberOfDataSetsPerCycle { get; set; } = 1;

        public int IdCounter { get; set; }

        public GeneticAlgorithmHandler(Random rand, List<Network> networkList)
        {
            this.rand = rand;
            NetworkList = networkList;
        }

        private List<double[]> NormalizeData(List<double[]> dataSetLists)
        {
            double minValue = dataSetLists[0][0];
            double maxValue = dataSetLists[0][0];

            foreach (double[] dataSet in dataSetLists)
            {
                foreach (double data in dataSet)
                {
                    if (data < minValue)
                    {
                        minValue = data;
                    }
                    if (data > maxValue)
                    {
                        maxValue = data;
                    }
                }
            }

            if (minValue != maxValue)
            {
                foreach (double[] dataSet in dataSetLists)
                {
                    for (int i = 0; i < dataSet.Length; ++i)
                    {
                        dataSet[i] = ((dataSet[i] - minValue) / (maxValue - minValue)) /*-0.5*/;
                    }
                }
                return dataSetLists;
            }
            else
            {
                throw new Exception("Inputs not normalizable");
            }
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

            for (int i = 0; i < NumberOfDataSetsPerCycle; ++i)
            {
                int currentDataSetToAdd = rand.Next(0, InputList.Count);
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
                    network.SetInputs(InputList[currentDataSet]);
                    network.Propagate();

                    totalFitness += network.CalculateFitness(ExpectedList[currentDataSet]);
                }

                network.Fitness = totalFitness / CurrentDataSets.Count;
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
            List<int> networkIdPool = CreateNumberPool();

            for (int i = NumberOfNetworksToKeep; i < NetworkList.Count; ++i)
            {
                for (int nodeIndex = 0; nodeIndex < NetworkList[i].HiddenNodes.Length; nodeIndex++)
                {
                    double[] currentWeights = new double[NetworkList[i].HiddenNodes[nodeIndex].Weights.Length];
                    for (int weightIndex = 0; weightIndex < currentWeights.Length; ++weightIndex)
                    {
                        int networkWeightsIndexToClone = networkIdPool[rand.Next(0, networkIdPool.Count)];
                        currentWeights[weightIndex] = NetworkList[networkWeightsIndexToClone].HiddenNodes[nodeIndex].Weights[weightIndex];
                    }
                    for(int weightIndex = 0; weightIndex < currentWeights.Length; ++weightIndex)
                    {
                        if(rand.NextDouble() < MutationChance)
                        {
                            double currentMutationFactor = rand.NextDouble() * MutationRate;
                            if(rand.Next(0, 2) == 0)
                            {
                                currentMutationFactor = -currentMutationFactor;
                            }
                            currentWeights[weightIndex] += currentMutationFactor;
                        }
                    }

                    NetworkList[i].HiddenNodes[nodeIndex].Weights = currentWeights;
                }

                for (int nodeIndex = 0; nodeIndex < NetworkList[i].OutputNodes.Length; nodeIndex++)
                {
                    double[] currentWeights = new double[NetworkList[i].OutputNodes[nodeIndex].Weights.Length];
                    for (int weightIndex = 0; weightIndex < currentWeights.Length; ++weightIndex)
                    {
                        int networkWeightsIndexToClone = networkIdPool[rand.Next(0, networkIdPool.Count)];
                        currentWeights[weightIndex] = NetworkList[networkWeightsIndexToClone].OutputNodes[nodeIndex].Weights[weightIndex];
                    }
                    for (int weightIndex = 0; weightIndex < currentWeights.Length; ++weightIndex)
                    {
                        if (rand.NextDouble() < MutationChance)
                        {
                            double currentMutationFactor = rand.NextDouble() * MutationRate;
                            if (rand.Next(0, 2) == 0)
                            {
                                currentMutationFactor = -currentMutationFactor;
                            }
                            currentWeights[weightIndex] += currentMutationFactor;
                        }
                    }

                    NetworkList[i].OutputNodes[nodeIndex].Weights = currentWeights;
                }

                NetworkList[i].ID = IdCounter; // New ID since its basically a new network
                ++IdCounter;
            }
        }

        private List<int> CreateNumberPool()
        {
            List<int> numberPool = new List<int>(); // Create a Pool. Highest Fitness-Network is most likely to breed
            for (int i = 0; i < NumberOfNetworksToKeep; ++i)
            {
                for (int j = i; j < NumberOfNetworksToKeep; ++j)
                {
                    numberPool.Add(i);
                }
            }
            return numberPool;
        }
    }
}
