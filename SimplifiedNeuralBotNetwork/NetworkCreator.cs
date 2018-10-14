using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplifiedNeuralBotNetwork
{
    public class NetworkCreator
    {
        public int InputLayerSize { get; }
        public int HiddenLayerSize { get; }
        public int OutputLayerSize { get; }
        public int NetworkAmount { get; }

        public List<Network> NetworkList { get; set; } = new List<Network>();

        private readonly Random rand;

        private static int idCounter;

        public NetworkCreator(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, int networkAmount, Random rand)
        {
            InputLayerSize = inputLayerSize;
            HiddenLayerSize = hiddenLayerSize;
            OutputLayerSize = outputLayerSize;
            NetworkAmount = networkAmount;

            this.rand = rand;

            InitializeNeuralNetworks();
        }

        private void InitializeNeuralNetworks()
        {
            for (int i = 0; i < NetworkAmount; ++i)
            {
                Network network = new Network(InputLayerSize, HiddenLayerSize, OutputLayerSize, rand, idCounter);
                NetworkList.Add(network);
                ++idCounter;
            }
        }
    }
}
