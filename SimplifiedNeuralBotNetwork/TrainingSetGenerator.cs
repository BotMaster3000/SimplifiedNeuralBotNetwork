using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplifiedNeuralBotNetwork
{
    public static class TrainingSetGenerator
    {
        private static readonly List<double[]> inputList = new List<double[]>();
        private static readonly List<double[]> expectedList = new List<double[]>();

        public static List<double[]> GetCopyOfList(List<double[]> listToCopy)
        {
            List<double[]> returnList = new List<double[]>();
            foreach (double[] doubleArray in listToCopy)
            {
                returnList.Add((double[])doubleArray.Clone());
            }

            return returnList;
        }

        public static List<double[]> GetInputList()
        {
            return GetCopyOfList(inputList);
        }

        public static List<double[]> GetExpectedList()
        {
            return GetCopyOfList(expectedList);
        }

        private static void ResetLists()
        {
            inputList.Clear();
            expectedList.Clear();
        }

        public static void GenerateModulusDataSet(int numberOfTrainingSets)
        {
            ResetLists();
            for (int i = 1; i <= numberOfTrainingSets; ++i)
            {
                inputList.Add(new double[] { i });
                expectedList.Add(new double[] { i % 2 });
            }
        }
    }
}
