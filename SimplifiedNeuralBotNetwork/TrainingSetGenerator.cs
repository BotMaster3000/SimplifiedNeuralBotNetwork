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
                double[] expectedDouble = new double[]
                {
                    i % 2
                };
                expectedList.Add(expectedDouble);
            }
        }

        public static void GenerateBitDetermineDataSet()
        {
            ResetLists();

            inputList.Add(new double[] { 0 });
            expectedList.Add(new double[] { 0 });

            inputList.Add(new double[] { 1 });
            expectedList.Add(new double[] { 1 });
        }

        public static void GenerateBinaryNumberDataSet()
        {
            ResetLists();

            for (int i = 0; i < 255; ++i)
            {
                char[] numberArray = Convert.ToString(i, 2).PadLeft(8, '0').ToCharArray();

                double[] tempDouble = new double[numberArray.Length];
                for (int j = 0; j < numberArray.Length; j++)
                {
                    switch (numberArray[j])
                    {
                        case '0':
                            tempDouble[j] = 0;
                            break;
                        case '1':
                            tempDouble[j] = 1;
                            break;
                        default:
                            throw new Exception();
                    }
                }
                inputList.Add(tempDouble);
                expectedList.Add(new double[] { i > 127 ? 1 : 0 });
            }
        }
    }
}
