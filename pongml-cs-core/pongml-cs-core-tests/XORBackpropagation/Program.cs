using System;
using System.Collections.Generic;
using System.Linq;
using PongML.NeuralNetworks;
using PongML.NeuralNetworks.Activation;
using PongML.NeuralNetworks.Training;

namespace XORTest
{
    class Program
    {
        public static FeedFowardNetwork network = new FeedFowardNetwork(10.5, new[] { 2, 2, 1 }, new Sigmoid());

        static void Main(string[] args)
        {
            List<List<double>> ins = new List<List<double>>();
            ins.Add(new[] { 0.0, 0.0 }.ToList());
            ins.Add(new[] { 1.0, 0.0 }.ToList());
            ins.Add(new[] { 0.0, 1.0 }.ToList());
            ins.Add(new[] { 0.0, 0.0 }.ToList());

            List<List<double>> ots = new List<List<double>>();
            ots.Add(new[] { 0.0 }.ToList());
            ots.Add(new[] { 1.0 }.ToList());
            ots.Add(new[] { 1.0 }.ToList());
            ots.Add(new[] { 0.0 }.ToList());

            ITraining trainer = new Backpropagation();
            trainer.TrainToError(ref network, ins, ots, 0.01);

            foreach (var item in ins)
            {
                var output = network.Run(item);
                foreach (var n in network.Layers[0].Neurons)
                {
                    Console.WriteLine("Input: {0}", n.Value);
                }
                Console.WriteLine("{0}", output[0]);
            }
            Console.ReadLine();
        }
    }
}
