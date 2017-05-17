using System.Collections.Generic;
using System.Linq;
using PongML.NeuralNetworks.Structure;
using System;

namespace PongML.NeuralNetworks.Training
{
    public class Backpropagation : ITraining
    {
        private FeedFowardNetwork Network { get; set; }

        private void CalculateGradient(double[] TrainingOutputs)
        {
            for (int layer = Network.Layers.Count - 1; layer > 0; layer--)
            {
                if (layer < Network.Layers.Count - 1)
                {
                    CalcuateGradTermsNonLast(Network.Layers[layer], Network.Layers[layer + 1]);
                }
                else
                {
                    CalculateGradTermsLast(Network.Layers[layer], TrainingOutputs);
                }
            }
        }

        private void CalculateGradTermsLast(Layer layer, double[] tValues)
        {
            //usual backprop for output layer - f'(output) * (target - output)
            for (int i = 0; i < layer.Neurons.Count; ++i)
            {
                layer.Neurons[i].Delta = Network.Activation.Derivative(layer.Neurons[i].Value) * (tValues[i] - layer.Neurons[i].Value);
            }
        }

        private void CalcuateGradTermsNonLast(Layer layer, Layer nextLayer)
        {
            //usual backprop for hidden layers - f(output) * sum(gradTerms[j][i] * weights[j][i])
            for (int i = 0; i < layer.Neurons.Count; ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < nextLayer.Neurons.Count; ++j)
                {
                    sum += nextLayer.Neurons[j].Delta * nextLayer.Neurons[j].Dendrites[i].Weight;
                }
                layer.Neurons[i].Delta = Network.Activation.Derivative(layer.Neurons[i].Value) * sum;
            }
        }

        //private void CalculateGradient(double[] ideal)
        //{
        //    for (int l = Network.Layers.Count - 1; l > 0; l--)
        //    {
        //        for (int i = 0; i < Network.Layers[l].Neurons.Count; i++)
        //        {
        //            Network.Layers[l].Neurons[i].Delta = l < Network.Layers.Count - 1
        //                ? CalculateNonLastGradient(l + 1, i, Network.Layers[l].Neurons[i].Value)
        //                : CalculateLastGradient(ideal[i], Network.Layers[l].Neurons[i].Value);
        //        }
        //    }
        //}

        //private double CalculateLastGradient(double ideal, double nValue)
        //{
        //    return Network.Activation.Derivative(nValue) * (ideal - nValue);
        //}
        //private double CalculateNonLastGradient(int nextLayer, int j, double nValue)
        //{
        //    double sum = 0.0;
        //    for (int i = 0; i < Network.Layers[nextLayer].Neurons.Count; i++)
        //    {
        //        sum += Network.Layers[nextLayer].Neurons[i].Delta * Network.Layers[nextLayer].Neurons[i].Dendrites[j].Weight;
        //    }
        //    return Network.Activation.Derivative(nValue) * sum;
        //}

        //THIS IS NOT USED - WILL BE REMOVED WITH A BETTER SOLUTION LATER ON
        public bool Algorithm() { throw new NotImplementedException(); }
        public bool Algorithm(WeightComposite[] allGradsAcc, WeightComposite[] prevGradsAcc, WeightComposite[] prevDelta)
        {
            throw new NotImplementedException();
        }

        public bool Algorithm(List<double> input, List<double> ideal)
        {
            if ((input.Count != Network.Layers[0].Neurons.Count) || (ideal.Count != Network.Layers[Network.Layers.Count - 1].Neurons.Count)) return false;

            Network.Run(input);
            CalculateGradient(ideal.ToArray());
            //int x = 0;
            //Network.Layers[Network.Layers.Count - 1].Neurons.ForEach(n =>
            //{
            //    n.Delta = Network.Activation.Derivitive(n.Value) * (ideal[x++] - n.Value);
            //});
            //for (int l = Network.Layers.Count - 2; l > 0; l--)
            //{
            //    for (int i = 0; i < Network.Layers[l].Neurons.Count; i++)
            //    {
            //        double sum = 0.0;
            //        for (int j = 0; j < Network.Layers[l + 1].Neurons.Count; j++)
            //        {
            //            sum += Network.Layers[l + 1].Neurons[j].Delta * Network.Layers[l + 1].Neurons[j].Dendrites[i].Weight;
            //        }
            //        Network.Layers[l].Neurons[i].Delta = Network.Activation.Derivitive(Network.Layers[l].Neurons[i].Value) * sum;
            //    }
            //}
            //for(int i = 0; i < Network.Layers[Network.Layers.Count - 1].Neurons.Count; i++) 
            //{ 
            //    Neuron neuron = Network.Layers[Network.Layers.Count - 1].Neurons[i]; 
 
            //    neuron.Delta = neuron.Value * (1 - neuron.Value) * (ideal[i] - neuron.Value); 
 
            //    for(int j = Network.Layers.Count - 2; j > 0; j--) 
            //    { 
            //        for(int k = 0; k < Network.Layers[j].Neurons.Count; k++) 
            //        { 
            //            Neuron n = Network.Layers[j].Neurons[k]; 
 
            //            n.Delta = Network.Activation.Derivitive(n.Value) *
            //                      Network.Layers[j + 1].Neurons[i].Dendrites[k].Weight * 
            //                      Network.Layers[j + 1].Neurons[i].Delta; 
            //        } 
            //    } 
            //} 

            for (int i = Network.Layers.Count - 1; i > 1; i--)
            {
                for (int j = 0; j < Network.Layers[i].Neurons.Count; j++)
                {
                    Neuron n = Network.Layers[i].Neurons[j];
                    n.Bias = n.Bias + (Network.LearningRate * n.Delta);

                    for (int k = 0; k < n.Dendrites.Count; k++)
                        n.Dendrites[k].Weight = n.Dendrites[k].Weight + (Network.LearningRate * Network.Layers[i - 1].Neurons[k].Value * n.Delta);
                }
            }

            return true;
        }
        public void TrainToEpoch(ref FeedFowardNetwork network, List<List<double>> inputs, List<List<double>> ideals, int maxEpoch)
        {
            int epoch = 0;
            double error = 1.0;
            Network = network;

            while (epoch < maxEpoch)
            {
                var errors = new List<double>();
                for (int i = 0; i < inputs.Count; i++)
                {
                    Algorithm(inputs[i], ideals[i]);

                    int n = 0;
                    errors.Add(Network.Layers[Network.Layers.Count - 1].Neurons.Sum(a => System.Math.Abs(ideals[i][n++] - a.Value)));
                }
                error = errors.Average();
                Console.WriteLine("Epoch: #{0} --- Error: {1}", epoch, error);
                epoch++;
            }
        }
        public void TrainToError(ref FeedFowardNetwork network, List<List<double>> inputs, List<List<double>> ideals, double minError)
        {
            int epoch = 0;
            double error = 1.0;
            Network = network;

            while (error > minError && epoch < int.MaxValue)
            {
                var errors = new List<double>();
                for (int i = 0; i < inputs.Count; i++)
                {
                    Algorithm(inputs[i], ideals[i]);

                    int n = 0;
                    errors.Add(Network.Layers[Network.Layers.Count - 1].Neurons.Sum(a => System.Math.Abs(ideals[i][n++] - a.Value)));
                }
                error = errors.Average();
                Console.WriteLine("Epoch: #{0} --- Error: {1}", epoch, error);
                epoch++;
            }
        }
    }
}
