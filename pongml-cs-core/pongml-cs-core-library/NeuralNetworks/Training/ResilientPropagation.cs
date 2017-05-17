using PongML.NeuralNetworks.Structure;
using System;
using System.Collections.Generic;
using System.Linq;

namespace PongML.NeuralNetworks.Training
{
    public class ResilientPropagation : ITraining
    {
        private double etaPlus = 1.2;
        private double etaMinus = 0.5;
        private double deltaMax = 50.0;
        private double deltaMin = 1.0E-6;

        private double[][][] gradsAcc;
        private double[][][] prevGrads;
        private double[][][] prevDeltas;

        private double[][] biasGradsAcc;
        private double[][] prevBiasGrads;
        private double[][] prevBiasDeltas;

        private List<double> errors = new List<double>();

        public FeedFowardNetwork Network { get; set; }
        private double Error { get; set; }
        private List<List<double>> TrainingInputs { get; set; }
        private List<List<double>> TrainingOutputs { get; set; }

        public ResilientPropagation()
        {
            Error = 1.0;
        }

        private void InitializeMatrices()
        {
            gradsAcc = prevGrads = prevDeltas = new double[Network.Layers.Count][][];
            biasGradsAcc = prevBiasGrads = prevBiasDeltas = new double[Network.Layers.Count][];
            for (int l = 1; l < Network.Layers.Count; l++)
            {
                gradsAcc[l] = prevGrads[l] = prevDeltas[l] = new double[Network.Layers[l].Neurons.Count][];
                biasGradsAcc[l] = prevBiasGrads[l] = prevBiasDeltas[l] = new double[Network.Layers[l].Neurons.Count];
                for (int i = 0; i < Network.Layers[l].Neurons.Count; i++)
                {
                    gradsAcc[l][i] = prevGrads[l][i] = prevDeltas[l][i] = new double[Network.Layers[l - 1].Neurons.Count];
                    prevBiasDeltas[l][i] = 0.01;
                    for (int j = 0; j < Network.Layers[l - 1].Neurons.Count; j++)
                    {
                        prevDeltas[l][i][j] = 0.01;
                    }
                }
            }
        }

        private void ZeroOut()
        {
            for (int l = 1; l < Network.Layers.Count; l++)
            {
                for (int i = 0; i < Network.Layers[l].Neurons.Count; i++)
                {
                    biasGradsAcc[l][i] = 0.0;
                    for (int j = 0; j < Network.Layers[l - 1].Neurons.Count; j++)
                    {
                        gradsAcc[l][i][j] = 0.0;
                    }
                }
            }
        }

        private void CalculateGradient()
        {
            for (int t = 0; t < TrainingInputs.Count; ++t) //loop training data
            {
                //calculate individual 
                Network.Run(TrainingInputs[t]);
                for (int l = Network.Layers.Count - 1; l > 0; l--)
                {
                    for (int i = 0; i < Network.Layers[l].Neurons.Count; ++i)
                    {
                        Network.Layers[l].Neurons[i].Delta = l < Network.Layers.Count - 1
                            ? CalculateNonLastGradient(l + 1, i, Network.Layers[l].Neurons[i].Value)
                            : CalculateLastGradient(TrainingOutputs[t][i], Network.Layers[l].Neurons[i].Value);
                    }
                }

                //accumulate gradients
                for (int l = Network.Layers.Count - 1; l > 0; l--)
                {
                    for (int j = 0; j < Network.Layers[l].Neurons.Count; ++j)
                    {
                        double grad = Network.Layers[l].Neurons[j].Delta;
                        biasGradsAcc[l][j] += grad;

                        for (int i = 0; i < Network.Layers[l - 1].Neurons.Count; ++i)
                        {
                            grad = Network.Layers[l].Neurons[j].Delta * Network.Layers[l - 1].Neurons[i].Value;
                            gradsAcc[l][j][i] += grad;
                        }
                    }
                }

                int o = 0;
                Layer layer = Network.Layers[Network.Layers.Count - 1];
                errors.Add(layer.Neurons.Sum(n => Math.Abs(TrainingOutputs[t][o++] - n.Value)));
            }
            Error = errors.Average();
        }

        private double CalculateLastGradient(double ideal, double nValue)
        {
            return Network.Activation.Derivative(nValue) * (ideal - nValue);
        }
        private double CalculateNonLastGradient(int nextLayer, int j, double nValue)
        {
            double sum = 0.0;
            for (int i = 0; i < Network.Layers[nextLayer].Neurons.Count; i++)
            {
                sum += Network.Layers[nextLayer].Neurons[i].Delta * Network.Layers[nextLayer].Neurons[i].Dendrites[j].Weight;
            }
            return Network.Activation.Derivative(nValue) * sum;
        }

        //THIS IS NOT USED - WILL BE REMOVED WITH A BETTER SOLUTION LATER ON
        public bool Algorithm(List<double> input, List<double> ideal) { throw new NotImplementedException(); }
        public bool Algorithm(WeightComposite[] allGradsAcc, WeightComposite[] prevGradsAcc, WeightComposite[] prevDelta)
        {
            throw new NotImplementedException();
        }

        public bool Algorithm()
        {
            ZeroOut();

            //calculate gradients
            CalculateGradient();
            
            for (int l = 1; l < Network.Layers.Count; l++) //layers
            {
                for (int i = 0; i < Network.Layers[l - 1].Neurons.Count; ++i) //prev layer neurons
                {
                    for (int j = 0; j < Network.Layers[l].Neurons.Count; ++j) //current layer neurons
                    {
                        double delta = prevDeltas[l][j][i];
                        double change = prevGrads[l][j][i] * gradsAcc[l][j][i];
                        if (change > 0)
                        {
                            delta = Math.Min(delta * etaPlus, deltaMax);
                            double deltaWeight = -Math.Sign(gradsAcc[l][j][i]) * delta;
                            Network.Layers[l].Neurons[j].Dendrites[i].Weight += deltaWeight;
                        }
                        else if (change < 0)
                        {
                            delta = Math.Max(delta * etaMinus, deltaMin);
                            Network.Layers[l].Neurons[j].Dendrites[i].Weight -= prevDeltas[l][j][i];
                            prevGrads[l][j][i] = 0;
                        }
                        else
                        {
                            double deltaWeight = -Math.Sign(gradsAcc[l][j][i]) * delta;
                            Network.Layers[l].Neurons[j].Dendrites[i].Weight += deltaWeight;
                        }
                        prevGrads[l][j][i] = gradsAcc[l][j][i];
                        prevDeltas[l][j][i] = delta;
                    } //j
                } //i

                for (int i = 0; i < Network.Layers[l].Neurons.Count; ++i)
                {
                    double delta = prevBiasDeltas[l][i];
                    double change = prevBiasGrads[l][i] * biasGradsAcc[l][i];
                    if (change > 0)
                    {
                        delta = Math.Min(prevBiasDeltas[l][i] * etaPlus, deltaMax);
                        double biasDeltaWeight = -Math.Sign(biasGradsAcc[l][i]) * delta;
                        Network.Layers[l].Neurons[i].Bias += biasDeltaWeight;
                    }
                    else if (change < 0)
                    {
                        delta = Math.Max(prevBiasDeltas[l][i] * etaMinus, deltaMin);
                        Network.Layers[l].Neurons[i].Bias -= prevBiasDeltas[l][i];
                        prevBiasGrads[l][i] = 0;
                    }
                    else
                    {
                        double biasDeltaWeight = -Math.Sign(biasGradsAcc[l][i]) * delta;
                        Network.Layers[l].Neurons[i].Bias += biasDeltaWeight;
                    }
                    prevBiasGrads[l][i] = biasGradsAcc[l][i];
                    prevBiasDeltas[l][i] = delta;
                }
            }
            return true;
        }

        public void TrainToEpoch(ref FeedFowardNetwork network, List<List<double>> inputs, List<List<double>> ideals, int maxEpoch)
        {
            int epoch = 0;
            Network = network;
            TrainingInputs = inputs;
            TrainingOutputs = ideals;

            //initialise matrices
            InitializeMatrices();

            while (epoch < maxEpoch)
            {
                                if (epoch % 100 == 0 && epoch != int.MaxValue)
                {
                    double sumSquaredError = 0.0;
                    for (int row = 0; row < TrainingInputs.Count; row++)
                    {
                        Network.Run(TrainingInputs[row]);
                        for (int j = 0; j < Network.Layers[Network.Layers.Count - 1].Neurons.Count; j++)
                        {
                            Neuron n = Network.Layers[Network.Layers.Count - 1].Neurons[j];
                            sumSquaredError += ((n.Value - TrainingOutputs[row][j]) * (n.Value - TrainingOutputs[row][j]));
                        }
                    }
                    Console.WriteLine("Epoch: #{0} --- SqrError: {1}", epoch, (sumSquaredError / TrainingInputs.Count).ToString("F4"));
                }

                Algorithm();
                Console.WriteLine("Epoch: #{0} --- Error: {1}", epoch, Error);
                epoch++;
            }
        }
        public void TrainToError(ref FeedFowardNetwork network, List<List<double>> inputs, List<List<double>> ideals, double minError)
        {
            int epoch = 0;
            Network = network;
            TrainingInputs = inputs;
            TrainingOutputs = ideals;

            //initialise matrices
            InitializeMatrices();

            while (Error > minError && epoch < int.MaxValue)
            {
                if (epoch % 100 == 0 && epoch != int.MaxValue)
                {
                    double sumSquaredError = 0.0;
                    for (int row = 0; row < TrainingInputs.Count; row++)
                    {
                        Network.Run(TrainingInputs[row]);
                        for (int j = 0; j < Network.Layers[Network.Layers.Count - 1].Neurons.Count; j++)
                        {
                            Neuron n = Network.Layers[Network.Layers.Count - 1].Neurons[j];
                            sumSquaredError += ((n.Value - TrainingOutputs[row][j]) * (n.Value - TrainingOutputs[row][j]));
                        }
                    }
                    Console.WriteLine("Epoch: #{0} --- SqrError: {1}", epoch, (sumSquaredError / TrainingInputs.Count).ToString("F4"));
                }

                Algorithm();
                Console.WriteLine("Epoch: #{0} --- Error: {1}", epoch, Error);
                epoch++;
            }
        }
    }
}
