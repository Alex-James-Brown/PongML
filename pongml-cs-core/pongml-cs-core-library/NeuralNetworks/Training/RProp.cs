using PongML.NeuralNetworks.Structure;
using System;
using System.Collections.Generic;

namespace PongML.NeuralNetworks.Training
{
    class RProp : ITraining
    {
        //from paper - http://www.inf.fu-berlin.de/lehre/WS06/Musterererkennung/Paper/rprop.pdf
        public const double etaPlus = 1.2;
        public const double etaMinus = 0.5;
        public const double deltaMax = 50.0;
        public const double deltaMin = 1.0E-6;

        private List<List<double>> TrainingInputs;
        private List<List<double>> TrainingOutputs;
        private FeedFowardNetwork Network { get; set; }   

        public void TrainToEpoch(ref FeedFowardNetwork network, List<List<double>> inputs, List<List<double>> ideals, int maxEpoch)
        {
            Network = network;
            TrainingInputs = inputs;
            TrainingOutputs = ideals;

            //initialise matrices/vectors
            WeightComposite[] allGradsAcc = new WeightComposite[Network.Layers.Count];
            WeightComposite[] prevGradsAcc = new WeightComposite[Network.Layers.Count];
            WeightComposite[] prevDeltas = new WeightComposite[Network.Layers.Count];
            for (int i = 1; i < Network.Layers.Count; i++)
            {
                int size = Network.Layers[i].Neurons.Count;
                int prevSize = Network.Layers[i - 1].Neurons.Count;

                allGradsAcc[i].Biases = new double[size];
                allGradsAcc[i].Weights = MakeMatrix(size, prevSize, 0.0);

                prevGradsAcc[i].Biases = new double[size];
                prevGradsAcc[i].Weights = MakeMatrix(size, prevSize, 0.0);

                //0.01 = initial delta from paper
                prevDeltas[i].Biases = MakeVector(size, 0.01);
                prevDeltas[i].Weights = MakeMatrix(size, prevSize, 0.01);
            }

            //check the error is before training
            double[] initial_err = RootMeanSquaredError();
            Console.WriteLine("\nepoch = pre; err = {0:F4} [{1:F4}]", initial_err[0], initial_err[1]);
            if (initial_err[0] <= 0.001)
            {
                return; //no need to train
            }

            //begin training
            int epoch = 0;
            while (epoch < maxEpoch)
            {
                ++epoch;

                //reset accumulated gradients to 0.0
                for (int l = 1; l < Network.Layers.Count; l++)
                {
                    ZeroOut(allGradsAcc[l].Weights);
                    ZeroOut(allGradsAcc[l].Biases);
                    //Console.WriteLine("Zero'd out");
                }

                double[] err = new double[2];
                //compute accumulated gradients
                //double[] err = ComputeGraduate(allGradsAcc);
                for (int t = 0; t < TrainingInputs.Count; t++)
                {
                    network.Run(TrainingInputs[t]);

                    double[][] gradterms = new double[network.Layers.Count][];
                    for (int layer = network.Layers.Count - 1; layer > 0; layer--)
                    {
                        Console.WriteLine("Layer: {0}", layer);
                        //calculate gradient terms
                        gradterms[layer] = new double[network.Layers[layer].Neurons.Count];
                        if (layer < network.Layers.Count - 1)
                        {
                            for (int i = 0; i < network.Layers[layer].Neurons.Count; i++)
                            {
                                Console.WriteLine("Neuron: {0}", i);
                                double sum = 0.0;
                                for (int j = 0; j < network.Layers[layer + 1].Neurons.Count; j++)
                                {
                                    sum += gradterms[layer + 1][j] * network.Layers[layer + 1].Neurons[j].Dendrites[i].Weight;
                                    Console.WriteLine("Next Layer GradTerm: {0} --- Next Layer Weight: {1}",
                                        gradterms[layer + 1][j], network.Layers[layer + 1].Neurons[j].Dendrites[i].Weight);
                                }
                                gradterms[layer][i] = network.Activation.Derivative(network.Layers[layer].Neurons[i].Value) * sum;

                                Console.WriteLine("GradTerms: {0} --- Value: {1}", 
                                    gradterms[layer][i], network.Layers[layer].Neurons[i].Value);
                            }
                        }
                        else
                        {
                            for (int i = 0; i < network.Layers[layer].Neurons.Count; i++)
                            {
                                Console.WriteLine("Neuron: {0}", i);
                                gradterms[layer][i] = network.Activation.Derivative(network.Layers[layer].Neurons[i].Value) *
                                    (TrainingOutputs[t][i] - network.Layers[layer].Neurons[i].Value);

                                Console.WriteLine("GradTerms: {0} --- Value: {1} --- Ideal: {2}", 
                                    gradterms[layer][i], network.Layers[layer].Neurons[i].Value, TrainingOutputs[t][i]);
                            }
                        }

                        //accumulate gradients
                        for (int j = 0; j < network.Layers[layer].Neurons.Count; j++)
                        {
                            double grad = gradterms[layer][j];
                            allGradsAcc[layer].Biases[j] += grad;

                            for (int i = 0; i < network.Layers[layer - 1].Neurons.Count; i++)
                            {
                                grad = gradterms[layer][j] * network.Layers[layer - 1].Neurons[i].Value;
                                allGradsAcc[layer].Weights[j][i] += grad;
                            }
                        }
                    }

                    for (int j = 0; j < network.Layers[network.Layers.Count - 1].Neurons.Count; j++)
                    {
                        double error = Math.Pow(Network.Layers[network.Layers.Count - 1].Neurons[j].Value - TrainingInputs[t][j], 2);
                        err[0] += error / TrainingInputs.Count;
                        err[1] += error / TrainingInputs.Count / Network.Layers[network.Layers.Count - 1].Neurons.Count;
                    }
                }

                //run rprop algorithm
                Algorithm(allGradsAcc, prevGradsAcc, prevDeltas);

                //output error every 10 epochs or when error is 1%
                if (epoch % 1 == 0 || err[0] <= 0.01)
                {
                    double[] err_t = RootMeanSquaredError();
                    Console.WriteLine("\nepoch = {0} err = {1:F4} [{2:F4}]\ttest err = {3:F4} [{4:F4}]",
                        epoch, err[0], err[1], err_t[0], err_t[1]);
                    if (err[0] <= 0.01)
                    {
                        break;
                    }
                }
            }
        }

        private double[] RootMeanSquaredError()
        {
            int outputLayer = Network.Layers.Count - 1;
            int outputSize = Network.Layers[outputLayer].Neurons.Count;

            double sumSquaredError = 0.0;
            double sumSquaredErrorItem = 0.0;

            for (int t = 0; t < TrainingInputs.Count; t++)
            {
                Network.Run(TrainingInputs[t]);

                for (int j = 0; j < outputSize; ++j)
                {
                    double err = Math.Pow(Network.Layers[outputLayer].Neurons[j].Value - TrainingOutputs[t][j], 2);
                    sumSquaredError += err / TrainingInputs.Count;
                    sumSquaredErrorItem += err / TrainingInputs.Count / outputSize;
                }
            }
            double[] d = { Math.Sqrt(sumSquaredErrorItem), Math.Sqrt(sumSquaredError) };
            return d;
        }

        private double[] ComputeGraduate(WeightComposite[] allGradsAcc)
        {
            double[] sumSquaredErrors = { 0, 0 };
            int outputLayer = Network.Layers.Count - 1;
            int outputSize = Network.Layers[outputLayer].Neurons.Count;

            for (int t = 0; t < TrainingInputs.Count; t++) //iterate training inputs
            {
                Network.Run(TrainingInputs[t]); //compute outputs
                double[][] gradTerms = CalculateGradTerms(); //calculate the gradient terms for each neuron

                //accumulate gradients
                for (int layer = Network.Layers.Count - 1; layer > 0; layer--)
                {
                    for (int j = 0; j < Network.Layers[layer].Neurons.Count; ++j)
                    {
                        double grad = gradTerms[layer][j];
                        allGradsAcc[layer].Biases[j] += grad;

                        for (int i = 0; i < Network.Layers[layer - 1].Neurons.Count; ++i)
                        {
                            Console.WriteLine("Layer: #{0} --- Neuron: #{1} --- Training Item: #{2} --- GradTerm: {3:F10} --- " +
                                "PrevValue: {4}", layer, j, t, grad, Network.Layers[layer - 1].Neurons[i].Value);
                            grad = gradTerms[layer][j] * Network.Layers[layer - 1].Neurons[i].Value;
                            allGradsAcc[layer].Weights[j][i] += grad;
                        }
                    }
                }

                for (int j = 0; j < outputSize; ++j)
                {
                    double err = Math.Pow(Network.Layers[outputLayer].Neurons[j].Value - TrainingOutputs[t][j], 2);
                    sumSquaredErrors[0] += err / TrainingInputs.Count;
                    sumSquaredErrors[1] += err / TrainingInputs.Count / outputSize;
                }
            }

            return sumSquaredErrors;
        }

        private double[][] CalculateGradTerms()
        {
            double[][] gradTerms = new double[Network.Layers.Count][];
            for (int layer = Network.Layers.Count - 1; layer > 0; layer--)
            {
                Console.WriteLine("Layer: #{0}", layer);
                gradTerms[layer] = layer < Network.Layers.Count - 1
                    ? CalcuateGradTermsNonLast(Network.Layers[layer], Network.Layers[layer + 1], gradTerms[layer + 1])
                    : CalculateGradTermsLast(Network.Layers[layer], TrainingOutputs[layer].ToArray());
            }
            return gradTerms;
        }

        private double[] CalculateGradTermsLast (Layer layer, double[] tValues)
        {
            //usual backprop for output layer - f'(output) * (target - output)
            double[] gradTerms = new double[layer.Neurons.Count];
            for (int i = 0; i < layer.Neurons.Count; ++i)
            {
                gradTerms[i] = Network.Activation.Derivative(layer.Neurons[i].Value) * (tValues[i] - layer.Neurons[i].Value);
                Console.WriteLine("Neuron: #{0} --- Value: {1} --- Ideal: {2} --- GradTerm: {3:F10}",
                    i, layer.Neurons[i].Value, tValues[i], gradTerms[i]);
            }
            return gradTerms;
        }

        private double[] CalcuateGradTermsNonLast(Layer layer, Layer nextLayer, IReadOnlyList<double> nextGradTerms)
        {
            //usual backprop for hidden layers - f(output) * sum(gradTerms[j][i] * weights[j][i])
            double[] gradTerms = new double[layer.Neurons.Count];
            for (int i = 0; i < layer.Neurons.Count; ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < nextLayer.Neurons.Count; ++j)
                {
                    sum += nextGradTerms[j] * nextLayer.Neurons[j].Dendrites[i].Weight;
                }
                gradTerms[i] = Network.Activation.Derivative(layer.Neurons[i].Value) * sum;
                Console.WriteLine("Neuron: #{0} --- Value: {1} --- Sum: {2:F10} --- GradTerm: {3:F10}",
                    i, layer.Neurons[i].Value, sum, gradTerms[i]);
            }
            return gradTerms;
        }

        private int Sign(double v)
        {
            if ((Math.Abs(v) < 0.00001) && (Math.Abs(v) > -0.00001))
            {
                return 0;
            }
            if (v > 0.00001)
            {
                return 1;
            }
            return -1;
        }

        public bool Algorithm(WeightComposite[] allGradsAcc, WeightComposite[] prevGradsAcc, WeightComposite[] prevDeltas)
        {
            for (int layer = 1; layer < Network.Layers.Count; layer++) //loop layers
            {
                int size = Network.Layers[layer].Neurons.Count;
                int prevSize = Network.Layers[layer - 1].Neurons.Count;

                for (int i = 0; i < prevSize; ++i) //loop previous layers neurons
                {
                    for (int j = 0; j < size; ++j) //loop current layers neurons
                    {
                        double delta = prevDeltas[layer].Weights[j][i];
                        double change = prevGradsAcc[layer].Weights[j][i] * allGradsAcc[layer].Weights[j][i]; //get sign

                        Console.WriteLine("Layer: #{0} --- Neuron: #{1} --- Change: {2:F10} --- GradsAcc: {3} --- Prev GradsAcc: {4} " +
                            "-- GradsAcc EQ?: {5}",
                            layer, j, prevGradsAcc[layer].Weights[j][i] * allGradsAcc[layer].Weights[j][i], 
                            allGradsAcc[layer].Weights[j][i], prevGradsAcc[layer].Weights[j][i], 
                            (allGradsAcc[layer].Weights[j][i] == prevGradsAcc[layer].Weights[j][i]));

                        if (change > 0) //sign hasn't change, increase delta
                        {
                            delta = Math.Min(delta * etaPlus, deltaMax);
                            double dw = -Math.Sign(allGradsAcc[layer].Weights[j][i]) * delta;
                            Network.Layers[layer].Neurons[j].Dendrites[i].Weight += dw;
                            Console.WriteLine("Hello, I'm in the first branch. (WEIGHT)");
                        }
                        else if (change < 0) //sign changed, decrease delta
                        {
                            delta = Math.Max(delta * etaMinus, deltaMin);
                            Network.Layers[layer].Neurons[j].Dendrites[i].Weight -= prevDeltas[layer].Weights[j][i]; //revert to previous weight
                            allGradsAcc[layer].Weights[j][i] = 0;
                            Console.WriteLine("Hello, I'm in the second branch. (WEIGHT)");
                        }
                        else if (change == 0) //happens after 2nd branch
                        {
                            //no change to delta
                            //delta should not be 0
                            double dw = -Math.Sign(allGradsAcc[layer].Weights[j][i]) * delta;
                            Network.Layers[layer].Neurons[j].Dendrites[i].Weight += dw;
                            Console.WriteLine("Hello, I'm in the third branch. (WEIGHT)");
                        }

                        if (delta == 0)
                        {
                            Console.WriteLine("delta was 0");
                        }

                        Console.WriteLine("Layer: #{0} --- Neuron: #{1} --- Delta: {2} --- Prev Delta: {3} -- Delta EQ?: {4}",
                            layer, j, delta, prevDeltas[layer].Weights[j][i], (delta == prevDeltas[layer].Weights[j][i]));
                        prevDeltas[layer].Weights[j][i] = delta; //save delta
                        prevGradsAcc[layer].Weights[j][i] = allGradsAcc[layer].Weights[j][i]; //save accumulated gradient
                    }//j
                }//i

                for (int i = 0; i < size; ++i)
                {
                    double delta = prevDeltas[layer].Biases[i];
                    double change = prevGradsAcc[layer].Biases[i] * allGradsAcc[layer].Biases[i];

                    if (change > 0) //sign hasn't change, increase delta
                    {
                        delta = Math.Min(delta * etaPlus, deltaMax);
                        double dw = -Math.Sign(allGradsAcc[layer].Biases[i]) * delta;
                        Network.Layers[layer].Neurons[i].Bias += dw;
                        Console.WriteLine("Hello, I'm in the first branch. (BIAS)");
                    }
                    else if (change < 0) //sign changed, decrease delta
                    {
                        delta = Math.Max(delta * etaMinus, deltaMin);
                        Network.Layers[layer].Neurons[i].Bias -= prevDeltas[layer].Biases[i];
                        allGradsAcc[layer].Biases[i] = 0;
                        Console.WriteLine("Hello, I'm in the second branch. (BIAS)");
                    }
                    else if(change == 0) //happens after 2nd branch
                    {
                        //delta should not be 0
                        double dw = -Math.Sign(allGradsAcc[layer].Biases[i]) * delta;
                        Network.Layers[layer].Neurons[i].Bias += dw;
                        Console.WriteLine("Hello, I'm in the third branch. (BIAS)");
                    }

                    if (delta == 0)
                    {
                        Console.WriteLine("delta was 0");
                    }

                    Console.WriteLine("Layer: #{0} --- Bias: #{1} --- Delta: {2} --- Prev Delta: {3} -- Delta EQ?: {4}",
                        layer, i, delta, prevDeltas[layer].Biases[i], (delta == prevDeltas[layer].Biases[i]));

                    prevDeltas[layer].Biases[i] = delta; //save delta
                    prevGradsAcc[layer].Biases[i] = allGradsAcc[layer].Biases[i]; //save accumulated gradient
                }
            }
            return true;
        }

        private double[] MakeVector(int len, double v)
        {
            double[] result = new double[len];
            for (int i = 0; i < len; i++)
            {
                result[i] = v;
            }
            return result;
        }

        private double[][] MakeMatrix(int rows, int cols, double v)
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
            {
                result[r] = new double[cols];
            }
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    result[i][j] = v;
                }
            }
            return result;
        }
        
        private void ZeroOut(double[][] matrix)
        {
            foreach (var t in matrix)
            {
                for (int j = 0; j < t.Length; ++j)
                {
                    t[j] = 0.0;
                }
            }
        }
        private void ZeroOut(double[] array)
        {
            for (int i = 0; i < array.Length; ++i)
            {
                array[i] = 0.0;
            }
        }

        public void TrainToError(ref FeedFowardNetwork network, List<List<double>> inputs, List<List<double>> ideals, double minError)
        {
            throw new NotImplementedException();
        }

        public bool Algorithm() { throw new NotImplementedException(); }
        public bool Algorithm(List<double> input, List<double> ideal) { throw new NotImplementedException(); }

    }
    public struct WeightComposite
    {
        public double[][] Weights;
        public double[] Biases;
    }
}
