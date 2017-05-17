using System.Collections.Generic;
using System.Linq;
using PongML.NeuralNetworks.Activation;
using PongML.NeuralNetworks.Structure;

namespace PongML.NeuralNetworks
{
    public class FeedFowardNetwork
    {
        public List<Layer> Layers { get; set; }
        public double LearningRate { get; set; }
        public IActivation Activation { get; set; }
        public int LayerCount
        {
            get
            {
                return Layers.Count;
            }
        }

        public FeedFowardNetwork(double learningRate, int[] layers, IActivation activation)
        {
            if (layers.Length < 2) return;

            this.LearningRate = learningRate;
            this.Layers = new List<Layer>();

            for(int l = 0; l < layers.Length; l++)
            {
                Layer layer = new Layer(layers[l]);
                this.Layers.Add(layer);

                for (int n = 0; n < layers[l]; n++)
                    layer.Neurons.Add(new Neuron());

                layer.Neurons.ForEach((nn) =>
                {
                    if (l == 0)
                        nn.Bias = 0;
                    else
                        for (int d = 0; d < layers[l - 1]; d++)
                            nn.Dendrites.Add(new Dendrite());
                });
            }

            this.Activation = activation;
        }

        public double[] Run(List<double> input)
        {
            if (input.Count != this.Layers[0].NeuronCount) return null;

            for (int l = 0; l < Layers.Count; l++)
            {
                Layer layer = Layers[l];

                int i = 0;
                layer.Neurons.ForEach(neuron =>
                {
                    if (l == 0)
                    {
                        neuron.Value = input[i++];
                    }
                    else
                    {
                        int np = 0;
                        neuron.Value = Layers[l - 1].Neurons.Sum(n => n.Value * neuron.Dendrites[np++].Weight);
                        neuron.Value = Activation.Output(neuron.Value + neuron.Bias);
                    }
                });
            }

            List<double> outputs = new List<double>();
            Layers[Layers.Count - 1].Neurons.ForEach(neuron => outputs.Add(neuron.Value));

            return outputs.ToArray();
        }
    }
}
