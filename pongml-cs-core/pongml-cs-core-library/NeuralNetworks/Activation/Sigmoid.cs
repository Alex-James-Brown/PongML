using System;

namespace PongML.NeuralNetworks.Activation
{
    public class Sigmoid : IActivation
    {
        public double Output(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
        public double Derivative(double x)
        {
            return x * (1 - x);
        }
    }
}
