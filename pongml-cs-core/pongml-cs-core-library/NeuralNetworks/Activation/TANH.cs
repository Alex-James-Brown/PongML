using System;

namespace PongML.NeuralNetworks.Activation
{
    class TANH : IActivation
    {
        public double Output(double d)
        {
            if (d < -45.0) return -1.0;
            else if (d > 45.0) return 1.0;
            else return Math.Tanh(d);
        }
        public double Derivative(double d)
        {
            return (1 - d) * (1 + d);
        }
    }
}
