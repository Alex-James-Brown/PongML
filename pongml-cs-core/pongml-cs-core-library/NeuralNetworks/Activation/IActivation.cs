namespace PongML.NeuralNetworks.Activation
{
    public interface IActivation
    {
        double Output(double x);
        double Derivative(double x);
    }
}
