using System.Collections.Generic;

namespace PongML.NeuralNetworks.Training
{
    public interface ITraining
    {
        bool Algorithm();
        bool Algorithm(List<double> input, List<double> ideal);
        bool Algorithm(WeightComposite[] allGradsAcc, WeightComposite[] prevGradsAcc, WeightComposite[] prevDeltas);
        void TrainToEpoch(ref FeedFowardNetwork network, List<List<double>> inputs, List<List<double>> ideals, int maxEpoch);
        void TrainToError(ref FeedFowardNetwork network, List<List<double>> inputs, List<List<double>> ideals, double minError);
    }
}
