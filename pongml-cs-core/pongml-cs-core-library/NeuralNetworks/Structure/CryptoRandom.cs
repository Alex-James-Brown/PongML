using System;
using System.Security.Cryptography;

namespace PongML.NeuralNetworks.Structure
{
    public class CryptoRandom
    {
        public double RandomValue { get; set; }

        public CryptoRandom()
        {
            using (var p = RandomNumberGenerator.Create())
            {
                Random r = new Random(p.GetHashCode());
                this.RandomValue = r.NextDouble();
            }
        }

    }
}
