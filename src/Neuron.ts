import { normalDistribution, sigmoid } from './helpers';

export default class Neuron {
    public weights: number[] = [];

    public bias = 0;

    setWeights(weights: number[]): void {
        this.weights = weights;
    }

    setBias(bias: number): void {
        this.bias = bias;
    }

    feedForward(input: number[]): number {
        if (this.weights.length !== input.length) {
            return -2;
        }

        let total = 0;

        this.weights.forEach((weight, index) => {
            total += weight * input[index];
        });

        total += this.bias;

        return sigmoid(total);
    }

    static generateRandomNeuron(connections: number, bias?: number): Neuron {
        const neuron = new Neuron();
        neuron.setBias(bias ?? normalDistribution());
        const weights: number[] = [];
        for (let y = 0; y < connections; y += 1) {
            weights.push(normalDistribution());
        }
        neuron.setWeights(weights);

        return neuron;
    }
}
