import { sigmoid } from './uteis';

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
}
