import Neuron from './Neuron';
import { calculateLoss, derivSigmoid, normalDistribution } from './uteis';

export default class NeuronNetwork {
  private learnRate = 0.1;

  private ticks = 1000;

  private neurons: Neuron[] = [];

  private output: Neuron;

  constructor(neuronsNumber: number, learnRate: number, ticks: number) {
    for (let y = 0; y < neuronsNumber; y += 1) {
      this.neurons.push(new Neuron());
    }

    this.output = new Neuron();

    this.ticks = ticks;
    this.learnRate = learnRate;
  }

  feedForward(input: number[]): number {
    const neuronResults: number[] = [];

    this.neurons.forEach(neuron => {
      const calculate = neuron.feedForward(input);
      neuronResults.push(calculate);
    });

    return this.output.feedForward(neuronResults);
  }

  train(data: number[][], right: number[]): void {
    if (data.length !== right.length) {
      throw new Error('Generic error 1');
    }

    for (let epoch = 0; epoch < this.ticks; epoch += 1) {
      data.forEach((input, index) => {
        this.tick(input, right[index]);
      });

      if (epoch % 10 === 0) {
        const preds: number[] = [];
        data.forEach(item => preds.push(this.feedForward(item)));
        const loss = calculateLoss(right, preds);
        console.log(`Loss at ${loss}`);
      }
    }
  }

  tick(input: number[], value: number): void {
    this.neurons.forEach(neuron => {
      if (neuron.weights.length === 0) {
        neuron.setBias(normalDistribution());
        const weights: number[] = [];
        for (let y = 0; y < input.length; y += 1) {
          weights.push(normalDistribution());
        }
        neuron.setWeights(weights);
      }
    });

    if (this.output.weights.length === 0) {
      this.output.setBias(normalDistribution());
      const weights: number[] = [];
      for (let y = 0; y < input.length; y += 1) {
        weights.push(normalDistribution());
      }
      this.output.setWeights(weights);
    }

    const neuronResults: number[] = [];

    this.neurons.forEach(neuron => {
      const calculate = neuron.feedForward(input);
      neuronResults.push(calculate);
    });

    const output = this.output.feedForward(neuronResults);

    const partialDerivative = -2 * (value - output);

    this.neurons.forEach((neuron, index) => {
      const newWeights: number[] = [];

      neuron.weights.forEach((weight, i) => {
        const w = input[index] * derivSigmoid(neuronResults[index]);
        newWeights.push(weight - this.learnRate * partialDerivative * (this.output.weights[index] * derivSigmoid(output) * w));
      });

      neuron.setBias(
        neuron.bias - this.learnRate * partialDerivative * (this.output.weights[index] * derivSigmoid(output) * derivSigmoid(neuronResults[index])),
      );
    });

    const outputNewWeights: number[] = [];

    this.output.weights.forEach((weight, i) => {
      outputNewWeights.push(weight - this.learnRate * partialDerivative * (neuronResults[i] * derivSigmoid(output)));
    });

    this.output.setWeights(outputNewWeights);

    this.output.setBias(this.output.bias - this.learnRate * partialDerivative * derivSigmoid(output));
  }
}
