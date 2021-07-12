import Neuron from './Neuron';
import { calculateLoss, derivSigmoid } from './helpers';

interface NeuronExport {
    w: number[];
    b: number;
}

export default class NeuronNetwork {
    private learnRate = 0.1;

    private iterations = 1000;

    private layers: Neuron[][] = [];

    private output: Neuron;

    public showLoss = false;

    constructor(schema?: number[], learnRate?: number, iterations?: number) {
        if (schema) {
            for (let y = 0; y < schema[0]; y += 1) {
                this.layers.push([]);
                for (let x = 0; x < schema[1]; x += 1) {
                    this.layers[y].push(Neuron.generateRandomNeuron(schema[1]));
                }
            }

            this.output = Neuron.generateRandomNeuron(schema[1]);
        }

        if (iterations) {
            this.iterations = iterations;
        }
        if (learnRate) {
            this.learnRate = learnRate;
        }
    }

    feedForward(input: number[]): number {
        const layerResults: number[][] = [];

        this.layers.forEach((neurons, layer) => {
            layerResults[layer] = [];
            neurons.forEach(neuron => {
                const calculate = neuron.feedForward(input);
                layerResults[layer].push(calculate);
            });
        });

        return this.output.feedForward(layerResults[layerResults.length - 1]);
    }

    train(data: number[][], answers: number[]): void {
        if (data.length !== answers.length) {
            throw new Error('Invalid training data');
        }

        for (let epoch = 0; epoch < this.iterations; epoch += 1) {
            data.forEach((input, index) => {
                this.iteration(input, answers[index]);
            });

            if (this.showLoss) {
                if (epoch % 10 === 0) {
                    const preds: number[] = [];
                    data.forEach(item => preds.push(this.feedForward(item)));
                    const loss = calculateLoss(answers, preds);
                    console.log(`Loss at ${epoch} iteration: ${loss}`);
                }
            }
        }
    }

    iteration(input: number[], answer: number): void {
        const layerResults: number[][] = [];

        this.layers.forEach((neurons, layer) => {
            layerResults[layer] = [];
            neurons.forEach(neuron => {
                const calculate = neuron.feedForward(input);
                layerResults[layer].push(calculate);
            });
        });

        const lastLayerResults: number[] = layerResults[layerResults.length - 1];

        const output = this.output.feedForward(lastLayerResults);

        const partialDerivative = -2 * (answer - output);

        this.layers.forEach((neurons, layerIndex) => {
            neurons.forEach((neuron, neuronIndex) => {
                const newWeights: number[] = [];

                neuron.weights.forEach(weight => {
                    const w = input[neuronIndex] * derivSigmoid(layerResults[layerIndex][neuronIndex]);
                    newWeights.push(weight - this.learnRate * partialDerivative * (this.output.weights[neuronIndex] * derivSigmoid(output) * w));
                });

                neuron.setWeights(newWeights);

                neuron.setBias(
                    neuron.bias -
                        this.learnRate *
                            partialDerivative *
                            (this.output.weights[neuronIndex] * derivSigmoid(output) * derivSigmoid(layerResults[layerIndex][neuronIndex])),
                );
            });
        });

        const outputNewWeights: number[] = [];

        this.output.weights.forEach((weight, i) => {
            outputNewWeights.push(weight - this.learnRate * partialDerivative * (lastLayerResults[i] * derivSigmoid(output)));
        });

        this.output.setWeights(outputNewWeights);

        this.output.setBias(this.output.bias - this.learnRate * partialDerivative * derivSigmoid(output));
    }

    save(): NeuronExport[][] {
        const minimalData: NeuronExport[][] = [];

        this.layers.forEach((layer, layerIndex) => {
            minimalData[layerIndex] = [];
            layer.forEach(neuron => {
                const neuronData: NeuronExport = {
                    w: neuron.weights,
                    b: neuron.bias,
                };
                minimalData[layerIndex].push(neuronData);
            });
        });

        minimalData.push([
            {
                w: this.output.weights,
                b: this.output.bias,
            },
        ]);

        return minimalData;
    }

    load(data: NeuronExport[][]): void {
        data.forEach((item, itemIndex) => {
            const layer: Neuron[] = [];
            item.forEach(neuron => {
                const newNeuron = new Neuron();
                newNeuron.setWeights(neuron.w);
                newNeuron.setBias(neuron.b);

                if (itemIndex + 1 === data.length) {
                    this.output = newNeuron;
                } else {
                    layer.push(newNeuron);
                }
            });
            if (layer.length > 0) {
                this.layers.push(layer);
            }
        });
    }
}
