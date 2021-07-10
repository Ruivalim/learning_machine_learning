import NeuronNetwork from './NeuronNetwork';

//  One neuron for each parameter in the trainData;
const nn = new NeuronNetwork(2, 0.1, 10000);

const trainData = [
  [-2, -1],
  [25, 6],
  [17, 4],
  [-15, -6],
];

const rights = [1, 0, 0, 1];

nn.train(trainData, rights);

const male = nn.feedForward([20, 2]); // Should be closer to 0

console.log(male);

const female = nn.feedForward([-7, -3]); // Should be closer to 1

console.log(female);
