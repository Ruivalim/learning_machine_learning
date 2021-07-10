import NeuronNetwork from './NeuronNetwork';

const nn = new NeuronNetwork(3);

const trainData = [
  [-2, -1],
  [25, 6],
  [17, 4],
  [-15, -6],
];

const rights = [1, 0, 0, 1];

nn.train(trainData, rights);

const teste = nn.feedForward([20, 2]);

console.log(teste);

const teste2 = nn.feedForward([-7, -3]);

console.log(teste2);
