import NeuronNetwork from './NeuronNetwork';

const nn = new NeuronNetwork([5, 2], 0.1, 100000);

const trainData = [
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6],
];

const answers = [1, 0, 0, 1];
nn.train(trainData, answers);

const male = nn.feedForward([20, 2]);
console.log('Should be closer to 0: ', male);

const female = nn.feedForward([-7, -3]);
console.log('Should be closer to 1: ', female);
