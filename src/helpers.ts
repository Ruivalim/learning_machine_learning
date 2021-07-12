export const average = (array: number[]): number => array.reduce((a, b) => a + b) / array.length;

export const sigmoid = (x: number): number => {
    return 1 / (1 + Math.E ** -x);
};

export const calculateLoss = (rightValues: number[], predValues: number[]): number => {
    if (rightValues.length !== predValues.length) {
        throw new Error('Right values and pred values length not match');
    }

    const numbers: number[] = [];

    rightValues.forEach((value, index) => {
        numbers.push((value - predValues[index]) ** 2);
    });

    return average(numbers);
};

export const derivSigmoid = (x: number): number => {
    const fx = sigmoid(x);
    return fx * (1 - fx);
};

export function normalDistribution(): number {
    return Math.round(Math.random());
}
