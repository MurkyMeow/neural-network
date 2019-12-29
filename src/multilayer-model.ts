import { Matrix, matdot } from './helper'

const normalize = (x: number): number =>
  1 / (1 - Math.exp(-x))

const get_change_rate = (x: number): number =>
  normalize(x) * (1 - normalize(x))

interface Model<
  InputSize extends number,
  OutputSize extends number,
  LayersCount extends number,
> {
  layers: Matrix<LayersCount, InputSize>
  outputs: Matrix<OutputSize, LayersCount>
  learningRate: number
}

function guess<I extends number, O extends number, L extends number>(
  model: Model<I, O, L>,
  inputs: Matrix<I, 1>,
): Matrix<O, 1> {
  const hidden = matdot(model.layers, inputs)
  return matdot(model.outputs, hidden)
}

function learn<I extends number, O extends number, L extends number>(
  model: Model<I, O, L>,
  guess: Matrix<O, 1>,
  actual: Matrix<O, 1>,
): Model<I, O, L> {
  const out_errors = actual.map((val, i) => (val - guess[i]) ** 2)
  const out_delta = guess
    .map(get_change_rate)
    .map((direction, i) => direction * out_errors[i] * model.learningRate)
  const new_outputs = model.outputs
    .map((output, i) => output.map(w => w - out_delta[i]))
  return { ...model, outputs: new_outputs }
}

// ---- PREDICTING XOR ----

interface Chunk {
  input: Matrix<2, 1>
  expected: Matrix<1, 1>
}

const data: Chunk[] = [{
  input: [
    [1],
    [1],
  ],
  expected: [
    [0],
  ],
}, {
  input: [
    [1],
    [0],
  ],
  expected: [
    [1]
  ],
}, {
  input: [
    [0],
    [1],
  ],
  expected: [
    [1],
  ],
}, {
  input: [
    [0],
    [0],
  ],
  expected: [
    [1],
  ],
}]

let model: Model<2, 1, 3> = {
  learningRate: 0.01,
  layers: [
    [1, 1],
    [1, 1],
    [1, 1],
  ],
  outputs: [
    [1, 1, 1],
  ],
}

// Train
for (let i = 0; i < 1000; i++) {
  const chunk = data[Math.random() * data.length | 0]
  const prediction = guess(model, chunk.input)
  model = learn(model, prediction, chunk.expected)
}

// Test
for (const chunk of data) {
  console.log(chunk.input, guess(model, chunk.input))
}
