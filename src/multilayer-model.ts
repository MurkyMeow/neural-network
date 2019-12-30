import { Matrix, NTuple, fixedMap, dot } from './math'

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
  input: NTuple<I>,
): NTuple<O> {
  const hidden = fixedMap(model.layers, row => dot(row, input))
  return fixedMap(model.outputs, row => dot(row, hidden))
}

function learn<I extends number, O extends number, L extends number>(
  model: Model<I, O, L>,
  input: NTuple<I>,
  expectation: NTuple<O>,
): Model<I, O, L> {
  const hidden = fixedMap(model.layers, row => dot(input, row))
  const prediction = fixedMap(model.outputs, row => dot(row, hidden))

  const outErrors = fixedMap(prediction, (x, i) =>
    0.5 * (x - expectation[i]) ** 2)

  const newOutputs = fixedMap(model.outputs, (output, i) =>
    fixedMap(output, x => normalize(x + outErrors[i])))

  return { ...model, outputs: newOutputs }
}

// ---- PREDICTING XOR ----

interface Chunk {
  input: NTuple<2>
  expected: NTuple<1>
}

const data: Chunk[] = [{
  input: [1, 1],
  expected: [0],
}, {
  input: [1, 0],
  expected: [1],
}, {
  input: [0, 1],
  expected: [1],
}, {
  input: [0, 0],
  expected: [1],
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
  model = learn(model, chunk.input, chunk.expected)
}

// // Test
for (const chunk of data) {
  console.log(chunk.input, guess(model, chunk.input))
}
