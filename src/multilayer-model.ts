import { Matrix, NTuple, fixedMap, dot } from './math'

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

function train<I extends number, O extends number, L extends number>(
  model: Model<I, O, L>,
  input: NTuple<I>,
  expectation: NTuple<O>,
): Model<I, O, L> {
  const hidden = fixedMap(model.layers, row => dot(input, row))
  const prediction = fixedMap(model.outputs, row => dot(row, hidden))

  const outErrors = fixedMap(prediction, (x, i) => x - expectation[i])

  const newOutputs = fixedMap(model.outputs, (output, i) => {
    return fixedMap(output, x => x - Math.sign(outErrors[i]) * model.learningRate)
  })

  const totalErrors = outErrors.reduce((acc, el) => acc + el)

  const hiddenErrors = fixedMap(hidden, el => el * totalErrors)

  const newLayers = fixedMap(model.layers, (layer, i) => {
    return fixedMap(layer, el => el - Math.sign(hiddenErrors[i]) * model.learningRate)
  })

  return { ...model, layers: newLayers, outputs: newOutputs }
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
  expected: [0],
}]

let model: Model<2, 1, 3> = {
  learningRate: 0.001,
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
for (let i = 0; i < 50000; i++) {
  const chunk = data[Math.random() * data.length | 0]
  model = train(model, chunk.input, chunk.expected)
}

// // Test
for (const chunk of data) {
  console.log(chunk.input, guess(model, chunk.input))
}
