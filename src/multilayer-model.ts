import { Matrix, NTuple, fixedMap, dot, fixedMapPair } from './math'

interface Model<
  InputSize extends number,
  OutputSize extends number,
  LayersCount extends number,
> {
  layers: Matrix<LayersCount, InputSize>
  outputs: Matrix<OutputSize, LayersCount>
  learningRate: number

  layerBias: NTuple<LayersCount>
  outputBias: NTuple<OutputSize>
}

const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x))
const dsigmoid = (y: number): number => y * (1 - y)

function guess<I extends number, O extends number, L extends number>(
  model: Model<I, O, L>,
  input: NTuple<I>,
): { hidden: NTuple<L>, output: NTuple<O> } {
  const hidden = fixedMapPair(model.layers, model.layerBias, (h, b) => sigmoid(dot(h, input) + b))
  const output = fixedMapPair(model.outputs, model.outputBias, (o, b) => sigmoid(dot(o, hidden) + b))
  return { hidden, output }
}

function train<I extends number, O extends number, L extends number>(
  model: Model<I, O, L>,
  input: NTuple<I>,
  expectation: NTuple<O>,
): Model<I, O, L> {
  const { output, hidden } = guess(model, input)

  const outputErrors = fixedMapPair(output, expectation, (y, yHat) =>
    (y - yHat) * dsigmoid(y))

  const newOutputBias = fixedMapPair(model.outputBias, outputErrors,
    (bias, error) => bias - error * model.learningRate)

  const newOutputs = fixedMapPair(model.outputs, outputErrors,
    (output, error, i) => fixedMap(output, w => w - error * hidden[i] * model.learningRate))

  const hiddenErrors = fixedMap(hidden, (_, i) => {
    const weights = fixedMap(newOutputs, newOutput => newOutput[i])

    const errors = fixedMapPair(weights, outputErrors, (w, err) => w * err)

    return errors.reduce((acc, err) => acc + err)
  })

  const newLayers = fixedMapPair(model.layers, hiddenErrors, (layer, err) => {
    return fixedMapPair(layer, input, (w, x) => w - x * err * model.learningRate)
  })

  const newLayersBias = fixedMapPair(model.layerBias, hidden,
    (bias, h, i) => bias - hiddenErrors[i] * dsigmoid(h))

  return { ...model, layers: newLayers, outputs: newOutputs, outputBias: newOutputBias, layerBias: newLayersBias }
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
  learningRate: 0.1,
  layerBias: [Math.random(), Math.random(), Math.random()],
  layers: [
    [Math.random(), Math.random()],
    [Math.random(), Math.random()],
    [Math.random(), Math.random()],
  ],
  outputBias: [Math.random()],
  outputs: [
    [Math.random(), Math.random(), Math.random()],
  ],
}

// Train
for (let i = 0; i < 500000; i++) {
  const chunk = data[Math.random() * data.length | 0]
  model = train(model, chunk.input, chunk.expected)
}

// Test
console.table(
  data.map(chunk => ({
    input: chunk.input,
    expected: chunk.expected,
    prediction: guess(model, chunk.input).output,
  }))
)
