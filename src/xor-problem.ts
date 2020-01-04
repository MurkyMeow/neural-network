import { NTuple } from './math'
import { Model } from './model'

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

const model = new Model({
  learningRate: 0.1,
  inputSize: 2,
  outputSize: 1,
  layersCount: 3,
})

// Train
for (let i = 0; i < 50000; i++) {
  const chunk = data[Math.random() * data.length | 0]
  model.train(chunk.input, chunk.expected)
}

// Test
console.table(
  data.map(chunk => ({
    input: chunk.input,
    expected: chunk.expected,
    prediction: model.guess(chunk.input).output,
  }))
)
