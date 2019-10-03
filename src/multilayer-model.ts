 import { dot } from './helper'

const make_model = (
  layers: number[][],
  outputs: number[][],
  learning_rate: number = 0.01,
) => ({
  guess(inputs: number[]): number[] {
    const hidden = layers.map(layer => dot(layer, inputs))
    return outputs.map(output => dot(output, hidden))
  },
  learn(guess: number[], actual: number[]) {
    const out_errors = actual.map((val, i) => val - guess[i])
    const new_outputs = outputs.map((output, i) => {
      return output.map(w => w + out_errors[i] * actual[i])
    })
    const hidden_errors = layers.map(layer => {
      return dot(layer, out_errors)
    })
    const new_layers = layers.map((layer, i) => {
      return layer.map(w => w + hidden_errors[i] * actual[i])
    })
    return make_model(new_layers, new_outputs, learning_rate)
  },
})

// ---- PREDICTING XOR ----

const data = [{
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

let model = make_model(
  [
    [1, 1],
    [1, 1],
    [1, 1],
  ],
  [
    [1, 1, 1],
  ],
)

// Train
for (let i = 0; i < 1000; i++) {
  const chunk = data[Math.random() * data.length | 0]
  const guess = model.guess(chunk.input)
  model = model.learn(guess, chunk.expected)
}

// Test
for (const chunk of data) {
  console.log(chunk.input, model.guess(chunk.input))
}
