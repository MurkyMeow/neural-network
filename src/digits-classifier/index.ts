import { Model } from '../model'
import { NTuple } from '../math'
import data from './digits.json'

interface Digit {
  label: number
  samples: number[][]
}

function createCanvas(): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = 28
  canvas.height = 28
  document.body.append(canvas)
  return canvas
}

;(async function main() {
  const canvas = createCanvas()
  const ctx = canvas.getContext('2d')!

  const nn = new Model({
    learningRate: 0.1,
    inputSize: 784,
    layersCount: 64,
    outputSize: 3,
  })

  const desired = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]

  const digits = data as Digit[]

  digits.forEach(digit => {
    digit.samples.forEach(sample => {
      sample.forEach((x, i) => sample[i] = x / 255)
    })
  })

  const samples = digits.reduce((acc, x) =>
    acc.concat(x.samples
      .map(sample => ({ sample, label: x.label }))), [] as { sample: number[], label: number }[])

  for (let i = 0; i < 1000; i++) {
    const training = samples
      .map(a => [Math.random(), a] as const)
      .sort((a, b) => a[0] - b[0])
      .map(a => a[1])
    for (const sample of training) {
      nn.train(sample.sample as NTuple<784>, desired[sample.label] as NTuple<3>)
    }
  }

  for (const sample of samples) {
    const { output } = nn.guess(sample.sample as NTuple<784>)
    console.log(sample.label, output)
  }

  // const digit = digits[2]
  // const sample = digit.samples[0]

  // const nsample = sample.map(x => x / 255) as NTuple<784>

  // const colors: number[] = []
  // for (const opacity of sample) colors.push(0, 0, 0, opacity * 255)

  // const uintArray = new Uint8ClampedArray(colors)
  // const imageData = new ImageData(uintArray, 28, 28)
  // ctx.putImageData(imageData, 0, 0)

  // console.log(digit.label, nn.guess(nsample).output)
}())
