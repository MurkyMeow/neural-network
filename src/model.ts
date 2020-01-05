import { Matrix, NTuple, fixedMap, dot, fixedMapPair, fixedArray } from './math'

// a random number between -0.5 and 0.5
const rand = (): number => Math.random() - 0.5

export class Model<I extends number, O extends number, L extends number> {
  layers: Matrix<L, I>
  outputs: Matrix<O, L>
  layerBias: NTuple<L>
  outputBias: NTuple<O>
  learningRate: number

  constructor(args: { inputSize: I, outputSize: O, layersCount: L, learningRate: number }) {
    this.layers = fixedArray(args.layersCount, () => fixedArray(args.inputSize, rand))
    this.outputs = fixedArray(args.outputSize, () => fixedArray(args.layersCount, rand))

    this.layerBias = fixedArray(args.layersCount, rand)
    this.outputBias = fixedArray(args.outputSize, rand)

    this.learningRate = args.learningRate
  }

  _nonlin(x: number): number {
    return Math.tanh(x)
  }

  _dnonlin(y: number): number {
    return 1 - y ** 2
  }

  guess(input: NTuple<I>): { hidden: NTuple<L>, output: NTuple<O> } {
    const { layers, layerBias, outputs, outputBias } = this
    const hidden = fixedMapPair(layers, layerBias, (h, b) => this._nonlin(dot(h, input) + b))
    const output = fixedMapPair(outputs, outputBias, (o, b) => this._nonlin(dot(o, hidden) + b))
    return { hidden, output }
  }

  train(input: NTuple<I>, desire: NTuple<O>): void {
    const { output, hidden } = this.guess(input)
  
    const outputErrors = fixedMapPair(output, desire,
      (y, yHat) => this._dnonlin(y) * (y - yHat))

    const newOutputs = fixedMapPair(this.outputs, outputErrors,
      (output, error, i) => fixedMap(output, w => w - error * hidden[i] * this.learningRate))

    const newOutBias = fixedMapPair(this.outputBias, outputErrors,
      (bias, error) => bias - error * this.learningRate)
  
    const hiddenErrors = fixedMap(hidden, (h, i) => {
      const weights = fixedMap(newOutputs, newOutput => newOutput[i])
  
      const errors = fixedMapPair(weights, outputErrors, (w, err) => w * err * this._dnonlin(h))
  
      return errors.reduce((acc, err) => acc + err)
    })

    const newLayers = fixedMapPair(this.layers, hiddenErrors, (layer, err) => {
      return fixedMapPair(layer, input, (w, x) => w - x * err * this.learningRate)
    })
  
    const newLayerBias = fixedMapPair(this.layerBias, hiddenErrors,
      (bias, err) => bias - err * this.learningRate)

    this.outputs = newOutputs
    this.outputBias = newOutBias
    this.layers = newLayers
    this.layerBias = newLayerBias
  }
}
