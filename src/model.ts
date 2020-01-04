import { Matrix, NTuple, fixedMap, dot, fixedMapPair, fixedArray } from './math'

export class Model<I extends number, O extends number, L extends number> {
  layers: Matrix<L, I>
  outputs: Matrix<O, L>
  learningRate: number
  layerBias: NTuple<L>
  outputBias: NTuple<O>

  constructor(args: { inputSize: I, outputSize: O, layersCount: L, learningRate: number }) {
    this.layers = fixedMap(fixedArray(args.layersCount), () => {
      return fixedMap(fixedArray(args.inputSize), Math.random)
    })
    this.outputs = fixedMap(fixedArray(args.outputSize), () => {
      return fixedMap(fixedArray(args.layersCount), Math.random)
    })

    this.layerBias = fixedMap(fixedArray(args.layersCount), Math.random)
    this.outputBias = fixedMap(fixedArray(args.outputSize), Math.random)

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

  train(input: NTuple<I>, expectation: NTuple<O>): void {
    const { output, hidden } = this.guess(input)
  
    const outputErrors = fixedMapPair(output, expectation, (y, yHat) =>
      (y - yHat) * this._dnonlin(y))

    const newOutputs = fixedMapPair(this.outputs, outputErrors,
      (output, error, i) => fixedMap(output, w => w - error * hidden[i] * this.learningRate))

    const newOutBias = fixedMapPair(this.outputBias, outputErrors,
      (bias, error) => bias - error * this.learningRate)
  
    const hiddenErrors = fixedMap(hidden, (_, i) => {
      const weights = fixedMap(newOutputs, newOutput => newOutput[i])
  
      const errors = fixedMapPair(weights, outputErrors, (w, err) => w * err)
  
      return errors.reduce((acc, err) => acc + err)
    })

    const newLayers = fixedMapPair(this.layers, hiddenErrors, (layer, err) => {
      return fixedMapPair(layer, input, (w, x) => w - x * err * this.learningRate)
    })
  
    const newLayerBias = fixedMapPair(this.layerBias, hidden,
      (bias, h, i) => bias - hiddenErrors[i] * this._dnonlin(h) * this.learningRate)

    this.outputs = newOutputs
    this.outputBias = newOutBias
    this.layers = newLayers
    this.layerBias = newLayerBias
  }
}
