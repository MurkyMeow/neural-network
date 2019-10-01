import P5 from 'p5'
import { dot } from './helper'

const enum Type {
  Up = 1,
  Down = -1,
}

type Point = [number, number]

// divides the points by the line y = x
const get_type = ([x, y]: Point) =>
  x >= y ? Type.Up : Type.Down

const make_model = (weights: number[], learning_rate: number = 0.0001) => ({
  guess(inputs: number[]): number {
    return Math.sign(dot(weights, inputs))
  },
  learn(inputs: number[], guess: number, actual: number) {
    const error = actual - guess
    const new_w = inputs.map((input, i) => weights[i] + error * input * learning_rate)
    return make_model(new_w, learning_rate)
  },
});

let model = make_model([0, 0])

export default (p: P5) => {
  p.frameRate(1)
  p.textSize(35)
  p.textAlign('center')
  const data: Point[] = Array(80).fill(0).map(() => [
    Math.random() * p.width,
    Math.random() * p.height,
  ])
  p.draw = function() {
    p.background('#323232')
    p.line(0, 0, p.width, p.height).strokeWeight(2)
    data.forEach(point => {
      const [x, y] = point
      const actual = get_type(point)
      const guess = model.guess(point)
      model = model.learn(point, guess, actual)
      p.fill(actual === guess ? 'green' : 'red')
      p.text(guess === Type.Up ? '⬆' : '⬇', x, y)
      p.stroke('#000')
    })
  }
}
