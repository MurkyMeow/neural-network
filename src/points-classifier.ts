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

const data: Point[] = Array(80).fill(0).map(() => [
  Math.random(),
  Math.random(),
])

const [w, h] = [640, 480]
const ctx = document.createElement('canvas').getContext('2d')!
ctx.canvas.width = w
ctx.canvas.height = h
ctx.font = '3rem Veranda'
document.body.append(ctx.canvas)

const dt = 1000;
(function run(time = performance.now()) {
  if (performance.now() - time < dt) {
    requestAnimationFrame(() => run(time))
    return
  }
  ctx.clearRect(0, 0, w, h)
  ctx.beginPath()
  ctx.moveTo(0, 0)
  ctx.lineTo(w, h)
  ctx.stroke()
  data.forEach(point => {
    const actual = get_type(point)
    const guess = model.guess(point)
    model = model.learn(point, guess, actual)
    const [x, y] = point
    const text = guess === Type.Up ? '⬆' : '⬇'
    ctx.fillStyle = actual === guess ? 'green' : 'red'
    ctx.fillText(text, x * w, y * h)
  })
  requestAnimationFrame(run)
}())
