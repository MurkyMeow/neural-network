import { dot, NTuple } from './math'

const enum Type {
  Up = 1,
  Down = -1,
}

type Point = [number, number]

// divides the points by the line y = x
const get_type = ([x, y]: Point) =>
  x >= y ? Type.Up : Type.Down

interface Model<T extends number> {
  weights: NTuple<T>
  learningRate: number
}

function guess<T extends number>(model: Model<T>, inputs: NTuple<T>): number {
  return Math.sign(dot(model.weights, inputs))
}

function learn<T extends number>(
  model: Model<T>,
  inputs: NTuple<T>,
  guess: number,
  actual: number,
): Model<T> {
  const error = actual - guess
  const weights = inputs.map((input, i) =>
    model.weights[i] + error * input * model.learningRate) as NTuple<T>
  return { ...model, weights }
}

let model: Model<2> = {
  weights: [0, 0],
  learningRate: 0.1,
}

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
    const prediction = guess(model, point)
    model = learn(model, point, prediction, actual)
    const [x, y] = point
    const text = prediction === Type.Up ? '⬆' : '⬇'
    ctx.fillStyle = actual === prediction ? 'green' : 'red'
    ctx.fillText(text, x * w, y * h)
  })
  requestAnimationFrame(run)
}())
