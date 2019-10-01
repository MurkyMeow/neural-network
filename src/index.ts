import P5 from 'p5'
import initSample from './points-classifier'

new P5((p: P5) => Object.assign(p, {
  setup() {
    p.resizeCanvas(400, 400)
    initSample(p)
  },
}), document.body)
