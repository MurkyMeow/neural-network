import fs from 'fs'
import path from 'path'
import fetch from 'node-fetch'

interface Digit {
  label: number
  samples: number[][]
}

const IMG_SIZE = 784
const IMG_AMOUNT = 10

function readImages(buffer: Buffer): number[][] {
  const images: number[][] = []

  for (let i = 0; i < IMG_SIZE * IMG_AMOUNT; i += IMG_SIZE) {
    const bytes: number[] = []

    for (let j = 0; j < IMG_SIZE; j++) bytes.push(buffer.readUInt8(i + j))

    images.push(bytes)
  }

  return images
}

;(async function main() {
  const digits: Digit[] = []

  for (let i = 0; i < 3; i++) {
    const url = `http://www.cis.jhu.edu/~sachin/digit/data${i}`
    const data = await fetch(url).then(res => res.buffer())
    const samples = readImages(data)
    digits.push({ label: i, samples })
  }

  const filename = path.join(__dirname, 'digits.json')

  fs.writeFileSync(filename, JSON.stringify(digits))
}())
