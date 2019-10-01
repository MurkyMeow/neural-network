function err(msg: string): never {
  throw new Error(msg)
}

export const dot = (a: number[], b: number[]) =>
  a.length === b.length
    ? a.reduce((acc, el, i) => acc + el * b[i], 0)
    : err('vectors must have equal lengths')
