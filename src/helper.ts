export interface FixedArray<T extends any, L extends number> extends Array<T> {
  0: T
  length: L
}

export type NTuple<L extends number> =
  FixedArray<number, L>

export const dot = <A extends number, B extends A>(
  a: FixedArray<number, A>,
  b: FixedArray<number, B>,
): number => {
  return a.reduce((acc, el, i) => acc + el * b[i])
}
