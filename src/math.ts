export interface FixedArray<T extends any, L extends number> extends Array<T> {
  0: T
  length: L
}

export type NTuple<L extends number> =
  FixedArray<number, L>

export type Matrix<R extends number, C extends number> =
  FixedArray<NTuple<C>, R>

function getFixedArr<T, L extends number>(size: L, fill: T): FixedArray<T, L> {
  return Array(size).fill(fill) as FixedArray<T, L>
}

export function fixedMap<T, L extends number, R>(
  arr: FixedArray<T, L>,
  fn: (element: T, index: number) => R,
): FixedArray<R, L> {
  return arr.map(fn) as FixedArray<R, L>
}

export function dot<A extends number, B extends A>(
  a: FixedArray<number, A>,
  b: FixedArray<number, B>,
): number {
  return a.reduce((acc, el, i) => acc + el * b[i])
}

// this uses fixed arrays to disallow multiplying matrices
// with incompatible dimensions in *compile time*
// and to infer the resulting dimensions automatically
export function matdot<R1 extends number, C1 extends number, R2 extends C1, C2 extends number>(
  a: Matrix<R1, C1>,
  b: Matrix<R2, C2>,
): Matrix<R1, C2> {
  return fixedMap(a, aRow => {
    return fixedMap(getFixedArr(b[0].length, 0), (_, i) => {
      const bCol = fixedMap(getFixedArr(b.length, 0), (_, j) => b[i][j])
      return dot(aRow, bCol)
    })
  })
}
