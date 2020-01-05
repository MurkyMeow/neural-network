export interface FixedArray<T extends any, L extends number> extends Array<T> {
  0: T
  length: L
}

export type NTuple<L extends number> =
  FixedArray<number, L>

export type Matrix<R extends number, C extends number> =
  FixedArray<NTuple<C>, R>

export function fixedArray<L extends number>(size: L): FixedArray<null, L>
export function fixedArray<L extends number, R>(size: L, map: (i: number) => R): FixedArray<R, L>

export function fixedArray<L extends number, R>(size: L, map?: (i: number) => R): FixedArray<R | null, L> {
  const array = Array(size).fill(null) as FixedArray<null, L>
  return map ? fixedMap(array, (_, i) => map(i)) : array
}

export function fixedMap<T, L extends number, R>(
  arr: FixedArray<T, L>,
  fn: (element: T, index: number) => R,
): FixedArray<R, L> {
  return arr.map(fn) as FixedArray<R, L>
}

export function fixedMapPair<T, K, L extends number, R>(
  a: FixedArray<T, L>,
  b: FixedArray<K, L>,
  fn: (ax: T, bx: K, index: number) => R,
): FixedArray<R, L> {
  return fixedMap(a, (ax, i) => fn(ax, b[i], i))
}

export function fixedReducePair<T, K, L extends number, R>(
  a: FixedArray<T, L>,
  b: FixedArray<K, L>,
  initial: R,
  fn: (acc: R, ax: T, bx: K, index: number) => R,
): R {
  return a.reduce((acc, ax, i) => fn(acc, ax, b[i], i), initial)
}

export function dot<A extends number, B extends A>(
  a: FixedArray<number, A>,
  b: FixedArray<number, B>,
): number {
  return fixedReducePair(a, b, 0, (acc, ax, bx) => acc + ax * bx)
}

// this uses fixed arrays to disallow multiplying matrices
// with incompatible dimensions in *compile time*
// and to infer the resulting dimensions automatically
export function matdot<R1 extends number, C1 extends number, R2 extends C1, C2 extends number>(
  a: Matrix<R1, C1>,
  b: Matrix<R2, C2>,
): Matrix<R1, C2> {
  return fixedMap(a, aRow => {
    return fixedArray(b[0].length, i => {
      const bCol = fixedArray(b.length, j => b[i][j])
      return dot(aRow, bCol)
    })
  })
}
