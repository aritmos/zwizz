## `zwizz`

A modification of Zig's default `std.HashMap` to make it into a [swiss table](https://abseil.io/about/design/swisstables).
Care has been taken to deviate as little as possible from the original implementation for easier reviewing and comparing.

---

The implementation is completed and resides in `stc/swiss_hash_map.zig`, however it is not yet labeled and released as 1.0 due to delays in testing and reviewing the correctness of the implementation.

The only metric that has been conducted so far is a modified version of the "remove one million elements in random order" unit test with 10 million elements instead. The (somewhat disappointing) results are as follows:

```
Benchmark 1: std
  Time (mean ± σ):      1.443 s ±  0.029 s    [User: 1.396 s, System: 0.043 s]
  Range (min … max):    1.399 s …  1.495 s    10 runs

Benchmark 1: zwizz
  Time (mean ± σ):      2.910 s ±  0.020 s    [User: 2.873 s, System: 0.032 s]
  Range (min … max):    2.880 s …  2.939 s    10 runs
```
