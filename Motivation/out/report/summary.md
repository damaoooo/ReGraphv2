# simple_add.c Motivation Summary

## Source

- `Motivation/simple_add.c`
- Focus function: `example(int a, int b)`

## Setup

- chosen binaries:
  - `x86_64 -O1`: `Motivation/out/obj/simple_add_x86_O1.o`
  - `aarch64 -O0`: `Motivation/out/obj/simple_add_arm64_O0.o`
- inline disabled during compilation with:
  - `-fno-inline -fno-inline-functions`
- lifted with:
  - `Scripts/ida2llvm.py`
- re-optimized with:
  - `opt -S -O3 -force-attribute=noinline`

## C-like pseudocode

### Source-level intent

```c
long example(int a, int b) {
    for (int i = 0; i < 1000; i++) a += i;
    for (int i = 0; i < 5; i++) b ^= i;
    return a + b;
}
```

### ARM -O0

`ARM -O0` keeps the two loops almost directly:

```c
long example(int a, int b) {
    for (int i = 0; i < 1000; i++) {
        a += i;
    }
    for (int j = 0; j < 5; j++) {
        b ^= j;
    }
    return a + b;
}
```

### x86 -O1

`x86 -O1` folds the first loop into a constant, but still keeps the small `xor` loop:

```c
long example(int a, int b) {
    for (int i = 0; i < 5; i++) {
        b ^= i;
    }
    return a + b + 499500;
}
```

### Lifted IR + `opt -O3`

After lifting and `O3`, both versions converge to essentially the same straight-line pseudocode:

```c
long example(int a, int b) {
    return a + (b ^ 4) + 499500;
}
```

## Key effect

- before lifting:
  - `ARM -O0` has two explicit loops
  - `x86 -O1` has one remaining loop and one folded constant computation
- after lifting + `opt -O3`:
  - both `example()` IRs reduce to the same high-level arithmetic form

## Artifact paths

- asm:
  - `Motivation/out/asm/simple_add_x86_O1.objdump.txt`
  - `Motivation/out/asm/simple_add_arm64_O0.objdump.txt`
- lifted IR:
  - `Motivation/out/ir/simple_add_x86_O1.ll`
  - `Motivation/out/ir/simple_add_arm64_O0.ll`
- optimized IR:
  - `Motivation/out/ir_o3/simple_add_x86_O1.O3.noinline.ll`
  - `Motivation/out/ir_o3/simple_add_arm64_O0.O3.noinline.ll`
