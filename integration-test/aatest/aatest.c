//===- aatest.c - Filtered accumulation kernel --------------- -*- C -*-===//
//
// Implements the aatest kernel.
//
//===----------------------------------------------------------------------===//

#include "aatest.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int aatest(in_int_t a[N], in_int_t b[N], inout_int_t c[N]) {
  int acc = 0;
  for (unsigned i = 0; i < N; i++) {
    // --- bb1: DEF ---
    int val = a[i] * b[i];

    if (a[i] > 50) {
      // --- bb2: conditional, can bypass bb4 ---
      if (val <= b[i])
        continue;
    } else {
      // --- bb3: conditional, can bypass bb4 ---
      if (val >= b[i])
        continue;
    }

    // --- bb4: COND — only reached if bb2/bb3 didn't continue ---
    acc = acc + val;
    if (acc > 100) {
      // --- bb5: USE ---
      c[i] = val;
    }
  }
  return acc;
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  inout_int_t c[N];

  for (int i = 0; i < N; ++i) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
    c[i] = 0;
  }

  CALL_KERNEL(aatest, a, b, c);
  return 0;
}