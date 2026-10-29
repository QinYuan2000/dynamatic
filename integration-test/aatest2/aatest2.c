//===- aatest2.c ------------------------------------------------ -*- C -*-===//
//
// Implements the aatest2 kernel.
//
// Desired CFG shape:
//
//   bb1: DEF (val = a[i] * b[i])
//      |
//   split on a[i] > 50
//    /            \
//  bb2            bb3
//   |              |
//  fork           fork
//   |              |
// continue       continue
//   \              /
//        bb4: COND (val > 100)
//         /              \
//   bb_use              bb5
//   (USE val)            |
//      \                /
//            bb5
//
// Key properties:
//  - val is defined in bb1
//  - val is used only in bb_use
//  - bb4 decides whether bb_use is reached
//  - both incoming paths to bb4 contain an earlier fork that may bypass bb4
//  - no phi is needed
//
//===----------------------------------------------------------------------===//

#include "aatest2.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

int aatest2(in_int_t a[N], in_int_t b[N], inout_int_t c[N]) {
  for (unsigned i = 0; i < N; i++) {
    int val = a[i] * b[i];

    if (a[i] > 50) {
      if (val <= b[i])
        continue;
    } else {
      if (val >= b[i])
        continue;
    }

    if (val > 100) {
      for (unsigned j = 0; j < N; j++) {
        c[j] = c[j] + val;
      }
    }

    c[i] = c[i] + 1;
  }
  return 0;
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

  CALL_KERNEL(aatest2, a, b, c);
  return 0;
}
