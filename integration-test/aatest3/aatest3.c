//===- aatest3.c --------------------------------------------- -*- C -*-===//
//
// Implements the aatest3 kernel.
//
// Target CFG shape:
//
//   bb0 -> bb1
//   bb1 -> bb2
//   bb1 -> bb4
//   bb2 -> bb3
//   bb2 -> bb1
//   bb3 -> bb4
//
// The delivered value is defined in bb0 and used in bb3.
//
// The loop bb1 <-> bb2 lies between the producer and the consumer, but it
// contains neither of them. The consumer block bb3 does not postdominate the
// loop, because bb1 can branch directly to bb4.
//
//===----------------------------------------------------------------------===//

#include "aatest3.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int aatest3(in_int_t a[N], in_int_t b[N], inout_int_t c[N]) {
  int delivered = a[0] + b[0]; // bb0: producer

  unsigned i = 0;
  int acc = 0;

  while (1) { // bb1
    if (i >= N)
      break; // bb1 -> bb4

    int x = a[i] + b[i]; // bb2

    if (x & 1) {
      c[i] = c[i] + delivered; // bb3: consumer
      acc += c[i];
      break; // bb3 -> bb4
    }

    acc += x;
    i++; // bb2 -> bb1
  }

  return acc + delivered; // bb4
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  inout_int_t c[N];

  for (int i = 0; i < N; ++i) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
    c[i] = rand() % 100;
  }

  CALL_KERNEL(aatest3, a, b, c);
  return 0;
}