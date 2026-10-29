//===- aatest4.c --------------------------------------------- -*- C -*-===//
//
// Implements the aatest4 kernel.
//
// This benchmark creates a two-level nested-loop control structure between
// the producer and the consumer.
//
// The delivered value is defined before both loops and used only in the USE
// block. The USE block does not postdominate either loop, because both the
// outer loop and the inner loop have exits that can bypass it.
//
// Intended high-level CFG shape:
//
//   bb0:
//     delivered = a[0] + b[0]
//     goto outer_header
//
//   outer_header:
//     if (i >= N) goto exit              // bypass USE
//     else goto inner_preheader
//
//   inner_header:
//     if (j >= N) goto outer_latch       // bypass USE
//     else goto inner_body
//
//   inner_body:
//     if (hit) goto use_block            // reach USE
//     else goto inner_latch
//
//   inner_latch:
//     j++
//     goto inner_header
//
//   use_block:
//     c[i] = c[i] + delivered            // consumer
//     acc += c[i]
//     goto exit
//
//   outer_latch:
//     i++
//     goto outer_header
//
//   exit:
//     return acc + delivered
//
//===----------------------------------------------------------------------===//

#include "aatest4.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int aatest4(in_int_t a[N], in_int_t b[N], inout_int_t c[N]) {
  int delivered = a[0] + b[0]; // producer before both loops

  int acc = 0;

  for (unsigned i = 0; i < N; i++) {
    unsigned j = 0;

    while (j < N) {
      int x = a[i] + b[j] + c[j];

      if ((x & 7) == 3) {
        c[i] = c[i] + delivered; // consumer
        acc += c[i];
        return acc + delivered;
      }

      acc += x;
      j++;
    }

    acc += a[i];
  }

  return acc + delivered;
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

  CALL_KERNEL(aatest4, a, b, c);
  return 0;
}