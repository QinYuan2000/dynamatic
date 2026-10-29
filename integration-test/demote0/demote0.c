//===- demote0.c - Minimal demotion test -------------------------*- C -*-===//
//
// Implements the demote0 kernel.
//
// This kernel exists to test token demotion in isolation.  It is the smallest
// program I could find that needs a demotion branch and needs nothing else:
// no gamma tree, no splitting, no distribution.
//
// For every i it computes val = a[i] * b[i] and scans a window of x.  The scan
// is written as a do-while on purpose: a for-loop is rotated by the front end
// and the peeled first iteration makes clang merge the two loop exits behind a
// boolean phi, which would turn into a gamma tree and defeat the purpose.  In
// this form the front end leaves the loop alone.
//
// Resulting CFG and analysis for the delivery of val from bb0 to the store:
//
//   bb0   val = a[i]*b[i]                producer
//    |
//   bb1   c1: x[base+j] > th             header of the Case-4 loop
//    | 1 -> next (abandon this i)
//    | 0
//   bb2   c2: j+1 < jlim
//    | 0 -> bb1                          back edge, level 1
//    | 1
//   bb3   out[i] = val                   consumer
//
//   decision graph   c1 -1-> u0   c1 -0-> c2
//                    c2 -1-> u1   c2 -0-> c1   (back edge)
//
//   level-0 ROBDD : {c1}         level-1 ROBDD : {c1, c2}
//
//   * c1 is at level 1 and is read at level 0, so exactly one demotion branch
//     is needed.  c1 is the root of the level-1 ROBDD, so that branch takes
//     the level-1 MUX-tree output directly, with no suppressed select input.
//   * c2 does not appear at level 0, so it is not demoted.
//   * Both ROBDDs are already read-once, so no vertex is split and no
//     distribution branch is built.
//   * The only phis are the two induction variables, both of which become MU
//     gates.  There is no gamma tree anywhere in the kernel.
//
// So the circuit contains exactly one demotion branch and nothing else that
// could go wrong.  If this deadlocks, the demotion construction itself is at
// fault; if it passes, demotion works and the problem is elsewhere.
//
// Run-time behaviour with the data set below, over the 32 producer firings:
// the scan runs 1..5 iterations, 28 of the 32 firings run at least two, so bb1
// emits more c1 tokens than the level-0 tree can consume and the surplus has
// to be discarded by demotion.  16 firings deliver val and 16 suppress it.
//
//===----------------------------------------------------------------------===//

#include "demote0.h"
#include "dynamatic/Integration.h"

int demote0(in_int_t a[N], in_int_t b[N], in_int_t x[N], inout_int_t out[N]) {

  for (unsigned i = 0; i < NI; i++) {
    // --- bb0: producer of val ---
    int val = a[i] * b[i];
    unsigned base = i & 7;
    int th = x[base + 32] & 15;
    unsigned jlim = (unsigned)a[i]; // 3..6 at run time, unknown at compile time
    unsigned j = 0;

    // --- bb1 (c1) / bb2 (c2): the Case-4 scan loop ---
    do {
      if (x[base + j] > th)
        goto next;
      j = j + 1;
    } while (j < jlim);

    // --- bb3: consumer of val ---
    out[i] = val;

  next:;
  }

  return 0;
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  in_int_t x[N];
  inout_int_t out[N];

  for (int t = 0; t < N; ++t) {
    a[t] = 3 + (t % 4); // scan length: 3..6
    b[t] = 2 + (t % 3);
    x[t] = (t < 32) ? (((t % 19) == 7) ? 50 : 1) // scan window
                    : (3 + (t % 10));            // threshold source
    out[t] = 0;
  }

  CALL_KERNEL(demote0, a, b, x, out);
  return 0;
}