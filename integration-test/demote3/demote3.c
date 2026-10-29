//===- demote3.c - Two-phase guarded scan ------------------------*- C -*-===//
//
// Implements the demote3 kernel.
//
// For every i the kernel computes val = a[i] * b[i] and then runs a two-phase
// scan that decides whether val is kept.  Each coarse step either takes the
// coarse phase (bb2, which accumulates into acc and may stop the scan) or the
// fine phase (the inner loop bb3/bb5, which may also stop the scan).  The scan
// can therefore leave through three different exits, and those exits land on
// two different post-scan checks, bb6 and bb7.
//
// The kernel exists for its control flow.  For the delivery of val from bb0 to
// bb8, both scan loops are Case-4 loops (producer before them, consumer after
// them, consumer post-dominating neither header), so the decision graph is
// cyclic with two nesting levels:
//
//   source CFG                        decision graph
//   ----------                        --------------
//   bb0  producer of val              c1 -1-> c2    c1 -0-> c3
//    |                                c2 -1-> c6    c2 -0-> c4
//   bb1 (c1) -1-> bb2 (c2) -1-> bb6   c3 -1-> c4    c3 -0-> c5
//    |0            |0                 c4 -1-> c6    c4 -0-> c1   back edge, L1
//   bb3 (c3) -1-> bb4 (c4) -1-> bb6   c5 -1-> c7    c5 -0-> c3   back edge, L2
//    |0            |0                 c6 -1-> u1    c6 -0-> c7
//   bb5 (c5) -1-> bb7             |   c7 -1-> u1    c7 -0-> u0
//    |0                           |
//    +--> bb3 (back edge, L2)     +--> bb1 (back edge, L1)
//
//   outer loop L1: header bb1, back edge bb4 -> bb1, holds c1 c2 c3 c4 c5
//   inner loop L2: header bb3, back edge bb5 -> bb3, holds c3 c5
//   levels: c1 c2 c4 at level 1, c3 c5 at level 2, c6 c7 at level 0
//
//   level-0 ROBDD: c1, c3, c6, c7     level-1 ROBDD: c1, c2, c3, c4
//   level-2 ROBDD: c3, c5
//
// What this exercises:
//
//   * c1 is at level 1 and is read at level 0, so it is demoted once.  It is
//     the root of the level-1 ROBDD, so its demotion branch takes the level-1
//     MUX-tree output directly.
//   * c3 is at level 2 and is read at level 1 and at level 0, so it is demoted
//     twice.  It is the root of the level-2 ROBDD, but NOT of the level-1
//     ROBDD: it is reached only on the iterations where c1 is false.  Its
//     second demotion branch therefore needs the extra suppress on its select
//     input, driven by the reachability of c3 from c1.  This is the case that
//     the simpler kernels never reach.
//   * The level-0 MUX-tree is not read-once: c6 is reachable from the root by
//     two paths and c7 by three, so the level-0 ROBDD has to be split and c6
//     and c7 need distribution branch trees.  The level-1 ROBDD needs a split
//     for c4 as well.
//
// Run-time behaviour with the data set built by main(), measured over the 32
// producer firings: bb1 emits 1..5 c1 tokens per firing and 28 of the 32
// firings emit at least two; bb3 emits 1..15 c3 tokens per firing; 28 firings
// see c1 take both values, which is what drives the suppressed select input of
// the c3 demotion; all three scan exits are taken (11 through bb2, 12 through
// bb4, 9 through bb5); c6 and c7 each take both outcomes; and 23 firings
// deliver val while 9 suppress it.
//
//===----------------------------------------------------------------------===//

#include "demote3.h"
#include "dynamatic/Integration.h"

int demote3(in_int_t a[N], in_int_t b[N], in_int_t x[N], in_int_t y[N],
            inout_int_t out[N]) {
  int total = 0;

  for (unsigned i = 0; i < NI; i++) {
    // --- bb0: producer of val, dominates the whole scan ---
    int val = a[i] * b[i];
    unsigned base = i & 7;
    int th = x[base + 48] & 15;
    unsigned jlim = (unsigned)a[i]; // coarse steps, 3..6 at run time
    unsigned klim = (unsigned)b[i]; // fine steps,   2..4 at run time
    int acc = 0;
    unsigned j = 0;
    unsigned k = 0;

  bb1: // --- c1: choose the phase of this coarse step (header of L1) ---
    if (x[base + j] > th)
      goto bb2;
    k = 0;
    goto bb3;

  bb2: // --- c2: coarse phase, may stop the scan ---
    acc = acc + x[base + j + 16];
    if (acc > th * 16)
      goto bb6;
    goto bb4;

  bb3: // --- c3: fine phase (header of L2) ---
    if (k >= klim)
      goto bb4;
    goto bb5;

  bb5: // --- c5: fine phase body, may stop the scan ---
    if (y[base + k] < th)
      goto bb7;
    k = k + 1;
    goto bb3;

  bb4: // --- c4: end of a coarse step (latch of L1) ---
    acc = acc + x[base + j + 32];
    j = j + 1;
    if (j >= jlim)
      goto bb6;
    goto bb1;

  bb6: // --- c6: first post-scan check ---
    if (y[base + 16] > th)
      goto bb8;
    goto bb7;

  bb7: // --- c7: second post-scan check ---
    if (y[base + 32] < acc)
      goto bb8;
    goto next;

  bb8: // --- consumer of val ---
    out[i] = val;
    total = total + val;

  next:;
  }

  return total;
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  in_int_t x[N];
  in_int_t y[N];
  inout_int_t out[N];

  // Each array is split into regions, one per decision block, so that the
  // branch probabilities can be set independently.  See the note above for the
  // coverage this data set achieves.
  for (int t = 0; t < N; ++t) {
    a[t] = 3 + (t % 4); // coarse scan length: 3..6
    b[t] = 2 + (t % 3); // fine scan length:   2..4
    if (t < 16)
      x[t] = (t & 1) ? 40 : 0; // c1: alternates within one scan
    else if (t < 32)
      x[t] = ((t * 7) % 5) * 15; // coarse-phase accumulation
    else if (t < 48)
      x[t] = ((t * 3) % 4) * 10; // latch accumulation
    else
      x[t] = 3 + (t % 10); // threshold source, th in 3..12
    if (t < 16)
      y[t] = ((t % 9) == 1) ? 2 : 25; // c5: rarely stops the fine phase
    else if (t < 32)
      y[t] = (t * 7) % 13; // c6 source
    else
      y[t] = (t * 11) % 100; // c7 source
    out[t] = 0;
  }

  CALL_KERNEL(demote3, a, b, x, y, out);
  return 0;
}