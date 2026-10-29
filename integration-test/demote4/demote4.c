//===- demote4.c - Three-level demotion stress test --------------*- C -*-===//
//
// Implements the demote4 kernel.
//
// For every i the kernel computes val = a[i] * b[i] and runs a two-phase scan
// that decides whether val is written.  The scan is built so that the delivery
// of val exercises the full cyclic-decision-graph machinery: three nesting
// levels, a split at every level, and two variables that are demoted twice
// each, with a suppressed select input on three of those four demotions.
//
// There is deliberately no gamma tree anywhere in the kernel.  Nothing defined
// inside the scan is live after it, so the only phis are the two loop
// counters, and both become MU gates.  The two latch blocks each contain a
// load, which stops the front end from folding them into their predecessors
// with a select (that folding would collapse the shared level-2 vertex and
// destroy the level-2 split).
//
//   source CFG                         decision graph
//   ----------                         --------------
//   bb0  val = a[i]*b[i]               c1 -1-> c2   c1 -0-> c3
//    |                                 c2 -1-> c8   c2 -0-> c7
//   h1  (c1) -1-> b2 (c2) -1-> e8      c3 -1-> c5   c3 -0-> c4
//    |0            |0                  c4 -1-> c6   c4 -0-> c7
//   h2  (c3) -1-> bC (c5) -1-> bD      c5 -1-> c6   c5 -0-> c9
//    |0            |0                  c6 -1-> c3   c6 -0-> c7   back edge L2
//   bB  (c4) -1-> bD (c6) -0-> b4      c7 -1-> c1   c7 -0-> c8   back edge L1
//    |0                    |1          c8 -1-> u1   c8 -0-> c9
//   b4  (c7) -1-> h1  -0-> e8          c9 -1-> u1   c9 -0-> u0
//
//   outer loop L1: header h1, back edge b4 -> h1, holds c1..c7
//   inner loop L2: header h2, back edge bD -> h2, holds c3 c4 c5 c6
//   levels: c1 c2 c7 at level 1, c3 c4 c5 c6 at level 2, c8 c9 at level 0
//
//   level-0 ROBDD : c1, c3, c5, c8, c9
//   level-1 ROBDD : c1, c2, c3, c5, c7
//   level-2 ROBDD : c3, c4, c5, c6
//
// What this exercises:
//
//   Splitting and distribution at every level.  At level 0 the tree reaches c8
//   by three paths and c9 by four; at level 1 it reaches c7 by three; at level
//   2 it reaches c6 by two.  Each of those needs split copies and a
//   distribution branch tree.
//
//   Demotion of a variable that is not a loop header.  c5 lives at level 2 and
//   appears in the level-1 and level-0 ROBDDs, so it is demoted twice.  It is
//   the root of neither: at level 2 it is reached only when c3 is true, and at
//   level 1 only when c1 is false and c3 is true.  Both of its demotion
//   branches therefore need the extra suppress on the select input, driven by
//   the reachability of c5 from the root of that level's ROBDD.
//
//   Demotion of the inner loop header.  c3 is also demoted twice; it is the
//   root at level 2 (no suppressed select) but not at level 1 (suppressed
//   select).  c1 is demoted once, as the root of level 1, so its branch takes
//   the level-1 MUX-tree output directly.  Between them the three cases are
//   all covered.
//
// Run-time behaviour with the data set built by main(), over the 32 producer
// firings.  h1 emits 1..5 c1 tokens per firing and 22 firings emit at least
// two; h2 emits up to 10 c3 tokens; bC emits up to 6 c5 tokens.  Every token
// past the first has to be discarded by demotion, so without it the level-0
// tree reads a stale condition and either the delivery decision is made on the
// wrong iteration (out[] mismatches) or the surplus stalls the feeding fork
// (deadlock).  14 firings see c1 take both values and 14 see c3 take both,
// which is what makes the two suppressed select inputs fire.  All three scan
// exits are taken (12 through b2, 10 through b4, 10 through bC), c8 and c9
// each take both outcomes, and 18 firings deliver val while 14 suppress it.
//
//===----------------------------------------------------------------------===//

#include "demote4.h"
#include "dynamatic/Integration.h"

int demote4(in_int_t a[N], in_int_t b[N], in_int_t x[N], in_int_t y[N],
            inout_int_t out[N]) {

  for (unsigned i = 0; i < NI; i++) {
    // --- bb0: producer of val ---
    int val = a[i] * b[i];
    unsigned base = i & 7;
    int th = x[base + 112] & 15;
    unsigned jlim = (unsigned)a[i]; // coarse steps, 3..6 at run time
    unsigned klim = (unsigned)b[i]; // fine steps,   2..4 at run time
    unsigned j = 0;
    unsigned k = 0;

  h1: // --- c1: pick the phase of this coarse step (header of L1) ---
    if (x[base + j] > th)
      goto b2;
    k = 0;
    goto h2;

  b2: // --- c2: coarse phase, may end the scan ---
    if (y[base + j] > th)
      goto e8;
    goto b4;

  h2: // --- c3: pick the phase of this fine step (header of L2) ---
    if (k >= klim)
      goto bB;
    goto bC;

  bB: // --- c4: late fine step ---
    if (x[base + k + 32] > th)
      goto bD;
    goto b4;

  bC: // --- c5: early fine step, may end the scan ---
    if (y[base + k + 32] < th)
      goto bD;
    goto e9;

  bD: // --- c6: end of a fine step (latch of L2) ---
    k = k + 1 + (unsigned)(x[base + k + 64] & 1);
    if (k >= klim + 4)
      goto b4;
    goto h2;

  b4: // --- c7: end of a coarse step (latch of L1) ---
    j = j + 1 + (unsigned)(y[base + j + 64] & 1);
    if (j >= jlim + 2)
      goto e8;
    goto h1;

  e8: // --- c8: first post-scan check ---
    if (x[base + 96] > th)
      goto cons;
    goto e9;

  e9: // --- c9: second post-scan check ---
    if (y[base + 96] < th)
      goto cons;
    goto next;

  cons: // --- consumer of val ---
    out[i] = val;

  next:;
  }

  return 0;
}

int main(void) {
  in_int_t a[N];
  in_int_t b[N];
  in_int_t x[N];
  in_int_t y[N];
  inout_int_t out[N];

  // Each array is split into regions, one per decision block, so that the
  // branch probabilities can be set independently.  The regions are sized so
  // that no index can leave the array: base <= 7, j <= 9, k <= 9.
  for (int t = 0; t < N; ++t) {
    a[t] = 3 + (t % 4); // coarse scan length
    b[t] = 2 + (t % 3); // fine scan length
    if (t < 32)
      x[t] = (t & 1) ? 40 : 0; // c1, alternates within one scan
    else if (t < 64)
      x[t] = (t & 2) ? 50 : 1; // c4
    else if (t < 96)
      x[t] = (t * 5) % 7; // fine step size
    else if (t < 112)
      x[t] = (t * 3) % 19; // c8
    else
      x[t] = 3 + (t % 10); // threshold source, th in 3..12
    if (t < 32)
      y[t] = ((t % 5) == 3) ? 50 : 1; // c2, rarely ends the scan
    else if (t < 64)
      y[t] = ((t % 11) == 4) ? 50 : 1; // c5, rarely ends the scan
    else if (t < 96)
      y[t] = (t * 5) % 7; // coarse step size
    else
      y[t] = (t * 7) % 25; // c9
    out[t] = 0;
  }

  CALL_KERNEL(demote4, a, b, x, y, out);
  return 0;
}