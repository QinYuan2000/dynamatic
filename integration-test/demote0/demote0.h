//===- demote0.h - Minimal demotion test -------------------------*- C -*-===//
//
// Declares the demote0 kernel.
//
//===----------------------------------------------------------------------===//

#ifndef DEMOTE0_DEMOTE0_H
#define DEMOTE0_DEMOTE0_H

// Size of every array.
#define N 64
// Number of producer firings.
#define NI 32

typedef int in_int_t;
typedef int inout_int_t;

int demote0(in_int_t a[N], in_int_t b[N], in_int_t x[N], inout_int_t out[N]);

#endif // DEMOTE0_DEMOTE0_H