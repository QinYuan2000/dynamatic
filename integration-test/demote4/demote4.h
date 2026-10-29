//===- demote4.h - Three-level demotion stress test --------------*- C -*-===//
//
// Declares the demote4 kernel.
//
//===----------------------------------------------------------------------===//

#ifndef DEMOTE4_DEMOTE4_H
#define DEMOTE4_DEMOTE4_H

// Size of every array.
#define N 128
// Number of producer firings.
#define NI 32

typedef int in_int_t;
typedef int inout_int_t;

int demote4(in_int_t a[N], in_int_t b[N], in_int_t x[N], in_int_t y[N],
            inout_int_t out[N]);

#endif // DEMOTE4_DEMOTE4_H