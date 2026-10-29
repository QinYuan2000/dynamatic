//===- demote3.h - Two-phase guarded scan ------------------------*- C -*-===//
//
// Declares the demote3 kernel.
//
//===----------------------------------------------------------------------===//

#ifndef DEMOTE3_DEMOTE3_H
#define DEMOTE3_DEMOTE3_H

// Size of every array.
#define N 64
// Number of producer firings, i.e. iterations of the driver loop.
#define NI 32

typedef int in_int_t;
typedef int inout_int_t;

int demote3(in_int_t a[N], in_int_t b[N], in_int_t x[N], in_int_t y[N],
            inout_int_t out[N]);

#endif // DEMOTE3_DEMOTE3_H