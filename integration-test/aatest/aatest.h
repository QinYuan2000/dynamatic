//===- aatest.h - Filtered accumulation kernel --------------- -*- C -*-===//
//
// Declares the aatest kernel.
//
//===----------------------------------------------------------------------===//

#ifndef AATEST_AATEST_H
#define AATEST_AATEST_H

#define N 30

typedef int in_int_t;
typedef int inout_int_t;

int aatest(in_int_t a[N], in_int_t b[N], inout_int_t c[N]);

#endif // AATEST_AATEST_H