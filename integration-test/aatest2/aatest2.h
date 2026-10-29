//===- aatest2.h ------------------------------------------------ -*- C -*-===//
//
// Declares the aatest2 kernel.
//
//===----------------------------------------------------------------------===//

#ifndef AATEST2_AATEST2_H
#define AATEST2_AATEST2_H

#define N 30

typedef int in_int_t;
typedef int inout_int_t;

int aatest2(in_int_t a[N], in_int_t b[N], inout_int_t c[N]);

#endif // AATEST2_AATEST2_H