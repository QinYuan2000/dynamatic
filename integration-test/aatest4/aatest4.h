//===- aatest4.h --------------------------------------------- -*- C -*-===//
//
// Declares the aatest4 kernel.
//
//===----------------------------------------------------------------------===//

#ifndef AATEST4_AATEST4_H
#define AATEST4_AATEST4_H

#define N 30

typedef int in_int_t;
typedef int inout_int_t;

int aatest4(in_int_t a[N], in_int_t b[N], inout_int_t c[N]);

#endif // AATEST4_AATEST4_H