//===- aatest3.h --------------------------------------------- -*- C -*-===//
//
// Declares the aatest3 kernel.
//
//===----------------------------------------------------------------------===//

#ifndef AATEST3_AATEST3_H
#define AATEST3_AATEST3_H

#define N 30

typedef int in_int_t;
typedef int inout_int_t;

int aatest3(in_int_t a[N], in_int_t b[N], inout_int_t c[N]);

#endif // AATEST3_AATEST3_H