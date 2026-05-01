/*
 * Minimal BLAS declarations for compiling the original TTeMPS MATLAB MEX
 * sources with GNU Octave on systems that do not ship MATLAB's blas.h.
 */
#ifndef TTEMPS_OCTAVE_BLAS_H
#define TTEMPS_OCTAVE_BLAS_H

typedef int blas_int;

#define daxpy daxpy_
#define dcopy dcopy_
#define dgemv dgemv_
#define dger dger_

extern void daxpy_(const blas_int *n, const double *alpha, const double *x,
                   const blas_int *incx, double *y, const blas_int *incy);

extern void dcopy_(const blas_int *n, const double *x, const blas_int *incx,
                   double *y, const blas_int *incy);

extern void dgemv_(const char *trans, const blas_int *m, const blas_int *n,
                   const double *alpha, const double *a, const blas_int *lda,
                   const double *x, const blas_int *incx,
                   const double *beta, double *y, const blas_int *incy);

extern void dger_(const blas_int *m, const blas_int *n, const double *alpha,
                  const double *x, const blas_int *incx,
                  const double *y, const blas_int *incy,
                  double *a, const blas_int *lda);

#endif
