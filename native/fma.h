#ifndef TS_FMA_H
#define TS_FMA_H

int ts_fma_supported(void);
float ts_fma_scalar_f32(float a, float b, float c);
double ts_fma_scalar_f64(double a, double b, double c);

#if defined(TS_HAVE_HARDWARE_FMA) && (defined(__x86_64__) || defined(_M_X64))
#include <emmintrin.h>
__m128 ts_fma_hardware_f32(__m128 a, __m128 b, __m128 c);
__m128d ts_fma_hardware_f64(__m128d a, __m128d b, __m128d c);
#endif

#endif
