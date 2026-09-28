#include <immintrin.h>
#include "fma.h"

__m128 ts_fma_hardware_f32(__m128 a, __m128 b, __m128 c) {
    return _mm_fmadd_ps(a, b, c);
}

__m128d ts_fma_hardware_f64(__m128d a, __m128d b, __m128d c) {
    return _mm_fmadd_pd(a, b, c);
}
