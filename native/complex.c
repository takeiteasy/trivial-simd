#include <stddef.h>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

void ts_complex_binary_c32(unsigned op, float *out, const float *left,
                           const float *right, size_t n) {
    size_t i = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t a = vld2q_f32(left + 2 * i);
        float32x4x2_t b = vld2q_f32(right + 2 * i);
        float32x4x2_t r;
        if (op == 0) {
            r.val[0] = vaddq_f32(a.val[0], b.val[0]);
            r.val[1] = vaddq_f32(a.val[1], b.val[1]);
        } else if (op == 1) {
            r.val[0] = vsubq_f32(a.val[0], b.val[0]);
            r.val[1] = vsubq_f32(a.val[1], b.val[1]);
        } else {
            r.val[0] = vsubq_f32(vmulq_f32(a.val[0], b.val[0]),
                                   vmulq_f32(a.val[1], b.val[1]));
            r.val[1] = vaddq_f32(vmulq_f32(a.val[0], b.val[1]),
                                   vmulq_f32(a.val[1], b.val[0]));
        }
        vst2q_f32(out + 2 * i, r);
    }
#elif defined(__x86_64__) || defined(_M_X64)
    for (; i + 2 <= n; i += 2) {
        __m128 a = _mm_loadu_ps(left + 2 * i);
        __m128 b = _mm_loadu_ps(right + 2 * i);
        __m128 r;
        if (op == 0) r = _mm_add_ps(a, b);
        else if (op == 1) r = _mm_sub_ps(a, b);
        else {
            __m128 ar = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 0, 0));
            __m128 ai = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 1, 1));
            __m128 bs = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 signs = _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f);
            r = _mm_add_ps(_mm_mul_ps(ar, b), _mm_mul_ps(_mm_mul_ps(ai, bs), signs));
        }
        _mm_storeu_ps(out + 2 * i, r);
    }
#endif
    for (; i < n; ++i) {
        float ar = left[2 * i], ai = left[2 * i + 1];
        float br = right[2 * i], bi = right[2 * i + 1];
        if (op == 0) { out[2 * i] = ar + br; out[2 * i + 1] = ai + bi; }
        else if (op == 1) { out[2 * i] = ar - br; out[2 * i + 1] = ai - bi; }
        else { out[2 * i] = ar * br - ai * bi;
               out[2 * i + 1] = ar * bi + ai * br; }
    }
}

void ts_complex_binary_c64(unsigned op, double *out, const double *left,
                           const double *right, size_t n) {
    size_t i = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
    for (; i + 2 <= n; i += 2) {
        float64x2x2_t a = vld2q_f64(left + 2 * i);
        float64x2x2_t b = vld2q_f64(right + 2 * i);
        float64x2x2_t r;
        if (op == 0) {
            r.val[0] = vaddq_f64(a.val[0], b.val[0]);
            r.val[1] = vaddq_f64(a.val[1], b.val[1]);
        } else if (op == 1) {
            r.val[0] = vsubq_f64(a.val[0], b.val[0]);
            r.val[1] = vsubq_f64(a.val[1], b.val[1]);
        } else {
            r.val[0] = vsubq_f64(vmulq_f64(a.val[0], b.val[0]),
                                   vmulq_f64(a.val[1], b.val[1]));
            r.val[1] = vaddq_f64(vmulq_f64(a.val[0], b.val[1]),
                                   vmulq_f64(a.val[1], b.val[0]));
        }
        vst2q_f64(out + 2 * i, r);
    }
#elif defined(__x86_64__) || defined(_M_X64)
    for (; i < n; ++i) {
        __m128d a = _mm_loadu_pd(left + 2 * i);
        __m128d b = _mm_loadu_pd(right + 2 * i);
        __m128d r;
        if (op == 0) r = _mm_add_pd(a, b);
        else if (op == 1) r = _mm_sub_pd(a, b);
        else {
            __m128d ar = _mm_shuffle_pd(a, a, 0);
            __m128d ai = _mm_shuffle_pd(a, a, 3);
            __m128d bs = _mm_shuffle_pd(b, b, 1);
            __m128d signs = _mm_set_pd(1.0, -1.0);
            r = _mm_add_pd(_mm_mul_pd(ar, b), _mm_mul_pd(_mm_mul_pd(ai, bs), signs));
        }
        _mm_storeu_pd(out + 2 * i, r);
    }
#endif
    for (; i < n; ++i) {
        double ar = left[2 * i], ai = left[2 * i + 1];
        double br = right[2 * i], bi = right[2 * i + 1];
        if (op == 0) { out[2 * i] = ar + br; out[2 * i + 1] = ai + bi; }
        else if (op == 1) { out[2 * i] = ar - br; out[2 * i + 1] = ai - bi; }
        else { out[2 * i] = ar * br - ai * bi;
               out[2 * i + 1] = ar * bi + ai * br; }
    }
}
