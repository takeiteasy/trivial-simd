#include <stddef.h>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define TS_F32_WIDTH 4
#define TS_F64_WIDTH 2
#define TS_F32_VECTOR float32x4_t
#define TS_F64_VECTOR float64x2_t
#define TS_F32_LOAD(p) vld1q_f32(p)
#define TS_F64_LOAD(p) vld1q_f64(p)
#define TS_F32_STORE(p, v) vst1q_f32(p, v)
#define TS_F64_STORE(p, v) vst1q_f64(p, v)
#define TS_F32_ZERO() vdupq_n_f32(0.0f)
#define TS_F64_ZERO() vdupq_n_f64(0.0)
#define TS_F32_ADD(a, b) vaddq_f32(a, b)
#define TS_F64_ADD(a, b) vaddq_f64(a, b)
#define TS_F32_SUB(a, b) vsubq_f32(a, b)
#define TS_F64_SUB(a, b) vsubq_f64(a, b)
#define TS_F32_MUL(a, b) vmulq_f32(a, b)
#define TS_F64_MUL(a, b) vmulq_f64(a, b)
#define TS_F32_DIV(a, b) vdivq_f32(a, b)
#define TS_F64_DIV(a, b) vdivq_f64(a, b)
#elif defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#include <xmmintrin.h>
#define TS_F32_WIDTH 4
#define TS_F64_WIDTH 2
#define TS_F32_VECTOR __m128
#define TS_F64_VECTOR __m128d
#define TS_F32_LOAD(p) _mm_loadu_ps(p)
#define TS_F64_LOAD(p) _mm_loadu_pd(p)
#define TS_F32_STORE(p, v) _mm_storeu_ps(p, v)
#define TS_F64_STORE(p, v) _mm_storeu_pd(p, v)
#define TS_F32_ZERO() _mm_setzero_ps()
#define TS_F64_ZERO() _mm_setzero_pd()
#define TS_F32_ADD(a, b) _mm_add_ps(a, b)
#define TS_F64_ADD(a, b) _mm_add_pd(a, b)
#define TS_F32_SUB(a, b) _mm_sub_ps(a, b)
#define TS_F64_SUB(a, b) _mm_sub_pd(a, b)
#define TS_F32_MUL(a, b) _mm_mul_ps(a, b)
#define TS_F64_MUL(a, b) _mm_mul_pd(a, b)
#define TS_F32_DIV(a, b) _mm_div_ps(a, b)
#define TS_F64_DIV(a, b) _mm_div_pd(a, b)
#else
#define TS_F32_WIDTH 1
#define TS_F64_WIDTH 1
#define TS_F32_VECTOR float
#define TS_F64_VECTOR double
#define TS_F32_LOAD(p) (*(p))
#define TS_F64_LOAD(p) (*(p))
#define TS_F32_STORE(p, v) (*(p) = (v))
#define TS_F64_STORE(p, v) (*(p) = (v))
#define TS_F32_ZERO() 0.0f
#define TS_F64_ZERO() 0.0
#define TS_F32_ADD(a, b) ((a) + (b))
#define TS_F64_ADD(a, b) ((a) + (b))
#define TS_F32_SUB(a, b) ((a) - (b))
#define TS_F64_SUB(a, b) ((a) - (b))
#define TS_F32_MUL(a, b) ((a) * (b))
#define TS_F64_MUL(a, b) ((a) * (b))
#define TS_F32_DIV(a, b) ((a) / (b))
#define TS_F64_DIV(a, b) ((a) / (b))
#endif

#define TS_BINARY(name, suffix, type, width, load, store, vector_op, scalar_op) \
void ts_##name##_##suffix(type *out, const type *left, const type *right, size_t n) { \
    size_t i = 0; \
    for (; i + width <= n; i += width) \
        store(out + i, vector_op(load(left + i), load(right + i))); \
    for (; i < n; ++i) out[i] = left[i] scalar_op right[i]; \
}

TS_BINARY(add, f32, float, TS_F32_WIDTH, TS_F32_LOAD, TS_F32_STORE, TS_F32_ADD, +)
TS_BINARY(subtract, f32, float, TS_F32_WIDTH, TS_F32_LOAD, TS_F32_STORE, TS_F32_SUB, -)
TS_BINARY(multiply, f32, float, TS_F32_WIDTH, TS_F32_LOAD, TS_F32_STORE, TS_F32_MUL, *)
TS_BINARY(divide, f32, float, TS_F32_WIDTH, TS_F32_LOAD, TS_F32_STORE, TS_F32_DIV, /)
TS_BINARY(add, f64, double, TS_F64_WIDTH, TS_F64_LOAD, TS_F64_STORE, TS_F64_ADD, +)
TS_BINARY(subtract, f64, double, TS_F64_WIDTH, TS_F64_LOAD, TS_F64_STORE, TS_F64_SUB, -)
TS_BINARY(multiply, f64, double, TS_F64_WIDTH, TS_F64_LOAD, TS_F64_STORE, TS_F64_MUL, *)
TS_BINARY(divide, f64, double, TS_F64_WIDTH, TS_F64_LOAD, TS_F64_STORE, TS_F64_DIV, /)

#define TS_REDUCTIONS(suffix, type, width, vector_type, load, store, zero, add, mul) \
type ts_sum_##suffix(const type *input, size_t n) { \
    size_t i = 0; \
    vector_type lanes = zero(); \
    for (; i + width <= n; i += width) lanes = add(lanes, load(input + i)); \
    type values[width]; \
    store(values, lanes); \
    type result = 0; \
    for (size_t lane = 0; lane < width; ++lane) result += values[lane]; \
    for (; i < n; ++i) result += input[i]; \
    return result; \
} \
type ts_dot_##suffix(const type *left, const type *right, size_t n) { \
    size_t i = 0; \
    vector_type lanes = zero(); \
    for (; i + width <= n; i += width) \
        lanes = add(lanes, mul(load(left + i), load(right + i))); \
    type values[width]; \
    store(values, lanes); \
    type result = 0; \
    for (size_t lane = 0; lane < width; ++lane) result += values[lane]; \
    for (; i < n; ++i) result += left[i] * right[i]; \
    return result; \
}

TS_REDUCTIONS(f32, float, TS_F32_WIDTH, TS_F32_VECTOR, TS_F32_LOAD, TS_F32_STORE,
              TS_F32_ZERO, TS_F32_ADD, TS_F32_MUL)
TS_REDUCTIONS(f64, double, TS_F64_WIDTH, TS_F64_VECTOR, TS_F64_LOAD, TS_F64_STORE,
              TS_F64_ZERO, TS_F64_ADD, TS_F64_MUL)
