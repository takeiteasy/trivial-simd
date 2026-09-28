#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
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
#define TS_F32_SQRT(a) vsqrtq_f32(a)
#define TS_F64_SQRT(a) vsqrtq_f64(a)
#define TS_F32_ABS(a) vabsq_f32(a)
#define TS_F64_ABS(a) vabsq_f64(a)
#define TS_F32_MIN(a, b) vbslq_f32(vcleq_f32(a, b), a, b)
#define TS_F64_MIN(a, b) vbslq_f64(vcleq_f64(a, b), a, b)
#define TS_F32_MAX(a, b) vbslq_f32(vcgeq_f32(a, b), a, b)
#define TS_F64_MAX(a, b) vbslq_f64(vcgeq_f64(a, b), a, b)
#define TS_F32_FMA(a, b, c) vfmaq_f32(c, a, b)
#define TS_F64_FMA(a, b, c) vfmaq_f64(c, a, b)
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
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
#define TS_F32_SQRT(a) _mm_sqrt_ps(a)
#define TS_F64_SQRT(a) _mm_sqrt_pd(a)
#define TS_F32_ABS(a) _mm_andnot_ps(_mm_set1_ps(-0.0f), a)
#define TS_F64_ABS(a) _mm_andnot_pd(_mm_set1_pd(-0.0), a)
#define TS_F32_MIN(a, b) _mm_min_ps(b, a)
#define TS_F64_MIN(a, b) _mm_min_pd(b, a)
#define TS_F32_MAX(a, b) _mm_max_ps(b, a)
#define TS_F64_MAX(a, b) _mm_max_pd(b, a)
/* TODO: SSE2 FMA is per-lane; select hardware FMA at runtime (#54). */
static __m128 ts_fma_f32(__m128 a, __m128 b, __m128 c) {
    float left[4], right[4], addend[4];
    _mm_storeu_ps(left, a); _mm_storeu_ps(right, b); _mm_storeu_ps(addend, c);
    for (size_t i = 0; i < 4; ++i) addend[i] = fmaf(left[i], right[i], addend[i]);
    return _mm_loadu_ps(addend);
}
static __m128d ts_fma_f64(__m128d a, __m128d b, __m128d c) {
    double left[2], right[2], addend[2];
    _mm_storeu_pd(left, a); _mm_storeu_pd(right, b); _mm_storeu_pd(addend, c);
    for (size_t i = 0; i < 2; ++i) addend[i] = fma(left[i], right[i], addend[i]);
    return _mm_loadu_pd(addend);
}
#define TS_F32_FMA(a, b, c) ts_fma_f32(a, b, c)
#define TS_F64_FMA(a, b, c) ts_fma_f64(a, b, c)
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
#define TS_F32_SQRT(a) ((a) == 0 ? copysignf(0.0f, a) : sqrtf(a))
#define TS_F64_SQRT(a) ((a) == 0 ? copysign(0.0, a) : sqrt(a))
#define TS_F32_ABS(a) fabsf(a)
#define TS_F64_ABS(a) fabs(a)
#define TS_F32_MIN(a, b) ((a) == (b) ? copysignf(a, a) : ((a) < (b) ? (a) : (b)))
#define TS_F64_MIN(a, b) ((a) == (b) ? copysign(a, a) : ((a) < (b) ? (a) : (b)))
#define TS_F32_MAX(a, b) ((a) == (b) ? copysignf(a, a) : ((a) > (b) ? (a) : (b)))
#define TS_F64_MAX(a, b) ((a) == (b) ? copysign(a, a) : ((a) > (b) ? (a) : (b)))
#define TS_F32_FMA(a, b, c) fmaf(a, b, c)
#define TS_F64_FMA(a, b, c) fma(a, b, c)
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

enum {
    TS_OP_COPY, TS_OP_CONSTANT, TS_OP_ADD, TS_OP_SUBTRACT,
    TS_OP_MULTIPLY, TS_OP_DIVIDE, TS_OP_NEGATE, TS_OP_SPILL, TS_OP_RELOAD,
    TS_OP_SQRT, TS_OP_ABS, TS_OP_MIN, TS_OP_MAX, TS_OP_FMA
};

#define TS_KERNEL_BLOCK 256
#define TS_KERNEL_REGISTERS 8
#define TS_KERNEL_OUTPUT 0xFF

#define TS_KERNEL_LOOP(width, load, store, vector_op, scalar_op) \
    for (; i + width <= m; i += width) \
        store(destination + i, vector_op(load(left + i), load(right + i))); \
    for (; i < m; ++i) destination[i] = left[i] scalar_op right[i]

/*
 * Code is 4-byte instructions: opcode, destination, operand a, operand b.
 * A destination is a register (0-7) or TS_KERNEL_OUTPUT. An operand below
 * TS_KERNEL_REGISTERS is a register; otherwise it is input (operand - 8).
 * TS_OP_CONSTANT reads operand a as a constant index.
 * SPILL/RELOAD use a register and a little-endian 16-bit scratch index.
 * FMA uses a register destination as its addend.
 */
/* TODO: sums materialise 256-element output blocks; reduce final VM instructions directly (#56). */
#define TS_KERNEL(suffix, type, width, load, store, zero, add, sub, mul, div, \
                  sqrtv, absv, minv, maxv, fmav, sqrts, abss, fmas) \
static int ts_kernel_run_##suffix(const uint8_t *code, size_t code_length, \
                        const type *constants, const type *const *inputs, \
                        type *out, size_t n, size_t scratch_count, int sum) { \
    if (n == 0) { if (sum) *out = 0; return 0; } \
    if (scratch_count > SIZE_MAX / TS_KERNEL_BLOCK / sizeof(type)) return -1; \
    type *scratch = scratch_count ? malloc(scratch_count * TS_KERNEL_BLOCK * sizeof(type)) : NULL; \
    if (scratch_count && !scratch) return -1; \
    type registers[TS_KERNEL_REGISTERS][TS_KERNEL_BLOCK]; \
    type block_output[TS_KERNEL_BLOCK], result = 0; \
    for (size_t base = 0; base < n; base += TS_KERNEL_BLOCK) { \
        size_t m = n - base < TS_KERNEL_BLOCK ? n - base : TS_KERNEL_BLOCK; \
        for (size_t pc = 0; pc < code_length; pc += 4) { \
            if (code[pc] == TS_OP_SPILL || code[pc] == TS_OP_RELOAD) { \
                size_t slot = (size_t)code[pc + 2] | ((size_t)code[pc + 3] << 8); \
                type *block = scratch + slot * TS_KERNEL_BLOCK; \
                type *reg = registers[code[pc + 1]]; \
                if (code[pc] == TS_OP_SPILL) memcpy(block, reg, m * sizeof(type)); \
                else memcpy(reg, block, m * sizeof(type)); \
                continue; \
            } \
            type *destination = code[pc + 1] == TS_KERNEL_OUTPUT \
                ? (sum ? block_output : out + base) : registers[code[pc + 1] & (TS_KERNEL_REGISTERS - 1)]; \
            size_t i = 0; \
            if (code[pc] == TS_OP_CONSTANT) { \
                type value = constants[code[pc + 2]]; \
                for (; i < m; ++i) destination[i] = value; \
                continue; \
            } \
            const type *left = code[pc + 2] < TS_KERNEL_REGISTERS \
                ? registers[code[pc + 2]] : inputs[code[pc + 2] - TS_KERNEL_REGISTERS] + base; \
            const type *right = code[pc + 3] < TS_KERNEL_REGISTERS \
                ? registers[code[pc + 3]] : inputs[code[pc + 3] - TS_KERNEL_REGISTERS] + base; \
            switch (code[pc]) { \
            case TS_OP_COPY: \
                for (; i < m; ++i) destination[i] = left[i]; \
                break; \
            case TS_OP_ADD: TS_KERNEL_LOOP(width, load, store, add, +); break; \
            case TS_OP_SUBTRACT: TS_KERNEL_LOOP(width, load, store, sub, -); break; \
            case TS_OP_MULTIPLY: TS_KERNEL_LOOP(width, load, store, mul, *); break; \
            case TS_OP_DIVIDE: TS_KERNEL_LOOP(width, load, store, div, /); break; \
            case TS_OP_SQRT: \
                for (size_t j = 0; j < m; ++j) { \
                    if (left[j] < 0) { free(scratch); return -2; } \
                } \
                for (; i + width <= m; i += width) store(destination + i, sqrtv(load(left + i))); \
                for (; i < m; ++i) { \
                    if (left[i] == 0) { \
                        if (destination != left) memcpy(destination + i, left + i, sizeof(type)); \
                    } \
                    else destination[i] = sqrts(left[i]); \
                } \
                break; \
            case TS_OP_ABS: \
                for (; i + width <= m; i += width) store(destination + i, absv(load(left + i))); \
                for (; i < m; ++i) destination[i] = abss(left[i]); \
                break; \
            case TS_OP_MIN: \
                for (; i + width <= m; i += width) \
                    store(destination + i, minv(load(left + i), load(right + i))); \
                for (; i < m; ++i) { \
                    if (left[i] == right[i]) { \
                        if (destination != left) memcpy(destination + i, left + i, sizeof(type)); \
                    } \
                    else destination[i] = left[i] < right[i] ? left[i] : right[i]; \
                } \
                break; \
            case TS_OP_MAX: \
                for (; i + width <= m; i += width) \
                    store(destination + i, maxv(load(left + i), load(right + i))); \
                for (; i < m; ++i) { \
                    if (left[i] == right[i]) { \
                        if (destination != left) memcpy(destination + i, left + i, sizeof(type)); \
                    } \
                    else destination[i] = left[i] > right[i] ? left[i] : right[i]; \
                } \
                break; \
            case TS_OP_FMA: \
                for (; i + width <= m; i += width) \
                    store(destination + i, fmav(load(left + i), load(right + i), load(destination + i))); \
                for (; i < m; ++i) destination[i] = fmas(left[i], right[i], destination[i]); \
                break; \
            case TS_OP_NEGATE: \
                for (; i + width <= m; i += width) \
                    store(destination + i, sub(zero(), load(left + i))); \
                for (; i < m; ++i) destination[i] = (type)0 - left[i]; \
                break; \
            } \
        } \
        if (sum) result += ts_sum_##suffix(block_output, m); \
    } \
    free(scratch); \
    if (sum) *out = result; \
    return 0; \
} \
int ts_kernel_##suffix(const uint8_t *code, size_t code_length, \
                       const type *constants, const type *const *inputs, \
                       type *out, size_t n, size_t scratch_count) { \
    return ts_kernel_run_##suffix(code, code_length, constants, inputs, out, n, scratch_count, 0); \
} \
int ts_kernel_sum_##suffix(const uint8_t *code, size_t code_length, \
                           const type *constants, const type *const *inputs, \
                           type *out, size_t n, size_t scratch_count) { \
    return ts_kernel_run_##suffix(code, code_length, constants, inputs, out, n, scratch_count, 1); \
}

TS_KERNEL(f32, float, TS_F32_WIDTH, TS_F32_LOAD, TS_F32_STORE, TS_F32_ZERO,
          TS_F32_ADD, TS_F32_SUB, TS_F32_MUL, TS_F32_DIV,
          TS_F32_SQRT, TS_F32_ABS, TS_F32_MIN, TS_F32_MAX, TS_F32_FMA, sqrtf, fabsf, fmaf)
TS_KERNEL(f64, double, TS_F64_WIDTH, TS_F64_LOAD, TS_F64_STORE, TS_F64_ZERO,
          TS_F64_ADD, TS_F64_SUB, TS_F64_MUL, TS_F64_DIV,
          TS_F64_SQRT, TS_F64_ABS, TS_F64_MIN, TS_F64_MAX, TS_F64_FMA, sqrt, fabs, fma)
