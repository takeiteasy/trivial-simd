#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "fma.h"

#pragma STDC FP_CONTRACT OFF

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
#define TS_F32_SPLAT(a) vdupq_n_f32(a)
#define TS_F64_SPLAT(a) vdupq_n_f64(a)
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
static __m128 ts_fma_f32(__m128 a, __m128 b, __m128 c) {
#ifdef TS_HAVE_HARDWARE_FMA
    if (ts_fma_supported()) return ts_fma_hardware_f32(a, b, c);
#endif
    float left[4], right[4], addend[4];
    _mm_storeu_ps(left, a); _mm_storeu_ps(right, b); _mm_storeu_ps(addend, c);
    for (size_t i = 0; i < 4; ++i) addend[i] = fmaf(left[i], right[i], addend[i]);
    return _mm_loadu_ps(addend);
}
static __m128d ts_fma_f64(__m128d a, __m128d b, __m128d c) {
#ifdef TS_HAVE_HARDWARE_FMA
    if (ts_fma_supported()) return ts_fma_hardware_f64(a, b, c);
#endif
    double left[2], right[2], addend[2];
    _mm_storeu_pd(left, a); _mm_storeu_pd(right, b); _mm_storeu_pd(addend, c);
    for (size_t i = 0; i < 2; ++i) addend[i] = fma(left[i], right[i], addend[i]);
    return _mm_loadu_pd(addend);
}
#define TS_F32_FMA(a, b, c) ts_fma_f32(a, b, c)
#define TS_F64_FMA(a, b, c) ts_fma_f64(a, b, c)
#define TS_F32_SPLAT(a) _mm_set1_ps(a)
#define TS_F64_SPLAT(a) _mm_set1_pd(a)
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
#define TS_F32_SPLAT(a) (a)
#define TS_F64_SPLAT(a) (a)
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

#define TS_BULK_FLOAT(suffix, type, width, load, store, splat, add, sub, mul, div, fused, scalar_fma) \
void ts_bulk_binary_##suffix(unsigned op, type *out, const type *left, const type *right, \
                             type left_scalar, type right_scalar, size_t n) { \
    size_t i = 0; \
    for (; i + width <= n; i += width) { \
        const type *a = left ? left + i : NULL, *b = right ? right + i : NULL; \
        if (op == 0) store(out + i, add(a ? load(a) : splat(left_scalar), b ? load(b) : splat(right_scalar))); \
        else if (op == 1) store(out + i, sub(a ? load(a) : splat(left_scalar), b ? load(b) : splat(right_scalar))); \
        else if (op == 2) store(out + i, mul(a ? load(a) : splat(left_scalar), b ? load(b) : splat(right_scalar))); \
        else store(out + i, div(a ? load(a) : splat(left_scalar), b ? load(b) : splat(right_scalar))); \
    } \
    for (; i < n; ++i) { \
        type a = left ? left[i] : left_scalar, b = right ? right[i] : right_scalar; \
        if (op == 0) out[i] = a + b; \
        else if (op == 1) out[i] = a - b; \
        else if (op == 2) out[i] = a * b; \
        else out[i] = a / b; \
    } \
} \
void ts_bulk_axpy_##suffix(type *y, type a, const type *x, size_t n) { \
    size_t i = 0; \
    for (; i + width <= n; i += width) store(y + i, add(mul(splat(a), load(x + i)), load(y + i))); \
    for (; i < n; ++i) y[i] = a * x[i] + y[i]; \
} \
void ts_bulk_fma_##suffix(type *out, const type *x, const type *y, const type *z, \
                          type xs, type ys, type zs, size_t n) { \
    size_t i = 0; \
    for (; i + width <= n; i += width) \
        store(out + i, fused(x ? load(x + i) : splat(xs), \
                             y ? load(y + i) : splat(ys), z ? load(z + i) : splat(zs))); \
    for (; i < n; ++i) out[i] = scalar_fma(x ? x[i] : xs, y ? y[i] : ys, z ? z[i] : zs); \
}

TS_BULK_FLOAT(f32, float, TS_F32_WIDTH, TS_F32_LOAD, TS_F32_STORE, TS_F32_SPLAT,
              TS_F32_ADD, TS_F32_SUB, TS_F32_MUL, TS_F32_DIV, TS_F32_FMA, fmaf)
TS_BULK_FLOAT(f64, double, TS_F64_WIDTH, TS_F64_LOAD, TS_F64_STORE, TS_F64_SPLAT,
              TS_F64_ADD, TS_F64_SUB, TS_F64_MUL, TS_F64_DIV, TS_F64_FMA, fma)

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

#define TS_ABS_SCALAR(value) ((value) < 0 ? -(value) : (value))
#define TS_ASUM(suffix, type, width, vector_type, load, store, zero, add, abs) \
type ts_asum_##suffix(const type *input, size_t n) { \
    size_t i = 0; \
    vector_type lanes = zero(); \
    for (; i + width <= n; i += width) lanes = add(lanes, abs(load(input + i))); \
    type values[width]; \
    store(values, lanes); \
    type result = 0; \
    for (size_t lane = 0; lane < width; ++lane) result += values[lane]; \
    for (; i < n; ++i) result += TS_ABS_SCALAR(input[i]); \
    return result; \
}

TS_ASUM(f32, float, TS_F32_WIDTH, TS_F32_VECTOR, TS_F32_LOAD, TS_F32_STORE,
        TS_F32_ZERO, TS_F32_ADD, TS_F32_ABS)
TS_ASUM(f64, double, TS_F64_WIDTH, TS_F64_VECTOR, TS_F64_LOAD, TS_F64_STORE,
        TS_F64_ZERO, TS_F64_ADD, TS_F64_ABS)

/* Widened f32 reductions use four double lanes on every target so results match across ISAs. */
#define TS_WIDE_LANES 4
#define TS_WIDE_REDUCTION(name, args, term) \
double ts_##name##_acc_f32 args { \
    double lanes[TS_WIDE_LANES] = {0}, result = 0; \
    size_t i = 0; \
    for (; i + TS_WIDE_LANES <= n; i += TS_WIDE_LANES) \
        for (size_t lane = 0; lane < TS_WIDE_LANES; ++lane) { \
            size_t k = i + lane; \
            lanes[lane] += term; \
        } \
    for (size_t lane = 0; lane < TS_WIDE_LANES; ++lane) result += lanes[lane]; \
    for (; i < n; ++i) { size_t k = i; result += term; } \
    return result; \
}

TS_WIDE_REDUCTION(sum, (const float *a, size_t n), (double)a[k])
TS_WIDE_REDUCTION(dot, (const float *a, const float *b, size_t n), (double)a[k] * (double)b[k])
TS_WIDE_REDUCTION(asum, (const float *a, size_t n), fabs((double)a[k]))

/* First extreme element wins; blocks with no improvement skip the scalar scan. */
#define TS_ARG_BLOCK 32
#define TS_ARG_REDUCTION(name, suffix, type, better) \
size_t ts_##name##_##suffix(const type *input, size_t n) { \
    if (!n) return 0; \
    type best = input[0]; \
    size_t index = 0, i = 1; \
    for (; i + TS_ARG_BLOCK <= n; i += TS_ARG_BLOCK) { \
        type block = input[i]; \
        for (size_t j = 1; j < TS_ARG_BLOCK; ++j) \
            if (better(input[i + j], block)) block = input[i + j]; \
        if (!better(block, best)) continue; \
        for (size_t j = 0; j < TS_ARG_BLOCK; ++j) \
            if (better(input[i + j], best)) { best = input[i + j]; index = i + j; } \
    } \
    for (; i < n; ++i) \
        if (better(input[i], best)) { best = input[i]; index = i; } \
    return index; \
}

#define TS_LESS(a, b) ((a) < (b))
#define TS_GREATER(a, b) ((a) > (b))
TS_ARG_REDUCTION(argmin, f32, float, TS_LESS)
TS_ARG_REDUCTION(argmin, f64, double, TS_LESS)
TS_ARG_REDUCTION(argmax, f32, float, TS_GREATER)
TS_ARG_REDUCTION(argmax, f64, double, TS_GREATER)

/* Internal BLAS i?amax: first index of the largest magnitude. */
#define TS_ABS_GREATER(a, b) (TS_ABS_SCALAR(a) > TS_ABS_SCALAR(b))
TS_ARG_REDUCTION(iamax, f32, float, TS_ABS_GREATER)
TS_ARG_REDUCTION(iamax, f64, double, TS_ABS_GREATER)

enum {
    TS_OP_COPY, TS_OP_CONSTANT, TS_OP_ADD, TS_OP_SUBTRACT,
    TS_OP_MULTIPLY, TS_OP_DIVIDE, TS_OP_NEGATE, TS_OP_SPILL, TS_OP_RELOAD,
    TS_OP_SQRT, TS_OP_ABS, TS_OP_MIN, TS_OP_MAX, TS_OP_FMA,
    TS_OP_EQ, TS_OP_NE, TS_OP_LT, TS_OP_LE, TS_OP_GT, TS_OP_GE, TS_OP_SELECT
};

#define TS_KERNEL_BLOCK 256
#define TS_KERNEL_REGISTERS 8
#define TS_KERNEL_OUTPUT 0xFF

#define TS_KERNEL_LOOP(width, load, store, vector_op, scalar_op) \
    for (; i + width <= m; i += width) \
        store(destination + i, vector_op(load(left + i), load(right + i))); \
    for (; i < m; ++i) destination[i] = left[i] scalar_op right[i]

/* Preserve ts_sum's lane and tail order without materialising an output block. */
#define TS_KERNEL_REDUCE(type, width, store, add, vector_value, scalar_value) \
    do { \
        for (; i + width <= m; i += width) lanes = add(lanes, vector_value); \
        type values[width]; \
        store(values, lanes); \
        for (size_t lane = 0; lane < width; ++lane) result += values[lane]; \
        for (; i < m; ++i) result += scalar_value; \
    } while (0)

/*
 * Code is 4-byte instructions: opcode, destination, operand a, operand b.
 * A destination is a register (0-7) or TS_KERNEL_OUTPUT. An operand below
 * TS_KERNEL_REGISTERS is a register; otherwise it is input (operand - 8).
 * TS_OP_CONSTANT reads operand a as a constant index.
 * SPILL/RELOAD use a register and a little-endian 16-bit scratch index.
 * FMA uses a register destination as its addend.
 */
#define TS_KERNEL_RUN(suffix, mode, sum, type, width, load, store, zero, add, sub, mul, div, \
                      sqrtv, absv, minv, maxv, fmav, sqrts, abss, fmas) \
static int ts_kernel_run_##mode##_##suffix(const uint8_t *code, size_t code_length, \
                        const type *constants, const type *const *inputs, \
                        type *out, size_t n, size_t scratch_count, type *scratch) { \
    if (n == 0) { if (sum) *out = 0; return 0; } \
    if (scratch_count > SIZE_MAX / TS_KERNEL_BLOCK / sizeof(type)) return -1; \
    type registers[TS_KERNEL_REGISTERS][TS_KERNEL_BLOCK]; \
    type result = 0; \
    for (size_t base = 0; base < n; base += TS_KERNEL_BLOCK) { \
        size_t m = n - base < TS_KERNEL_BLOCK ? n - base : TS_KERNEL_BLOCK; \
        type block_result = 0; \
        for (size_t pc = 0; pc < code_length; pc += 4) { \
            if (code[pc] == TS_OP_SPILL || code[pc] == TS_OP_RELOAD) { \
                size_t slot = (size_t)code[pc + 2] | ((size_t)code[pc + 3] << 8); \
                type *block = scratch + slot * TS_KERNEL_BLOCK; \
                type *reg = registers[code[pc + 1]]; \
                if (code[pc] == TS_OP_SPILL) memcpy(block, reg, m * sizeof(type)); \
                else memcpy(reg, block, m * sizeof(type)); \
                continue; \
            } \
            int reduce = sum && code[pc + 1] == TS_KERNEL_OUTPUT; \
            type *destination = code[pc + 1] == TS_KERNEL_OUTPUT \
                ? out + (sum ? 0 : base) : registers[code[pc + 1] & (TS_KERNEL_REGISTERS - 1)]; \
            size_t i = 0; \
            if (code[pc] == TS_OP_CONSTANT) { \
                type value = constants[code[pc + 2]]; \
                if (reduce) { \
                    int status = ts_kernel_reduce_##suffix(TS_OP_CONSTANT, NULL, NULL, value, m, &block_result); \
                    if (status) return status; \
                } else for (; i < m; ++i) destination[i] = value; \
                continue; \
            } \
            const type *left = code[pc + 2] < TS_KERNEL_REGISTERS \
                ? registers[code[pc + 2]] : inputs[code[pc + 2] - TS_KERNEL_REGISTERS] + base; \
            const type *right = code[pc + 3] < TS_KERNEL_REGISTERS \
                ? registers[code[pc + 3]] : inputs[code[pc + 3] - TS_KERNEL_REGISTERS] + base; \
            if (reduce) { \
                int status = ts_kernel_reduce_##suffix(code[pc], left, right, 0, m, &block_result); \
                if (status) return status; \
                continue; \
            } \
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
                    if (left[j] < 0) return -2; \
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
            /* TODO: scalar comparison and selection lanes; add packed VM opcodes (#72). */ \
            case TS_OP_EQ: for (; i < m; ++i) destination[i] = left[i] == right[i]; break; \
            case TS_OP_NE: for (; i < m; ++i) destination[i] = left[i] != right[i]; break; \
            case TS_OP_LT: for (; i < m; ++i) destination[i] = left[i] < right[i]; break; \
            case TS_OP_LE: for (; i < m; ++i) destination[i] = left[i] <= right[i]; break; \
            case TS_OP_GT: for (; i < m; ++i) destination[i] = left[i] > right[i]; break; \
            case TS_OP_GE: for (; i < m; ++i) destination[i] = left[i] >= right[i]; break; \
            case TS_OP_SELECT: \
                for (; i < m; ++i) destination[i] = left[i] != 0 ? right[i] : destination[i]; \
                break; \
            case TS_OP_NEGATE: \
                for (; i + width <= m; i += width) \
                    store(destination + i, sub(zero(), load(left + i))); \
                for (; i < m; ++i) destination[i] = (type)0 - left[i]; \
                break; \
            } \
        } \
        if (sum) result += block_result; \
    } \
    if (sum) *out = result; \
    return 0; \
}

#define TS_KERNEL(suffix, type, vector_type, width, load, store, zero, add, sub, mul, div, \
                  sqrtv, absv, minv, maxv, fmav, sqrts, abss, fmas) \
static int ts_kernel_reduce_##suffix(uint8_t op, const type *left, const type *right, \
                                    type constant, size_t m, type *out) { \
    vector_type lanes = zero(); \
    type result = 0; \
    size_t i = 0; \
    switch (op) { \
    case TS_OP_CONSTANT: { \
        type repeated[width]; \
        for (size_t lane = 0; lane < width; ++lane) repeated[lane] = constant; \
        TS_KERNEL_REDUCE(type, width, store, add, load(repeated), constant); \
        break; \
    } \
    case TS_OP_COPY: \
        TS_KERNEL_REDUCE(type, width, store, add, load(left + i), left[i]); \
        break; \
    case TS_OP_ADD: \
        TS_KERNEL_REDUCE(type, width, store, add, add(load(left + i), load(right + i)), left[i] + right[i]); \
        break; \
    case TS_OP_SUBTRACT: \
        TS_KERNEL_REDUCE(type, width, store, add, sub(load(left + i), load(right + i)), left[i] - right[i]); \
        break; \
    case TS_OP_MULTIPLY: \
        TS_KERNEL_REDUCE(type, width, store, add, mul(load(left + i), load(right + i)), left[i] * right[i]); \
        break; \
    case TS_OP_DIVIDE: \
        TS_KERNEL_REDUCE(type, width, store, add, div(load(left + i), load(right + i)), left[i] / right[i]); \
        break; \
    case TS_OP_SQRT: \
        for (size_t j = 0; j < m; ++j) if (left[j] < 0) return -2; \
        TS_KERNEL_REDUCE(type, width, store, add, sqrtv(load(left + i)), (left[i] == 0 ? left[i] : sqrts(left[i]))); \
        break; \
    case TS_OP_ABS: \
        TS_KERNEL_REDUCE(type, width, store, add, absv(load(left + i)), abss(left[i])); \
        break; \
    case TS_OP_MIN: \
        TS_KERNEL_REDUCE(type, width, store, add, minv(load(left + i), load(right + i)), (left[i] <= right[i] ? left[i] : right[i])); \
        break; \
    case TS_OP_MAX: \
        TS_KERNEL_REDUCE(type, width, store, add, maxv(load(left + i), load(right + i)), (left[i] >= right[i] ? left[i] : right[i])); \
        break; \
    case TS_OP_NEGATE: \
        TS_KERNEL_REDUCE(type, width, store, add, sub(zero(), load(left + i)), (type)0 - left[i]); \
        break; \
    default: return -1; \
    } \
    *out = result; \
    return 0; \
} \
TS_KERNEL_RUN(suffix, elementwise, 0, type, width, load, store, zero, add, sub, mul, div, \
              sqrtv, absv, minv, maxv, fmav, sqrts, abss, fmas) \
TS_KERNEL_RUN(suffix, sum, 1, type, width, load, store, zero, add, sub, mul, div, \
              sqrtv, absv, minv, maxv, fmav, sqrts, abss, fmas) \
static int ts_kernel_allocate_##suffix(const uint8_t *code, size_t code_length, \
                       const type *constants, const type *const *inputs, \
                       type *out, size_t n, size_t scratch_count, int sum) { \
    if (n == 0) { if (sum) *out = 0; return 0; } \
    if (scratch_count > SIZE_MAX / TS_KERNEL_BLOCK / sizeof(type)) return -1; \
    type *scratch = scratch_count ? malloc(scratch_count * TS_KERNEL_BLOCK * sizeof(type)) : NULL; \
    if (scratch_count && !scratch) return -1; \
    int status = sum \
        ? ts_kernel_run_sum_##suffix(code, code_length, constants, inputs, out, n, scratch_count, scratch) \
        : ts_kernel_run_elementwise_##suffix(code, code_length, constants, inputs, out, n, scratch_count, scratch); \
    free(scratch); \
    return status; \
} \
int ts_kernel_##suffix(const uint8_t *code, size_t code_length, \
                       const type *constants, const type *const *inputs, \
                       type *out, size_t n, size_t scratch_count) { \
    return ts_kernel_allocate_##suffix(code, code_length, constants, inputs, out, n, scratch_count, 0); \
} \
int ts_kernel_sum_##suffix(const uint8_t *code, size_t code_length, \
                           const type *constants, const type *const *inputs, \
                           type *out, size_t n, size_t scratch_count) { \
    return ts_kernel_allocate_##suffix(code, code_length, constants, inputs, out, n, scratch_count, 1); \
} \
int ts_kernel_with_scratch_##suffix(const uint8_t *code, size_t code_length, \
                       const type *constants, const type *const *inputs, \
                       type *out, size_t n, size_t scratch_count, type *scratch, size_t capacity) { \
    if (n && scratch_count && (!scratch || capacity < scratch_count)) return -1; \
    return ts_kernel_run_elementwise_##suffix(code, code_length, constants, inputs, out, n, scratch_count, scratch); \
} \
int ts_kernel_sum_with_scratch_##suffix(const uint8_t *code, size_t code_length, \
                       const type *constants, const type *const *inputs, \
                       type *out, size_t n, size_t scratch_count, type *scratch, size_t capacity) { \
    if (n && scratch_count && (!scratch || capacity < scratch_count)) return -1; \
    return ts_kernel_run_sum_##suffix(code, code_length, constants, inputs, out, n, scratch_count, scratch); \
}

TS_KERNEL(f32, float, TS_F32_VECTOR, TS_F32_WIDTH, TS_F32_LOAD, TS_F32_STORE, TS_F32_ZERO,
          TS_F32_ADD, TS_F32_SUB, TS_F32_MUL, TS_F32_DIV,
          TS_F32_SQRT, TS_F32_ABS, TS_F32_MIN, TS_F32_MAX, TS_F32_FMA, sqrtf, fabsf, fmaf)
TS_KERNEL(f64, double, TS_F64_VECTOR, TS_F64_WIDTH, TS_F64_LOAD, TS_F64_STORE, TS_F64_ZERO,
          TS_F64_ADD, TS_F64_SUB, TS_F64_MUL, TS_F64_DIV,
          TS_F64_SQRT, TS_F64_ABS, TS_F64_MIN, TS_F64_MAX, TS_F64_FMA, sqrt, fabs, fma)

#include "integer.h"

#define TS_KERNEL_ROWS(suffix, type) \
int ts_kernel_sum_rows_##suffix(const uint8_t *code, size_t code_length, \
                               const type *constants, const type *const *inputs, \
                               size_t input_count, const int64_t *row_strides, \
                               type *out, size_t rows, size_t n, size_t scratch_count) { \
    if (!rows) return 0; \
    if (!out || rows > PTRDIFF_MAX / sizeof(type)) return -1; \
    if (!n) { for (size_t r = 0; r < rows; ++r) out[r] = 0; return 0; } \
    if (input_count > 248 || n > PTRDIFF_MAX / sizeof(type)) return -1; \
    if (input_count && (!inputs || !row_strides)) return -1; \
    for (size_t i = 0; i < input_count; ++i) { \
        uint64_t stride = row_strides[i] < 0 \
            ? (uint64_t)(-(row_strides[i] + 1)) + 1 : (uint64_t)row_strides[i]; \
        if (!inputs[i] || (stride && rows - 1 > (PTRDIFF_MAX / sizeof(type) - n) / stride)) return -1; \
    } \
    if (scratch_count > SIZE_MAX / TS_KERNEL_BLOCK / sizeof(type)) return -1; \
    type *scratch = scratch_count ? malloc(scratch_count * TS_KERNEL_BLOCK * sizeof(type)) : NULL; \
    if (scratch_count && !scratch) return -1; \
    const type *shifted[248]; \
    int status = 0; \
    for (size_t r = 0; r < rows; ++r) { \
        for (size_t i = 0; i < input_count; ++i) { \
            uint64_t stride = row_strides[i] < 0 \
                ? (uint64_t)(-(row_strides[i] + 1)) + 1 : (uint64_t)row_strides[i]; \
            ptrdiff_t offset = (ptrdiff_t)(r * stride); \
            shifted[i] = inputs[i] + (row_strides[i] < 0 ? -offset : offset); \
        } \
        status = ts_kernel_run_sum_##suffix(code, code_length, constants, shifted, \
                                             out + r, n, scratch_count, scratch); \
        if (status) break; \
    } \
    free(scratch); \
    return status; \
}

TS_KERNEL_ROWS(f32, float)
TS_KERNEL_ROWS(f64, double)

#define TS_KERNEL_MASK(suffix, type) \
int ts_kernel_mask_##suffix(const uint8_t *code, size_t code_length, const type *constants, \
                            const type *const *inputs, size_t input_count, uint8_t *mask, \
                            size_t n, size_t scratch_count, unsigned reduction, size_t *result) { \
    size_t total = 0; \
    for (size_t base = 0; base < n; base += TS_KERNEL_BLOCK) { \
        size_t m = n - base < TS_KERNEL_BLOCK ? n - base : TS_KERNEL_BLOCK; \
        const type *shifted[256]; \
        type values[TS_KERNEL_BLOCK]; \
        for (size_t j = 0; j < input_count; ++j) shifted[j] = inputs[j] + base; \
        /* TODO: spilled mask kernels allocate per block; reuse one scratch buffer (#72). */ \
        int status = ts_kernel_##suffix(code, code_length, constants, shifted, values, m, scratch_count); \
        if (status) return status; \
        for (size_t j = 0; j < m; ++j) { \
            int truth = values[j] != 0; \
            if (reduction == 0) mask[base + j] = (uint8_t)truth; \
            else total += truth; \
        } \
    } \
    if (reduction) *result = reduction == 1 ? total : reduction == 2 ? total != 0 : total == n; \
    return 0; \
}

TS_KERNEL_MASK(f32, float)
TS_KERNEL_MASK(f64, double)
TS_KERNEL_MASK(s8, uint8_t)
TS_KERNEL_MASK(u8, uint8_t)
TS_KERNEL_MASK(s16, uint16_t)
TS_KERNEL_MASK(u16, uint16_t)
TS_KERNEL_MASK(s32, uint32_t)
TS_KERNEL_MASK(u32, uint32_t)
TS_KERNEL_MASK(s64, uint64_t)
TS_KERNEL_MASK(u64, uint64_t)

/* Reducer codes for ts_kernel_reduction_*; SUMSQ and later reducers are float-only. */
enum { TS_REDUCE_ARGMIN = 1, TS_REDUCE_ARGMAX, TS_REDUCE_ASUM,
       TS_REDUCE_SUMSQ, TS_REDUCE_MAXABS, TS_REDUCE_SCALED_SUMSQ };

#define TS_REDUCE_FLOAT_OPERATIONS(suffix, type, absolute) \
static int ts_reduce_less_##suffix(type x, type y) { return x < y; } \
static int ts_reduce_greater_##suffix(type x, type y) { return x > y; } \
static type ts_reduce_abs_##suffix(type x) { return absolute(x); } \
static type ts_reduce_add_##suffix(type x, type y) { return x + y; }

#define TS_REDUCE_INTEGER_OPERATIONS(suffix, type) \
static int ts_reduce_less_##suffix(type x, type y) { \
    return x != y && ts_value_##suffix(TS_OP_MIN, x, y) == x; \
} \
static int ts_reduce_greater_##suffix(type x, type y) { \
    return x != y && ts_value_##suffix(TS_OP_MAX, x, y) == x; \
} \
static type ts_reduce_abs_##suffix(type x) { return ts_value_##suffix(TS_OP_ABS, x, 0); } \
static type ts_reduce_add_##suffix(type x, type y) { return ts_value_##suffix(TS_OP_ADD, x, y); }

/* Larger magnitudes report an out-of-range sum so callers can rescale. Clamping
 * keeps the multiply from overflowing, which traps under some hosts. */
#define TS_SQUARE_LIMIT 1e140

#define TS_REDUCE_LANES 8

/* Independent lanes find the block's extreme value without a serial compare chain; a second
 * scan then locates its first occurrence, so ties keep the earlier element. */
#define TS_REDUCE_EXTREME(suffix, type, better) \
    { \
        type lane[TS_REDUCE_LANES]; \
        for (size_t k = 0; k < TS_REDUCE_LANES; ++k) lane[k] = values[0]; \
        size_t j = 0; \
        for (; j + TS_REDUCE_LANES <= m; j += TS_REDUCE_LANES) \
            for (size_t k = 0; k < TS_REDUCE_LANES; ++k) \
                if (better(values[j + k], lane[k])) lane[k] = values[j + k]; \
        for (; j < m; ++j) if (better(values[j], lane[0])) lane[0] = values[j]; \
        type extreme = lane[0]; \
        for (size_t k = 1; k < TS_REDUCE_LANES; ++k) if (better(lane[k], extreme)) extreme = lane[k]; \
        size_t first = 0; \
        while (first + 1 < m && (better(extreme, values[first]) || better(values[first], extreme))) ++first; \
        if (base == 0 || better(values[first], best)) { best = values[first]; best_index = base + first; } \
    }

/* TODO: scalar reduction of an evaluated block; fuse into the VM with packed lanes (#96). */
/* The first extreme wins, so ties keep the earlier element. */
#define TS_KERNEL_REDUCTION(suffix, type, integral) \
int ts_kernel_reduction_##suffix(const uint8_t *code, size_t code_length, const type *constants, \
                                const type *const *inputs, size_t input_count, size_t n, \
                              size_t scratch_count, unsigned reducer, double scale, \
                              type *value, double *wide, size_t *index) { \
    if (reducer < TS_REDUCE_ARGMIN || reducer > TS_REDUCE_SCALED_SUMSQ || \
        (integral && reducer > TS_REDUCE_ASUM) || input_count > 256) return -1; \
    if (scratch_count > SIZE_MAX / TS_KERNEL_BLOCK / sizeof(type)) return -1; \
    type *scratch = scratch_count ? malloc(scratch_count * TS_KERNEL_BLOCK * sizeof(type)) : NULL; \
    if (scratch_count && !scratch) return -1; \
    type best = 0, total = 0; \
    double accumulator = 0; \
    int out_of_range = 0; \
    size_t best_index = 0; \
    for (size_t base = 0; base < n; base += TS_KERNEL_BLOCK) { \
        size_t m = n - base < TS_KERNEL_BLOCK ? n - base : TS_KERNEL_BLOCK; \
        const type *shifted[256]; \
        type values[TS_KERNEL_BLOCK]; \
        for (size_t j = 0; j < input_count; ++j) shifted[j] = inputs[j] + base; \
        int status = ts_kernel_with_scratch_##suffix(code, code_length, constants, shifted, \
                                                     values, m, scratch_count, scratch, \
                                                     scratch_count); \
        if (status) { free(scratch); return status; } \
        switch (reducer) { \
        case TS_REDUCE_ARGMIN: TS_REDUCE_EXTREME(suffix, type, ts_reduce_less_##suffix) break; \
        case TS_REDUCE_ARGMAX: TS_REDUCE_EXTREME(suffix, type, ts_reduce_greater_##suffix) break; \
        case TS_REDUCE_ASUM: { \
            type lane[TS_REDUCE_LANES] = {0}; \
            size_t j = 0; \
            for (; j + TS_REDUCE_LANES <= m; j += TS_REDUCE_LANES) \
                for (size_t k = 0; k < TS_REDUCE_LANES; ++k) \
                    lane[k] = ts_reduce_add_##suffix(lane[k], ts_reduce_abs_##suffix(values[j + k])); \
            for (; j < m; ++j) lane[0] = ts_reduce_add_##suffix(lane[0], ts_reduce_abs_##suffix(values[j])); \
            for (size_t k = 0; k < TS_REDUCE_LANES; ++k) total = ts_reduce_add_##suffix(total, lane[k]); \
            break; \
        } \
        case TS_REDUCE_SUMSQ: \
        case TS_REDUCE_SCALED_SUMSQ: { \
            double lane[TS_REDUCE_LANES] = {0}; \
            double divisor = reducer == TS_REDUCE_SUMSQ ? 1 : scale; \
            size_t j = 0; \
            for (; j + TS_REDUCE_LANES <= m; j += TS_REDUCE_LANES) \
                for (size_t k = 0; k < TS_REDUCE_LANES; ++k) { \
                    double d = integral ? 0 : (double)values[j + k] / divisor; \
                    double clamped = fmin(fabs(d), TS_SQUARE_LIMIT); \
                    if (!(fabs(d) <= TS_SQUARE_LIMIT)) out_of_range = 1; \
                    lane[k] += clamped * clamped; \
                } \
            for (; j < m; ++j) { \
                double d = integral ? 0 : (double)values[j] / divisor; \
                double clamped = fmin(fabs(d), TS_SQUARE_LIMIT); \
                if (!(fabs(d) <= TS_SQUARE_LIMIT)) out_of_range = 1; \
                lane[0] += clamped * clamped; \
            } \
            for (size_t k = 0; k < TS_REDUCE_LANES; ++k) accumulator += lane[k]; \
            break; \
        } \
        default: \
            for (size_t j = 0; j < m; ++j) { \
                double d = integral ? 0 : fabs((double)values[j]); \
                if (d > accumulator) accumulator = d; \
            } \
            break; \
        } \
    } \
    free(scratch); \
    *value = reducer == TS_REDUCE_ASUM ? total : best; \
    *wide = out_of_range ? HUGE_VAL : accumulator; \
    *index = best_index; \
    return 0; \
}

TS_REDUCE_FLOAT_OPERATIONS(f32, float, fabsf)
TS_REDUCE_FLOAT_OPERATIONS(f64, double, fabs)
TS_KERNEL_REDUCTION(f32, float, 0)
TS_KERNEL_REDUCTION(f64, double, 0)

#include "kernel-inputs.h"
#define TS_REDUCE_INTEGER(suffix, type) \
    TS_REDUCE_INTEGER_OPERATIONS(suffix, type) \
    TS_KERNEL_REDUCTION(suffix, type, 1)
TS_REDUCE_INTEGER(s8, uint8_t)
TS_REDUCE_INTEGER(u8, uint8_t)
TS_REDUCE_INTEGER(s16, uint16_t)
TS_REDUCE_INTEGER(u16, uint16_t)
TS_REDUCE_INTEGER(s32, uint32_t)
TS_REDUCE_INTEGER(u32, uint32_t)
TS_REDUCE_INTEGER(s64, uint64_t)
TS_REDUCE_INTEGER(u64, uint64_t)
#undef TS_REDUCE_INTEGER

void ts_swap_bytes(void *a, void *b, size_t bytes) {
    unsigned char *x = a, *y = b;
    for (; bytes >= 16; bytes -= 16, x += 16, y += 16) {
        unsigned char block[16];
        memcpy(block, x, 16);
        memcpy(x, y, 16);
        memcpy(y, block, 16);
    }
    for (; bytes; --bytes, ++x, ++y) {
        unsigned char value = *x;
        *x = *y;
        *y = value;
    }
}

#define TS_FILL(suffix, type) \
void ts_fill_##suffix(type *out, type value, size_t n) { \
    for (size_t i = 0; i < n; ++i) out[i] = value; \
}
TS_FILL(f32, float)
TS_FILL(f64, double)
TS_FILL(s8, uint8_t)
TS_FILL(u8, uint8_t)
TS_FILL(s16, uint16_t)
TS_FILL(u16, uint16_t)
TS_FILL(s32, uint32_t)
TS_FILL(u32, uint32_t)
TS_FILL(s64, uint64_t)
TS_FILL(u64, uint64_t)
#undef TS_FILL
