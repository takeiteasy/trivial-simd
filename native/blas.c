#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "fma.h"

#pragma STDC FP_CONTRACT OFF

#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
#include <arm_neon.h>
#define TS_BLAS_F32_VECTOR float32x4_t
#define TS_BLAS_F64_VECTOR float64x2_t
#define TS_BLAS_F32_WIDTH 4
#define TS_BLAS_F64_WIDTH 2
#define TS_BLAS_F32_LOAD(p) vld1q_f32(p)
#define TS_BLAS_F64_LOAD(p) vld1q_f64(p)
#define TS_BLAS_F32_STORE(p, v) vst1q_f32(p, v)
#define TS_BLAS_F64_STORE(p, v) vst1q_f64(p, v)
#define TS_BLAS_F32_ZERO() vdupq_n_f32(0.0f)
#define TS_BLAS_F64_ZERO() vdupq_n_f64(0.0)
#define TS_BLAS_F32_SPLAT(a) vdupq_n_f32(a)
#define TS_BLAS_F64_SPLAT(a) vdupq_n_f64(a)
#define TS_BLAS_F32_MULADD(acc, a, b) vfmaq_f32(acc, a, b)
#define TS_BLAS_F64_MULADD(acc, a, b) vfmaq_f64(acc, a, b)
#define TS_BLAS_F32_ROWS 2
#define TS_BLAS_F32_COLUMNS 8
#define TS_BLAS_F64_ROWS 4
#define TS_BLAS_F64_COLUMNS 4
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#define TS_BLAS_F32_VECTOR __m128
#define TS_BLAS_F64_VECTOR __m128d
#define TS_BLAS_F32_WIDTH 4
#define TS_BLAS_F64_WIDTH 2
#define TS_BLAS_F32_LOAD(p) _mm_loadu_ps(p)
#define TS_BLAS_F64_LOAD(p) _mm_loadu_pd(p)
#define TS_BLAS_F32_STORE(p, v) _mm_storeu_ps(p, v)
#define TS_BLAS_F64_STORE(p, v) _mm_storeu_pd(p, v)
#define TS_BLAS_F32_ZERO() _mm_setzero_ps()
#define TS_BLAS_F64_ZERO() _mm_setzero_pd()
#define TS_BLAS_F32_SPLAT(a) _mm_set1_ps(a)
#define TS_BLAS_F64_SPLAT(a) _mm_set1_pd(a)
#define TS_BLAS_F32_MULADD(acc, a, b) _mm_add_ps(acc, _mm_mul_ps(a, b))
#define TS_BLAS_F64_MULADD(acc, a, b) _mm_add_pd(acc, _mm_mul_pd(a, b))
#define TS_BLAS_F32_ROWS 2
#define TS_BLAS_F32_COLUMNS 4
#define TS_BLAS_F64_ROWS 2
#define TS_BLAS_F64_COLUMNS 4
#ifdef TS_HAVE_HARDWARE_FMA
#define TS_BLAS_DISPATCH
#define TS_BLAS_AVX_F32_VECTOR __m256
#define TS_BLAS_AVX_F64_VECTOR __m256d
#define TS_BLAS_AVX_F32_WIDTH 8
#define TS_BLAS_AVX_F64_WIDTH 4
#define TS_BLAS_AVX_F32_LOAD(p) _mm256_loadu_ps(p)
#define TS_BLAS_AVX_F64_LOAD(p) _mm256_loadu_pd(p)
#define TS_BLAS_AVX_F32_STORE(p, v) _mm256_storeu_ps(p, v)
#define TS_BLAS_AVX_F64_STORE(p, v) _mm256_storeu_pd(p, v)
#define TS_BLAS_AVX_F32_ZERO() _mm256_setzero_ps()
#define TS_BLAS_AVX_F64_ZERO() _mm256_setzero_pd()
#define TS_BLAS_AVX_F32_SPLAT(a) _mm256_set1_ps(a)
#define TS_BLAS_AVX_F64_SPLAT(a) _mm256_set1_pd(a)
#define TS_BLAS_AVX_F32_MULADD(acc, a, b) _mm256_fmadd_ps(a, b, acc)
#define TS_BLAS_AVX_F64_MULADD(acc, a, b) _mm256_fmadd_pd(a, b, acc)
#ifndef TS_BLAS_AVX_F32_ROWS
#define TS_BLAS_AVX_F32_ROWS 2
#endif
#ifndef TS_BLAS_AVX_F32_COLUMNS
#define TS_BLAS_AVX_F32_COLUMNS 6
#endif
#ifndef TS_BLAS_AVX_F64_ROWS
#define TS_BLAS_AVX_F64_ROWS 2
#endif
#ifndef TS_BLAS_AVX_F64_COLUMNS
#define TS_BLAS_AVX_F64_COLUMNS 6
#endif
#endif
#else
#define TS_BLAS_F32_VECTOR float
#define TS_BLAS_F64_VECTOR double
#define TS_BLAS_F32_WIDTH 1
#define TS_BLAS_F64_WIDTH 1
#define TS_BLAS_F32_LOAD(p) (*(p))
#define TS_BLAS_F64_LOAD(p) (*(p))
#define TS_BLAS_F32_STORE(p, v) (*(p) = (v))
#define TS_BLAS_F64_STORE(p, v) (*(p) = (v))
#define TS_BLAS_F32_ZERO() 0.0f
#define TS_BLAS_F64_ZERO() 0.0
#define TS_BLAS_F32_SPLAT(a) (a)
#define TS_BLAS_F64_SPLAT(a) (a)
#define TS_BLAS_F32_MULADD(acc, a, b) ((acc) + (a) * (b))
#define TS_BLAS_F64_MULADD(acc, a, b) ((acc) + (a) * (b))
#define TS_BLAS_F32_ROWS 4
#define TS_BLAS_F32_COLUMNS 4
#define TS_BLAS_F64_ROWS 4
#define TS_BLAS_F64_COLUMNS 4
#endif

#ifndef TS_BLAS_DEPTH
#define TS_BLAS_DEPTH 256
#endif
#ifndef TS_BLAS_ROW_BLOCK
#define TS_BLAS_ROW_BLOCK 128
#endif
#ifndef TS_BLAS_COLUMN_BLOCK
#define TS_BLAS_COLUMN_BLOCK 1024
#endif
enum { TS_BLAS_FULL, TS_BLAS_UPPER, TS_BLAS_LOWER };
#define TS_BLAS_MIN(a, b) ((a) < (b) ? (a) : (b))
#define TS_BLAS_MAX(a, b) ((a) > (b) ? (a) : (b))
#define TS_BLAS_ROUND_UP(a, to) (((a) + (to) - 1) / (to) * (to))
#ifdef _MSC_VER
#define TS_BLAS_NOINLINE __declspec(noinline)
#else
#define TS_BLAS_NOINLINE __attribute__((noinline))
#endif

/* Packed panels follow the BLIS layout: A in MR-row slivers, B in NR-column
   slivers, so the micro-kernel streams both contiguously whatever the strides. */
#define TS_BLAS_GEMM(suffix, type, vector, width, load, store, zero, splat, muladd, rows, columns) \
enum { TS_BLAS_MR_##suffix = (rows) * (width), TS_BLAS_NR_##suffix = (columns) }; \
\
static void ts_blas_micro_##suffix(ptrdiff_t depth, const type *a, const type *b, type *tile) { \
    vector acc[rows][columns]; \
    for (int v = 0; v < (rows); ++v) \
        for (int j = 0; j < (columns); ++j) acc[v][j] = zero(); \
    for (ptrdiff_t p = 0; p < depth; ++p) { \
        vector left[rows]; \
        for (int v = 0; v < (rows); ++v) left[v] = load(a + v * (width)); \
        for (int j = 0; j < (columns); ++j) { \
            vector right = splat(b[j]); \
            for (int v = 0; v < (rows); ++v) acc[v][j] = muladd(acc[v][j], left[v], right); \
        } \
        a += TS_BLAS_MR_##suffix; \
        b += TS_BLAS_NR_##suffix; \
    } \
    for (int j = 0; j < (columns); ++j) \
        for (int v = 0; v < (rows); ++v) \
            store(tile + j * TS_BLAS_MR_##suffix + v * (width), acc[v][j]); \
} \
\
static void ts_blas_pack_a_##suffix(ptrdiff_t m, ptrdiff_t depth, type alpha, const type *a, \
                                    ptrdiff_t rs, ptrdiff_t cs, type *out) { \
    for (ptrdiff_t i = 0; i < m; i += TS_BLAS_MR_##suffix) { \
        ptrdiff_t live = TS_BLAS_MIN(TS_BLAS_MR_##suffix, m - i); \
        for (ptrdiff_t p = 0; p < depth; ++p) { \
            const type *source = a + i * rs + p * cs; \
            ptrdiff_t r = 0; \
            for (; r < live; ++r) out[r] = alpha * source[r * rs]; \
            for (; r < TS_BLAS_MR_##suffix; ++r) out[r] = 0; \
            out += TS_BLAS_MR_##suffix; \
        } \
    } \
} \
\
static void ts_blas_pack_b_##suffix(ptrdiff_t n, ptrdiff_t depth, const type *b, \
                                    ptrdiff_t rs, ptrdiff_t cs, type *out) { \
    for (ptrdiff_t j = 0; j < n; j += TS_BLAS_NR_##suffix) { \
        ptrdiff_t live = TS_BLAS_MIN(TS_BLAS_NR_##suffix, n - j); \
        for (ptrdiff_t p = 0; p < depth; ++p) { \
            const type *source = b + p * rs + j * cs; \
            ptrdiff_t c = 0; \
            for (; c < live; ++c) out[c] = source[c * cs]; \
            for (; c < TS_BLAS_NR_##suffix; ++c) out[c] = 0; \
            out += TS_BLAS_NR_##suffix; \
        } \
    } \
} \
\
static void ts_blas_scale_##suffix(ptrdiff_t m, ptrdiff_t n, type beta, type *c, \
                                   ptrdiff_t rs, ptrdiff_t cs) { \
    if (beta == 1) return; \
    if (rs >= cs) { \
        for (ptrdiff_t i = 0; i < m; ++i) \
            for (ptrdiff_t j = 0; j < n; ++j) { \
                type *item = c + i * rs + j * cs; \
                *item = beta == 0 ? 0 : beta * *item; \
            } \
    } else { \
        for (ptrdiff_t j = 0; j < n; ++j) \
            for (ptrdiff_t i = 0; i < m; ++i) { \
                type *item = c + i * rs + j * cs; \
                *item = beta == 0 ? 0 : beta * *item; \
            } \
    } \
} \
\
static size_t ts_blas_gemm_size_##suffix(int64_t m, int64_t n, int64_t k) { \
    return (size_t)(TS_BLAS_MIN(k, TS_BLAS_DEPTH) \
                    * (TS_BLAS_ROUND_UP(TS_BLAS_MIN(m, TS_BLAS_ROW_BLOCK), TS_BLAS_MR_##suffix) \
                       + TS_BLAS_ROUND_UP(TS_BLAS_MIN(n, TS_BLAS_COLUMN_BLOCK), TS_BLAS_NR_##suffix))); \
} \
\
TS_BLAS_NOINLINE static void ts_blas_gemm_run_##suffix(int64_t m, int64_t n, int64_t k, type alpha, \
                                      const type *a, int64_t ars, int64_t acs, \
                                      const type *b, int64_t brs, int64_t bcs, type beta, \
                                      type *c, int64_t crs, int64_t ccs, type *work, int part) { \
    if (m <= 0 || n <= 0) return; \
    ts_blas_scale_##suffix(m, n, beta, c, crs, ccs); \
    if (k <= 0 || alpha == 0) return; \
    type *packed_a = work; \
    type *packed_b = work + TS_BLAS_ROUND_UP(TS_BLAS_MIN(m, TS_BLAS_ROW_BLOCK), TS_BLAS_MR_##suffix) \
                                * TS_BLAS_MIN(k, TS_BLAS_DEPTH); \
    type tile[TS_BLAS_MR_##suffix * TS_BLAS_NR_##suffix]; \
    for (ptrdiff_t jc = 0; jc < n; jc += TS_BLAS_COLUMN_BLOCK) { \
        ptrdiff_t nc = TS_BLAS_MIN(TS_BLAS_COLUMN_BLOCK, n - jc); \
        for (ptrdiff_t pc = 0; pc < k; pc += TS_BLAS_DEPTH) { \
            ptrdiff_t kc = TS_BLAS_MIN(TS_BLAS_DEPTH, k - pc); \
            ts_blas_pack_b_##suffix(nc, kc, b + pc * brs + jc * bcs, brs, bcs, packed_b); \
            for (ptrdiff_t ic = 0; ic < m; ic += TS_BLAS_ROW_BLOCK) { \
                ptrdiff_t mc = TS_BLAS_MIN(TS_BLAS_ROW_BLOCK, m - ic); \
                ts_blas_pack_a_##suffix(mc, kc, alpha, a + ic * ars + pc * acs, ars, acs, packed_a); \
                for (ptrdiff_t jr = 0; jr < nc; jr += TS_BLAS_NR_##suffix) { \
                    ptrdiff_t live_columns = TS_BLAS_MIN(TS_BLAS_NR_##suffix, nc - jr); \
                    for (ptrdiff_t ir = 0; ir < mc; ir += TS_BLAS_MR_##suffix) { \
                        ptrdiff_t live_rows = TS_BLAS_MIN(TS_BLAS_MR_##suffix, mc - ir); \
                        ptrdiff_t row = ic + ir, column = jc + jr; \
                        if (part == TS_BLAS_UPPER && column + live_columns - 1 < row) continue; \
                        if (part == TS_BLAS_LOWER && column > row + live_rows - 1) continue; \
                        type *target = c + row * crs + column * ccs; \
                        ts_blas_micro_##suffix(kc, packed_a + ir * kc, packed_b + jr * kc, tile); \
                        for (ptrdiff_t j = 0; j < live_columns; ++j) \
                            for (ptrdiff_t i = 0; i < live_rows; ++i) { \
                                ptrdiff_t offset = column + j - row - i; \
                                if ((part == TS_BLAS_UPPER && offset < 0) || (part == TS_BLAS_LOWER && offset > 0)) continue; \
                                target[i * crs + j * ccs] += tile[j * TS_BLAS_MR_##suffix + i]; \
                            } \
                    } \
                } \
            } \
        } \
    } \
} \
\
int ts_blas_gemm_##suffix(int64_t m, int64_t n, int64_t k, type alpha, \
                          const type *a, int64_t ars, int64_t acs, \
                          const type *b, int64_t brs, int64_t bcs, type beta, \
                          type *c, int64_t crs, int64_t ccs) { \
    if (m <= 0 || n <= 0) return 0; \
    if (k <= 0 || alpha == 0) { \
        ts_blas_gemm_run_##suffix(m, n, k, alpha, a, ars, acs, b, brs, bcs, beta, c, crs, ccs, NULL, TS_BLAS_FULL); \
        return 0; \
    } \
    type *work = malloc(ts_blas_gemm_size_##suffix(m, n, k) * sizeof(type)); \
    if (!work) return 1; \
    ts_blas_gemm_run_##suffix(m, n, k, alpha, a, ars, acs, b, brs, bcs, beta, c, crs, ccs, work, TS_BLAS_FULL); \
    free(work); \
    return 0; \
} \
\
int ts_blas_gemm_batch_##suffix(int64_t m, int64_t n, int64_t k, type alpha, \
                                const type *a, int64_t ars, int64_t acs, int64_t astep, \
                                const type *b, int64_t brs, int64_t bcs, int64_t bstep, type beta, \
                                type *c, int64_t crs, int64_t ccs, int64_t cstep, int64_t count) { \
    if (count <= 0 || m <= 0 || n <= 0) return 0; \
    type *work = NULL; \
    if (k > 0 && alpha != 0) { \
        work = malloc(ts_blas_gemm_size_##suffix(m, n, k) * sizeof(type)); \
        if (!work) return 1; \
    } \
    for (int64_t i = 0; i < count; ++i) \
        ts_blas_gemm_run_##suffix(m, n, k, alpha, a + i * astep, ars, acs, \
                                  b + i * bstep, brs, bcs, beta, c + i * cstep, crs, ccs, \
                                  work, TS_BLAS_FULL); \
    free(work); \
    return 0; \
}


#define TS_BLAS_BLOCK 16
#ifndef TS_BLAS_TRIANGLE_BLOCK
#define TS_BLAS_TRIANGLE_BLOCK 64
#endif

/* Left lower solve L X = W in place on w (row stride wrs, column stride 1); an upper
   solve runs as a lower one on the reversed matrix. Each block of TS_BLAS_DEPTH rows
   packs L as MR-row slivers [off-diagonal panel | negated strict triangle | inverse
   diagonal]. The fused kernel subtracts the panel product from B, then solves the
   MR x NR tile by multiplying with the inverse diagonal. Rows below the block are
   updated with one gemm. */
#define TS_BLAS_TRSM(suffix, type, vector, width, load, store, splat, muladd, vectors) \
enum { TS_BLAS_TRSM_BLOCK_##suffix = TS_BLAS_MAX(TS_BLAS_MR_##suffix, \
                                                 TS_BLAS_DEPTH / TS_BLAS_MR_##suffix * TS_BLAS_MR_##suffix) }; \
\
static size_t ts_blas_triangle_size_##suffix(int64_t rows) { \
    int64_t slivers = (TS_BLAS_MIN(rows, TS_BLAS_TRSM_BLOCK_##suffix) + TS_BLAS_MR_##suffix - 1) \
                      / TS_BLAS_MR_##suffix; \
    return (size_t)(TS_BLAS_MR_##suffix * (slivers * (TS_BLAS_MR_##suffix + 1) \
                                          + TS_BLAS_MR_##suffix * (slivers * (slivers - 1) / 2))); \
} \
\
static size_t ts_blas_trsm_size_##suffix(int64_t rows, int64_t cols) { \
    return ts_blas_triangle_size_##suffix(rows) \
           + ts_blas_gemm_size_##suffix(rows, cols, TS_BLAS_MIN(rows, TS_BLAS_TRSM_BLOCK_##suffix)); \
} \
\
static void ts_blas_pack_triangle_##suffix(int unit, ptrdiff_t m, const type *a, \
                                           ptrdiff_t rs, ptrdiff_t cs, type *out) { \
    for (ptrdiff_t r0 = 0; r0 < m; r0 += TS_BLAS_MR_##suffix) { \
        ptrdiff_t live = TS_BLAS_MIN(TS_BLAS_MR_##suffix, m - r0); \
        ts_blas_pack_a_##suffix(live, r0, 1, a + r0 * rs, rs, cs, out); \
        out += TS_BLAS_MR_##suffix * r0; \
        for (ptrdiff_t k = 0; k < TS_BLAS_MR_##suffix; ++k) \
            for (ptrdiff_t r = 0; r < TS_BLAS_MR_##suffix; ++r) \
                out[k * TS_BLAS_MR_##suffix + r] = r > k && r < live ? -a[(r0 + r) * rs + (r0 + k) * cs] : 0; \
        out += TS_BLAS_MR_##suffix * TS_BLAS_MR_##suffix; \
        for (ptrdiff_t r = 0; r < TS_BLAS_MR_##suffix; ++r) \
            out[r] = r >= live || unit ? 1 : (type)1 / a[(r0 + r) * rs + (r0 + r) * cs]; \
        out += TS_BLAS_MR_##suffix; \
    } \
} \
\
/* TODO: #86 the solve runs on the memory tile, not the live accumulators */ \
static void ts_blas_gemmtrsm_##suffix(ptrdiff_t depth, const type *a, type *b, ptrdiff_t live_rows, \
                                      ptrdiff_t live_columns, type *target, ptrdiff_t rs) { \
    const type *negated = a + TS_BLAS_MR_##suffix * depth; \
    const type *inverse = negated + TS_BLAS_MR_##suffix * TS_BLAS_MR_##suffix; \
    type *solved = b + depth * TS_BLAS_NR_##suffix; \
    type tile[TS_BLAS_MR_##suffix * TS_BLAS_NR_##suffix]; \
    ts_blas_micro_##suffix(depth, a, b, tile); \
    for (ptrdiff_t j = 0; j < live_columns; ++j) \
        for (ptrdiff_t r = 0; r < TS_BLAS_MR_##suffix; ++r) \
            tile[j * TS_BLAS_MR_##suffix + r] = \
                (r < live_rows ? solved[r * TS_BLAS_NR_##suffix + j] : 0) - tile[j * TS_BLAS_MR_##suffix + r]; \
    for (ptrdiff_t k = 0; k < TS_BLAS_MR_##suffix; ++k) { \
        const type *lower = negated + k * TS_BLAS_MR_##suffix; \
        ptrdiff_t next = k / (width) + 1; \
        for (ptrdiff_t j = 0; j < live_columns; ++j) { \
            type *column = tile + j * TS_BLAS_MR_##suffix; \
            type x = column[k] * inverse[k]; \
            column[k] = x; \
            for (ptrdiff_t r = k + 1; r < TS_BLAS_MIN(TS_BLAS_MR_##suffix, next * (width)); ++r) \
                column[r] += lower[r] * x; \
            vector factor = splat(x); \
            for (ptrdiff_t v = next; v < (vectors); ++v) \
                store(column + v * (width), muladd(load(column + v * (width)), load(lower + v * (width)), factor)); \
        } \
    } \
    for (ptrdiff_t j = 0; j < live_columns; ++j) \
        for (ptrdiff_t r = 0; r < live_rows; ++r) { \
            solved[r * TS_BLAS_NR_##suffix + j] = tile[j * TS_BLAS_MR_##suffix + r]; \
            target[r * rs + j] = tile[j * TS_BLAS_MR_##suffix + r]; \
        } \
} \
\
static void ts_blas_trsm_##suffix(int upper, int unit, int64_t rows, int64_t cols, \
                                  const type *a, int64_t ars, int64_t acs, type *w, type *work) { \
    int64_t wrs = cols; \
    if (upper) { \
        a += (rows - 1) * (ars + acs); \
        ars = -ars; \
        acs = -acs; \
        w += (rows - 1) * cols; \
        wrs = -cols; \
    } \
    type *packed_triangle = work; \
    type *scratch = work + ts_blas_triangle_size_##suffix(rows); \
    for (int64_t i0 = 0; i0 < rows; i0 += TS_BLAS_TRSM_BLOCK_##suffix) { \
        int64_t kc = TS_BLAS_MIN(TS_BLAS_TRSM_BLOCK_##suffix, rows - i0); \
        ts_blas_pack_triangle_##suffix(unit, kc, a + i0 * ars + i0 * acs, ars, acs, packed_triangle); \
        for (int64_t jc = 0; jc < cols; jc += TS_BLAS_COLUMN_BLOCK) { \
            int64_t nc = TS_BLAS_MIN(TS_BLAS_COLUMN_BLOCK, cols - jc); \
            ts_blas_pack_b_##suffix(nc, kc, w + i0 * wrs + jc, wrs, 1, scratch); \
            for (int64_t jr = 0; jr < nc; jr += TS_BLAS_NR_##suffix) { \
                const type *panel = packed_triangle; \
                for (int64_t r0 = 0; r0 < kc; r0 += TS_BLAS_MR_##suffix) { \
                    ts_blas_gemmtrsm_##suffix(r0, panel, scratch + jr * kc, \
                                              TS_BLAS_MIN(TS_BLAS_MR_##suffix, kc - r0), \
                                              TS_BLAS_MIN(TS_BLAS_NR_##suffix, nc - jr), \
                                              w + (i0 + r0) * wrs + jc + jr, wrs); \
                    panel += TS_BLAS_MR_##suffix * (r0 + TS_BLAS_MR_##suffix + 1); \
                } \
            } \
        } \
        if (rows - i0 - kc > 0) \
            ts_blas_gemm_run_##suffix(rows - i0 - kc, cols, kc, -1, \
                                      a + (i0 + kc) * ars + i0 * acs, ars, acs, \
                                      w + i0 * wrs, wrs, 1, 1, w + (i0 + kc) * wrs, wrs, 1, scratch, TS_BLAS_FULL); \
    } \
}

/* syrk and syr2k are one gemm that skips tiles outside the triangle. trsm uses the
   fused kernel above; trmm walks diagonal blocks, then updates the remaining rows
   with one gemm each. */
#define TS_BLAS_LEVEL3(suffix, type) \
static void ts_blas_scale_triangle_##suffix(int64_t n, type beta, type *c, int64_t rs, \
                                            int64_t cs, int upper) { \
    if (beta == 1) return; \
    for (int64_t i = 0; i < n; ++i) \
        for (int64_t j = upper ? i : 0; j < (upper ? n : i + 1); ++j) { \
            type *item = c + i * rs + j * cs; \
            *item = beta == 0 ? 0 : beta * *item; \
        } \
} \
\
int ts_blas_rank_##suffix(int64_t n, int64_t k, type alpha, \
                          const type *a, int64_t ars, int64_t acs, \
                          const type *b, int64_t brs, int64_t bcs, type beta, \
                          type *c, int64_t crs, int64_t ccs, int upper, int second) { \
    if (n <= 0) return 0; \
    if (k <= 0 || alpha == 0) { \
        ts_blas_scale_triangle_##suffix(n, beta, c, crs, ccs, upper); \
        return 0; \
    } \
    type *work = malloc(ts_blas_gemm_size_##suffix(n, n, k) * sizeof(type)); \
    if (!work) return 1; \
    int part = upper ? TS_BLAS_UPPER : TS_BLAS_LOWER; \
    ts_blas_scale_triangle_##suffix(n, beta, c, crs, ccs, upper); \
    ts_blas_gemm_run_##suffix(n, n, k, alpha, a, ars, acs, b, bcs, brs, 1, c, crs, ccs, work, part); \
    if (second) \
        ts_blas_gemm_run_##suffix(n, n, k, alpha, b, brs, bcs, a, acs, ars, 1, c, crs, ccs, work, part); \
    free(work); \
    return 0; \
} \
\
static void ts_blas_multiply_diagonal_##suffix(int upper, int unit, const type *a, \
                                               int64_t rs, int64_t cs, type *w, int64_t cols, \
                                               int64_t i0, int64_t i1) { \
    for (int64_t step = 0; step < i1 - i0; ++step) { \
        int64_t i = upper ? i0 + step : i1 - 1 - step; \
        type *row = w + i * cols; \
        type diagonal = a[i * rs + i * cs]; \
        if (!unit) \
            for (int64_t c = 0; c < cols; ++c) row[c] = diagonal * row[c]; \
        for (int64_t k = upper ? i + 1 : i0; k < (upper ? i1 : i); ++k) { \
            type scale = a[i * rs + k * cs]; \
            const type *other = w + k * cols; \
            for (int64_t c = 0; c < cols; ++c) row[c] += scale * other[c]; \
        } \
    } \
} \
\
int ts_blas_triangular_##suffix(int solve, int left, int upper, int unit, type alpha, \
                                const type *a, int64_t ars, int64_t acs, \
                                type *b, int64_t brs, int64_t bcs, int64_t m, int64_t n) { \
    int64_t rows = left ? m : n, cols = left ? n : m; \
    int64_t rs = left ? brs : bcs, cs = left ? bcs : brs; \
    int64_t tars = left ? ars : acs, tacs = left ? acs : ars; \
    int effective_upper = left ? upper : !upper; \
    if (rows <= 0 || cols <= 0) return 0; \
    type *w = malloc(((size_t)(rows * cols) + (solve ? ts_blas_trsm_size_##suffix(rows, cols) \
                          : ts_blas_gemm_size_##suffix(rows, cols, TS_BLAS_TRIANGLE_BLOCK))) \
                     * sizeof(type)); \
    if (!w) return 1; \
    type *work = w + rows * cols; \
    for (int64_t r = 0; r < rows; ++r) \
        for (int64_t c = 0; c < cols; ++c) \
            w[r * cols + c] = alpha == 0 ? 0 : alpha * b[r * rs + c * cs]; \
    if (alpha != 0 && solve) \
        ts_blas_trsm_##suffix(effective_upper, unit, rows, cols, a, tars, tacs, w, work); \
    else if (alpha != 0) { \
        for (int64_t step = 0; step < rows; step += TS_BLAS_TRIANGLE_BLOCK) { \
            int64_t i0 = effective_upper ? step : TS_BLAS_MAX(0, rows - step - TS_BLAS_TRIANGLE_BLOCK); \
            int64_t i1 = effective_upper ? TS_BLAS_MIN(rows, step + TS_BLAS_TRIANGLE_BLOCK) : rows - step; \
            int64_t first = effective_upper ? 0 : i1; \
            int64_t count = effective_upper ? i0 : rows - i1; \
            if (count > 0) \
                ts_blas_gemm_run_##suffix(count, cols, i1 - i0, 1, \
                                          a + first * tars + i0 * tacs, tars, tacs, \
                                          w + i0 * cols, cols, 1, 1, \
                                          w + first * cols, cols, 1, work, TS_BLAS_FULL); \
            ts_blas_multiply_diagonal_##suffix(effective_upper, unit, a, tars, tacs, w, cols, i0, i1); \
        } \
    } \
    for (int64_t r = 0; r < rows; ++r) \
        for (int64_t c = 0; c < cols; ++c) b[r * rs + c * cs] = w[r * cols + c]; \
    free(w); \
    return 0; \
}


/* Level 2 routines take the operated matrix as base pointer plus strides, and
   vectors with arbitrary increments. One stride of the matrix is always 1. */
#define TS_BLAS_LEVEL2(suffix, type, vector, width, load, store, zero, splat, muladd) \
static type ts_blas_dot_##suffix(int64_t n, const type *a, const type *x, int64_t incx) { \
    type sum = 0; \
    int64_t i = 0; \
    if (incx == 1) { \
        vector s0 = zero(), s1 = zero(), s2 = zero(), s3 = zero(); \
        type lanes[4 * (width)]; \
        for (; i + 4 * (width) <= n; i += 4 * (width)) { \
            s0 = muladd(s0, load(a + i), load(x + i)); \
            s1 = muladd(s1, load(a + i + (width)), load(x + i + (width))); \
            s2 = muladd(s2, load(a + i + 2 * (width)), load(x + i + 2 * (width))); \
            s3 = muladd(s3, load(a + i + 3 * (width)), load(x + i + 3 * (width))); \
        } \
        for (; i + (width) <= n; i += (width)) s0 = muladd(s0, load(a + i), load(x + i)); \
        store(lanes, s0); store(lanes + (width), s1); \
        store(lanes + 2 * (width), s2); store(lanes + 3 * (width), s3); \
        for (int lane = 0; lane < 4 * (width); ++lane) sum += lanes[lane]; \
        for (; i < n; ++i) sum += a[i] * x[i]; \
    } else { \
        for (; i < n; ++i) sum += a[i] * x[i * incx]; \
    } \
    return sum; \
} \
\
static void ts_blas_axpy_##suffix(int64_t n, type scale, const type *src, int64_t sinc, \
                                  type *dst, int64_t dinc) { \
    int64_t i = 0; \
    if (sinc == 1 && dinc == 1) { \
        vector factor = splat(scale); \
        for (; i + (width) <= n; i += (width)) \
            store(dst + i, muladd(load(dst + i), load(src + i), factor)); \
        for (; i < n; ++i) dst[i] += scale * src[i]; \
    } else { \
        for (; i < n; ++i) dst[i * dinc] += scale * src[i * sinc]; \
    } \
} \
\
int ts_blas_gemv_##suffix(int64_t m, int64_t n, type alpha, const type *a, int64_t ars, \
                          int64_t acs, const type *x, int64_t incx, type beta, \
                          type *y, int64_t incy) { \
    if (m <= 0) return 0; \
    if (alpha == 0 && beta == 1) return 0; \
    if (beta != 1) \
        for (int64_t i = 0; i < m; ++i) y[i * incy] = beta == 0 ? 0 : beta * y[i * incy]; \
    if (alpha == 0 || n <= 0) return 0; \
    if (acs == 1) { \
        for (int64_t i = 0; i < m; ++i) \
            y[i * incy] += alpha * ts_blas_dot_##suffix(n, a + i * ars, x, incx); \
    } else { \
        for (int64_t j = 0; j < n; ++j) \
            ts_blas_axpy_##suffix(m, alpha * x[j * incx], a + j * acs, ars, y, incy); \
    } \
    return 0; \
} \
\
int ts_blas_ger_##suffix(int64_t m, int64_t n, type alpha, const type *x, int64_t incx, \
                         const type *y, int64_t incy, type *a, int64_t ars, int64_t acs) { \
    if (m <= 0 || n <= 0 || alpha == 0) return 0; \
    if (acs == 1) { \
        for (int64_t i = 0; i < m; ++i) \
            ts_blas_axpy_##suffix(n, alpha * x[i * incx], y, incy, a + i * ars, 1); \
    } else { \
        for (int64_t j = 0; j < n; ++j) \
            ts_blas_axpy_##suffix(m, alpha * y[j * incy], x, incx, a + j * acs, ars); \
    } \
    return 0; \
} \
\
int ts_blas_trsv_##suffix(int upper, int unit, const type *a, int64_t ars, int64_t acs, \
                          int64_t n, type *x, int64_t incx) { \
    if (n <= 0) return 0; \
    type *w = malloc((size_t)n * sizeof(type)); \
    if (!w) return 1; \
    for (int64_t i = 0; i < n; ++i) w[i] = x[i * incx]; \
    if (acs != 1) { \
        for (int64_t s = 0; s < n; ++s) { \
            int64_t i = upper ? n - 1 - s : s; \
            if (!unit) w[i] /= a[i * ars + i * acs]; \
            int64_t from = upper ? 0 : i + 1; \
            ts_blas_axpy_##suffix(upper ? i : n - i - 1, -w[i], a + from * ars + i * acs, ars, \
                                  w + from, 1); \
        } \
    } \
    for (int64_t step = 0; step < (acs == 1 ? n : 0); step += TS_BLAS_BLOCK) { \
        int64_t i0 = upper ? TS_BLAS_MAX(0, n - step - TS_BLAS_BLOCK) : step; \
        int64_t i1 = upper ? n - step : TS_BLAS_MIN(n, step + TS_BLAS_BLOCK); \
        int64_t first = upper ? i1 : 0, count = upper ? n - i1 : i0; \
        if (count > 0) \
            ts_blas_gemv_##suffix(i1 - i0, count, -1, a + i0 * ars + first * acs, ars, acs, \
                                  w + first, 1, 1, w + i0, 1); \
        for (int64_t s = 0; s < i1 - i0; ++s) { \
            int64_t i = upper ? i1 - 1 - s : i0 + s; \
            type sum = 0; \
            for (int64_t k = upper ? i + 1 : i0; k < (upper ? i1 : i); ++k) \
                sum += a[i * ars + k * acs] * w[k]; \
            w[i] -= sum; \
            if (!unit) w[i] /= a[i * ars + i * acs]; \
        } \
    } \
    for (int64_t i = 0; i < n; ++i) x[i * incx] = w[i]; \
    free(w); \
    return 0; \
}

#define TS_BLAS_KERNELS(suffix, type, vector, width, load, store, zero, splat, muladd, rows, columns) \
    TS_BLAS_GEMM(suffix, type, vector, width, load, store, zero, splat, muladd, rows, columns) \
    TS_BLAS_TRSM(suffix, type, vector, width, load, store, splat, muladd, rows) \
    TS_BLAS_LEVEL3(suffix, type) \
    TS_BLAS_LEVEL2(suffix, type, vector, width, load, store, zero, splat, muladd)

#ifdef TS_BLAS_DISPATCH
TS_BLAS_KERNELS(f32_sse, float, TS_BLAS_F32_VECTOR, TS_BLAS_F32_WIDTH, TS_BLAS_F32_LOAD,
                TS_BLAS_F32_STORE, TS_BLAS_F32_ZERO, TS_BLAS_F32_SPLAT, TS_BLAS_F32_MULADD,
                TS_BLAS_F32_ROWS, TS_BLAS_F32_COLUMNS)
TS_BLAS_KERNELS(f64_sse, double, TS_BLAS_F64_VECTOR, TS_BLAS_F64_WIDTH, TS_BLAS_F64_LOAD,
                TS_BLAS_F64_STORE, TS_BLAS_F64_ZERO, TS_BLAS_F64_SPLAT, TS_BLAS_F64_MULADD,
                TS_BLAS_F64_ROWS, TS_BLAS_F64_COLUMNS)

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,fma"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx,fma")
#endif
TS_BLAS_KERNELS(f32_fma, float, TS_BLAS_AVX_F32_VECTOR, TS_BLAS_AVX_F32_WIDTH, TS_BLAS_AVX_F32_LOAD,
                TS_BLAS_AVX_F32_STORE, TS_BLAS_AVX_F32_ZERO, TS_BLAS_AVX_F32_SPLAT,
                TS_BLAS_AVX_F32_MULADD, TS_BLAS_AVX_F32_ROWS, TS_BLAS_AVX_F32_COLUMNS)
TS_BLAS_KERNELS(f64_fma, double, TS_BLAS_AVX_F64_VECTOR, TS_BLAS_AVX_F64_WIDTH, TS_BLAS_AVX_F64_LOAD,
                TS_BLAS_AVX_F64_STORE, TS_BLAS_AVX_F64_ZERO, TS_BLAS_AVX_F64_SPLAT,
                TS_BLAS_AVX_F64_MULADD, TS_BLAS_AVX_F64_ROWS, TS_BLAS_AVX_F64_COLUMNS)
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

/* The exported symbols pick AVX+FMA when the CPU and OS support it, else SSE2. */
#define TS_BLAS_PICK(name, suffix, ...) \
    (ts_fma_supported() ? ts_blas_##name##_##suffix##_fma : ts_blas_##name##_##suffix##_sse)(__VA_ARGS__)

#define TS_BLAS_FORWARD(suffix, type) \
int ts_blas_gemm_##suffix(int64_t m, int64_t n, int64_t k, type alpha, \
                          const type *a, int64_t ars, int64_t acs, \
                          const type *b, int64_t brs, int64_t bcs, type beta, \
                          type *c, int64_t crs, int64_t ccs) { \
    return TS_BLAS_PICK(gemm, suffix, m, n, k, alpha, a, ars, acs, b, brs, bcs, beta, c, crs, ccs); \
} \
\
int ts_blas_gemm_batch_##suffix(int64_t m, int64_t n, int64_t k, type alpha, \
                                const type *a, int64_t ars, int64_t acs, int64_t astep, \
                                const type *b, int64_t brs, int64_t bcs, int64_t bstep, type beta, \
                                type *c, int64_t crs, int64_t ccs, int64_t cstep, int64_t count) { \
    return TS_BLAS_PICK(gemm_batch, suffix, m, n, k, alpha, a, ars, acs, astep, \
                        b, brs, bcs, bstep, beta, c, crs, ccs, cstep, count); \
} \
\
int ts_blas_rank_##suffix(int64_t n, int64_t k, type alpha, \
                          const type *a, int64_t ars, int64_t acs, \
                          const type *b, int64_t brs, int64_t bcs, type beta, \
                          type *c, int64_t crs, int64_t ccs, int upper, int second) { \
    return TS_BLAS_PICK(rank, suffix, n, k, alpha, a, ars, acs, b, brs, bcs, beta, c, crs, ccs, \
                        upper, second); \
} \
\
int ts_blas_triangular_##suffix(int solve, int left, int upper, int unit, type alpha, \
                                const type *a, int64_t ars, int64_t acs, \
                                type *b, int64_t brs, int64_t bcs, int64_t m, int64_t n) { \
    return TS_BLAS_PICK(triangular, suffix, solve, left, upper, unit, alpha, a, ars, acs, \
                        b, brs, bcs, m, n); \
} \
\
int ts_blas_gemv_##suffix(int64_t m, int64_t n, type alpha, const type *a, int64_t ars, \
                          int64_t acs, const type *x, int64_t incx, type beta, \
                          type *y, int64_t incy) { \
    return TS_BLAS_PICK(gemv, suffix, m, n, alpha, a, ars, acs, x, incx, beta, y, incy); \
} \
\
int ts_blas_ger_##suffix(int64_t m, int64_t n, type alpha, const type *x, int64_t incx, \
                         const type *y, int64_t incy, type *a, int64_t ars, int64_t acs) { \
    return TS_BLAS_PICK(ger, suffix, m, n, alpha, x, incx, y, incy, a, ars, acs); \
} \
\
int ts_blas_trsv_##suffix(int upper, int unit, const type *a, int64_t ars, int64_t acs, \
                          int64_t n, type *x, int64_t incx) { \
    return TS_BLAS_PICK(trsv, suffix, upper, unit, a, ars, acs, n, x, incx); \
}

TS_BLAS_FORWARD(f32, float)
TS_BLAS_FORWARD(f64, double)
#else
TS_BLAS_KERNELS(f32, float, TS_BLAS_F32_VECTOR, TS_BLAS_F32_WIDTH, TS_BLAS_F32_LOAD,
                TS_BLAS_F32_STORE, TS_BLAS_F32_ZERO, TS_BLAS_F32_SPLAT, TS_BLAS_F32_MULADD,
                TS_BLAS_F32_ROWS, TS_BLAS_F32_COLUMNS)
TS_BLAS_KERNELS(f64, double, TS_BLAS_F64_VECTOR, TS_BLAS_F64_WIDTH, TS_BLAS_F64_LOAD,
                TS_BLAS_F64_STORE, TS_BLAS_F64_ZERO, TS_BLAS_F64_SPLAT, TS_BLAS_F64_MULADD,
                TS_BLAS_F64_ROWS, TS_BLAS_F64_COLUMNS)
#endif
