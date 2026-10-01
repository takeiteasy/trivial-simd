#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

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
#include <emmintrin.h>
#include <xmmintrin.h>
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

/* TODO: x86 uses SSE2 mul+add rather than FMA, blocking is tuned on M1 only, and
   every gemm call allocates its own packing buffers (#82). */
#define TS_BLAS_DEPTH 256
#define TS_BLAS_ROW_BLOCK 128
#define TS_BLAS_COLUMN_BLOCK 1024
#define TS_BLAS_MIN(a, b) ((a) < (b) ? (a) : (b))
#define TS_BLAS_MAX(a, b) ((a) > (b) ? (a) : (b))

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
int ts_blas_gemm_##suffix(int64_t m, int64_t n, int64_t k, type alpha, \
                          const type *a, int64_t ars, int64_t acs, \
                          const type *b, int64_t brs, int64_t bcs, type beta, \
                          type *c, int64_t crs, int64_t ccs) { \
    if (m <= 0 || n <= 0) return 0; \
    ts_blas_scale_##suffix(m, n, beta, c, crs, ccs); \
    if (k <= 0 || alpha == 0) return 0; \
    ptrdiff_t depth_block = TS_BLAS_MIN(k, TS_BLAS_DEPTH); \
    ptrdiff_t row_block = TS_BLAS_MIN((m + TS_BLAS_MR_##suffix - 1) / TS_BLAS_MR_##suffix \
                                          * TS_BLAS_MR_##suffix, TS_BLAS_ROW_BLOCK); \
    ptrdiff_t column_block = TS_BLAS_MIN((n + TS_BLAS_NR_##suffix - 1) / TS_BLAS_NR_##suffix \
                                             * TS_BLAS_NR_##suffix, TS_BLAS_COLUMN_BLOCK); \
    type *packed_a = malloc((size_t)(row_block * depth_block) * sizeof(type)); \
    type *packed_b = malloc((size_t)(depth_block * column_block) * sizeof(type)); \
    type tile[TS_BLAS_MR_##suffix * TS_BLAS_NR_##suffix]; \
    if (!packed_a || !packed_b) { free(packed_a); free(packed_b); return 1; } \
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
                        type *target = c + (ic + ir) * crs + (jc + jr) * ccs; \
                        ts_blas_micro_##suffix(kc, packed_a + ir * kc, packed_b + jr * kc, tile); \
                        for (ptrdiff_t j = 0; j < live_columns; ++j) \
                            for (ptrdiff_t i = 0; i < live_rows; ++i) \
                                target[i * crs + j * ccs] += tile[j * TS_BLAS_MR_##suffix + i]; \
                    } \
                } \
            } \
        } \
    } \
    free(packed_a); \
    free(packed_b); \
    return 0; \
}

TS_BLAS_GEMM(f32, float, TS_BLAS_F32_VECTOR, TS_BLAS_F32_WIDTH, TS_BLAS_F32_LOAD,
             TS_BLAS_F32_STORE, TS_BLAS_F32_ZERO, TS_BLAS_F32_SPLAT, TS_BLAS_F32_MULADD,
             TS_BLAS_F32_ROWS, TS_BLAS_F32_COLUMNS)
TS_BLAS_GEMM(f64, double, TS_BLAS_F64_VECTOR, TS_BLAS_F64_WIDTH, TS_BLAS_F64_LOAD,
             TS_BLAS_F64_STORE, TS_BLAS_F64_ZERO, TS_BLAS_F64_SPLAT, TS_BLAS_F64_MULADD,
             TS_BLAS_F64_ROWS, TS_BLAS_F64_COLUMNS)

#define TS_BLAS_BLOCK 16

/* Level 3 routines split into diagonal blocks handled here and off-diagonal
   panels handed to gemm, so they share its packing and micro-kernel. */
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
    type *tile = malloc(TS_BLAS_BLOCK * TS_BLAS_BLOCK * sizeof(type)); \
    if (!tile) return 1; \
    int status = 0; \
    for (int64_t i0 = 0; i0 < n && !status; i0 += TS_BLAS_BLOCK) { \
        int64_t nb = TS_BLAS_MIN(TS_BLAS_BLOCK, n - i0); \
        status = ts_blas_gemm_##suffix(nb, nb, k, alpha, a + i0 * ars, ars, acs, \
                                       b + i0 * brs, bcs, brs, 0, tile, nb, 1); \
        if (!status && second) \
            status = ts_blas_gemm_##suffix(nb, nb, k, alpha, b + i0 * brs, brs, bcs, \
                                           a + i0 * ars, acs, ars, 1, tile, nb, 1); \
        if (status) break; \
        for (int64_t i = 0; i < nb; ++i) \
            for (int64_t j = upper ? i : 0; j < (upper ? nb : i + 1); ++j) { \
                type *item = c + (i0 + i) * crs + (i0 + j) * ccs; \
                *item = beta == 0 ? tile[i * nb + j] : tile[i * nb + j] + beta * *item; \
            } \
        int64_t first = upper ? i0 + nb : 0; \
        int64_t count = upper ? n - first : i0; \
        if (count <= 0) continue; \
        type *panel = c + i0 * crs + first * ccs; \
        status = ts_blas_gemm_##suffix(nb, count, k, alpha, a + i0 * ars, ars, acs, \
                                       b + first * brs, bcs, brs, beta, panel, crs, ccs); \
        if (!status && second) \
            status = ts_blas_gemm_##suffix(nb, count, k, alpha, b + i0 * brs, brs, bcs, \
                                           a + first * ars, acs, ars, 1, panel, crs, ccs); \
    } \
    free(tile); \
    return status; \
} \
\
static void ts_blas_diagonal_##suffix(int solve, int upper, int unit, const type *a, \
                                      int64_t rs, int64_t cs, type *w, int64_t cols, \
                                      int64_t i0, int64_t i1) { \
    for (int64_t step = 0; step < i1 - i0; ++step) { \
        int64_t i = (solve ? upper : !upper) ? i1 - 1 - step : i0 + step; \
        type *row = w + i * cols; \
        type diagonal = a[i * rs + i * cs]; \
        if (!solve && !unit) \
            for (int64_t c = 0; c < cols; ++c) row[c] = diagonal * row[c]; \
        for (int64_t k = upper ? i + 1 : i0; k < (upper ? i1 : i); ++k) { \
            type scale = a[i * rs + k * cs]; \
            const type *other = w + k * cols; \
            if (solve) for (int64_t c = 0; c < cols; ++c) row[c] -= scale * other[c]; \
            else for (int64_t c = 0; c < cols; ++c) row[c] += scale * other[c]; \
        } \
        if (solve && !unit) \
            for (int64_t c = 0; c < cols; ++c) row[c] /= diagonal; \
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
    type *w = malloc((size_t)(rows * cols) * sizeof(type)); \
    if (!w) return 1; \
    int status = 0; \
    for (int64_t r = 0; r < rows; ++r) \
        for (int64_t c = 0; c < cols; ++c) \
            w[r * cols + c] = alpha == 0 ? 0 : alpha * b[r * rs + c * cs]; \
    if (alpha != 0) { \
        int forward = solve ? !effective_upper : effective_upper; \
        for (int64_t step = 0; step < rows && !status; step += TS_BLAS_BLOCK) { \
            int64_t i0 = forward ? step : TS_BLAS_MAX(0, rows - step - TS_BLAS_BLOCK); \
            int64_t i1 = forward ? TS_BLAS_MIN(rows, step + TS_BLAS_BLOCK) : rows - step; \
            int64_t first = effective_upper ? i1 : 0; \
            int64_t count = effective_upper ? rows - i1 : i0; \
            if (solve) { \
                if (count > 0) \
                    status = ts_blas_gemm_##suffix(i1 - i0, cols, count, -1, \
                                                   a + i0 * tars + first * tacs, tars, tacs, \
                                                   w + first * cols, cols, 1, 1, \
                                                   w + i0 * cols, cols, 1); \
                ts_blas_diagonal_##suffix(1, effective_upper, unit, a, tars, tacs, w, cols, i0, i1); \
            } else { \
                ts_blas_diagonal_##suffix(0, effective_upper, unit, a, tars, tacs, w, cols, i0, i1); \
                if (count > 0) \
                    status = ts_blas_gemm_##suffix(i1 - i0, cols, count, 1, \
                                                   a + i0 * tars + first * tacs, tars, tacs, \
                                                   w + first * cols, cols, 1, 1, \
                                                   w + i0 * cols, cols, 1); \
            } \
        } \
    } \
    if (!status) \
        for (int64_t r = 0; r < rows; ++r) \
            for (int64_t c = 0; c < cols; ++c) b[r * rs + c * cs] = w[r * cols + c]; \
    free(w); \
    return status; \
}

TS_BLAS_LEVEL3(f32, float)
TS_BLAS_LEVEL3(f64, double)

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
    for (int64_t step = 0; step < n; step += TS_BLAS_BLOCK) { \
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

TS_BLAS_LEVEL2(f32, float, TS_BLAS_F32_VECTOR, TS_BLAS_F32_WIDTH, TS_BLAS_F32_LOAD,
               TS_BLAS_F32_STORE, TS_BLAS_F32_ZERO, TS_BLAS_F32_SPLAT, TS_BLAS_F32_MULADD)
TS_BLAS_LEVEL2(f64, double, TS_BLAS_F64_VECTOR, TS_BLAS_F64_WIDTH, TS_BLAS_F64_LOAD,
               TS_BLAS_F64_STORE, TS_BLAS_F64_ZERO, TS_BLAS_F64_SPLAT, TS_BLAS_F64_MULADD)
