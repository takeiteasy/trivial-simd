#include <stdio.h>
#include <time.h>
#include "../native/blas.c"

static volatile double observed;

/* Median of three batches of at least 50 ms, in microseconds per call. */
#define MEASURE(result, body) do { \
    size_t iterations = 1; \
    double times[3]; \
    for (int batch = -1; batch < 3;) { \
        clock_t start = clock(); \
        for (size_t it = 0; it < iterations; ++it) { body; } \
        double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC; \
        if (batch == -1 && elapsed < 0.05) { iterations *= 2; continue; } \
        if (batch >= 0) times[batch] = elapsed * 1e6 / iterations; \
        ++batch; \
    } \
    if (times[0] > times[1]) { double t = times[0]; times[0] = times[1]; times[1] = t; } \
    if (times[1] > times[2]) times[1] = times[2]; \
    (result) = times[0] > times[1] ? times[0] : times[1]; \
} while (0)

#define PROFILE(suffix, type) \
static void profile_##suffix(const char *variant, int64_t n) { \
    size_t count = (size_t)(n * n); \
    type *a = malloc(count * sizeof(type)), *b = malloc(count * sizeof(type)); \
    type *c = malloc(count * sizeof(type)), *x = malloc((size_t)n * sizeof(type)); \
    if (!a || !b || !c || !x) exit(1); \
    for (size_t i = 0; i < count; ++i) { \
        a[i] = (type)((i * 7 + 3) % 17 - 8) / 64 + (i / n == i % n ? 2 : 0); \
        b[i] = (type)((i * 5 + 1) % 13 - 6) / 64; \
    } \
    double us; \
    MEASURE(us, ts_blas_gemm_##suffix(n, n, n, 1, a, n, 1, b, n, 1, 0, c, n, 1); observed = c[0]); \
    printf("%-8s gemm  %5lld %10.2f us %8.2f GFLOP/s\n", variant, (long long)n, us, 2e-3 * n * n * n / us); \
    MEASURE(us, ts_blas_rank_##suffix(n, n, 1, a, n, 1, a, n, 1, 0, c, n, 1, 0, 0); observed = c[0]); \
    printf("%-8s syrk  %5lld %10.2f us\n", variant, (long long)n, us); \
    MEASURE(us, for (size_t i = 0; i < count; ++i) c[i] = b[i]; \
                ts_blas_triangular_##suffix(1, 1, 0, 0, 1, a, n, 1, c, n, 1, n, n); observed = c[0]); \
    printf("%-8s trsm  %5lld %10.2f us\n", variant, (long long)n, us); \
    for (int64_t i = 0; i < n; ++i) x[i] = 1; \
    MEASURE(us, for (int64_t i = 0; i < n; ++i) x[i] = 1; \
                ts_blas_trsv_##suffix(0, 0, a, n, 1, n, x, 1); observed = x[0]); \
    printf("%-8s trsv  %5lld %10.2f us\n", variant, (long long)n, us); \
    MEASURE(us, for (int64_t i = 0; i < n; ++i) x[i] = 1; \
                ts_blas_trsv_##suffix(0, 0, a, 1, n, n, x, 1); observed = x[0]); \
    printf("%-8s trsvc %5lld %10.2f us\n", variant, (long long)n, us); \
    MEASURE(us, ts_blas_gemv_##suffix(n, n, 1, a, n, 1, b, 1, 0, c, 1); observed = c[0]); \
    printf("%-8s gemv  %5lld %10.2f us\n", variant, (long long)n, us); \
    free(a); free(b); free(c); free(x); \
}

#ifdef TS_BLAS_DISPATCH
#define PROFILE_VARIANTS(suffix) \
    profile_##suffix##_sse("sse", n); \
    if (ts_fma_supported()) profile_##suffix##_fma("avx-fma", n);
PROFILE(f32_sse, float)
PROFILE(f64_sse, double)
PROFILE(f32_fma, float)
PROFILE(f64_fma, double)
#else
#define PROFILE_VARIANTS(suffix) profile_##suffix("native", n);
PROFILE(f32, float)
PROFILE(f64, double)
#endif

int main(void) {
    printf("Hardware FMA: %d\n", ts_fma_supported());
    printf("Blocking: depth %d rows %d columns %d\n", TS_BLAS_DEPTH, TS_BLAS_ROW_BLOCK,
           TS_BLAS_COLUMN_BLOCK);
#ifdef TS_BLAS_DISPATCH
    printf("AVX tiles: f32 %dx%d f64 %dx%d\n", TS_BLAS_AVX_F32_ROWS * TS_BLAS_AVX_F32_WIDTH,
           TS_BLAS_AVX_F32_COLUMNS, TS_BLAS_AVX_F64_ROWS * TS_BLAS_AVX_F64_WIDTH,
           TS_BLAS_AVX_F64_COLUMNS);
#endif
    int64_t sizes[] = {64, 256, 512};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        int64_t n = sizes[i];
        puts("f64");
        PROFILE_VARIANTS(f64)
        puts("f32");
        PROFILE_VARIANTS(f32)
    }
    return 0;
}
