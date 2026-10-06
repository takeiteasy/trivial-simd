#include <assert.h>
#include <math.h>
#include <stdlib.h>

static int allocations, releases, fail_allocation;
static void *batch_malloc(size_t size) {
    ++allocations;
    return fail_allocation ? NULL : malloc(size);
}
static void batch_free(void *pointer) {
    if (pointer) ++releases;
    free(pointer);
}
#define malloc batch_malloc
#define free batch_free
#include "../native/blas.c"
#undef malloc
#undef free

#define CHECK_BATCH(suffix, type, tolerance) \
static void check_##suffix(void) { \
    type a[48], b[48], c[48], expected[48]; \
    for (int i = 0; i < 48; ++i) { \
        a[i] = (type)((i * 7) % 17 - 8) / 11; \
        b[i] = (type)(i % 9) / 7; \
        c[i] = expected[i] = 7; \
    } \
    for (int batch = 0; batch < 3; ++batch) \
        for (int i = 0; i < 2; ++i) \
            for (int j = 0; j < 2; ++j) { \
                type sum = 0; \
                for (int k = 0; k < 3; ++k) \
                    sum += a[23 - 9 * batch - 3 * i - k] * b[4 + 9 * batch + 3 * k + j]; \
                expected[26 - 9 * batch - 4 * i - j] = 21 + 2 * sum; \
            } \
    allocations = releases = 0; \
    assert(ts_blas_gemm_batch_##suffix(2, 2, 3, 2, a + 23, -3, -1, -9, \
                                       b + 4, 3, 1, 9, 3, c + 26, -4, -1, -9, 3) == 0); \
    assert(allocations == 1 && releases == 1); \
    for (int i = 0; i < 48; ++i) \
        assert(fabs((double)c[i] - expected[i]) <= tolerance * fmax(1, fabs(expected[i]))); \
    fail_allocation = 1; \
    assert(ts_blas_gemm_batch_##suffix(2, 2, 3, 1, a, 3, 1, 6, b, 2, 1, 6, \
                                       0, c, 2, 1, 4, 3) == 1); \
    for (int i = 0; i < 48; ++i) assert(fabs((double)c[i] - expected[i]) <= tolerance * fmax(1, fabs(expected[i]))); \
    allocations = releases = 0; \
    assert(ts_blas_gemm_batch_##suffix(2, 2, 0, 1, a, 0, 1, 0, b, 2, 1, 0, \
                                       0, c, 2, 1, 4, 3) == 0); \
    assert(allocations == 0 && releases == 0); \
    for (int i = 0; i < 12; ++i) assert(c[i] == 0); \
    assert(ts_blas_gemm_batch_##suffix(2, 2, 3, 0, a, 3, 1, 0, b, 2, 1, 0, \
                                       2, c, 2, 1, 4, 3) == 0); \
    assert(ts_blas_gemm_batch_##suffix(2, 2, 3, 1, a, 3, 1, 0, b, 2, 1, 0, \
                                       0, c, 2, 1, 4, 0) == 0); \
    assert(allocations == 0); \
    fail_allocation = 0; \
}

CHECK_BATCH(f32, float, 1e-5)
CHECK_BATCH(f64, double, 1e-12)

int main(void) {
    check_f32();
    check_f64();
    return 0;
}
