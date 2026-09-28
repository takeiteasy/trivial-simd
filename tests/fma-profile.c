#include <stdio.h>
#include <time.h>
#include "../native/trivial_simd.c"

static volatile double observed;

#define PROFILE(suffix, type) \
static void profile_##suffix(size_t n) { \
    type *a = malloc(n * sizeof(type)), *b = malloc(n * sizeof(type)); \
    type *c = malloc(n * sizeof(type)), *out = malloc(n * sizeof(type)); \
    if (!a || !b || !c || !out) exit(1); \
    for (size_t i = 0; i < n; ++i) { a[i] = 1.25; b[i] = 2.5; c[i] = -0.5; } \
    const type *inputs[] = {a, b, c}; \
    uint8_t code[] = {TS_OP_COPY, 0, 10, 0, TS_OP_FMA, 0, 8, 9, TS_OP_COPY, TS_KERNEL_OUTPUT, 0, 0}; \
    size_t iterations = 1; \
    double times[3]; \
    for (int batch = -1; batch < 3;) { \
        clock_t start = clock(); \
        for (size_t i = 0; i < iterations; ++i) \
            if (ts_kernel_##suffix(code, sizeof(code), NULL, inputs, out, n, 0)) exit(1); \
        double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC; \
        for (size_t i = 0; i < n; ++i) if (out[i] != (type)2.625) exit(1); \
        observed = out[n - 1]; \
        if (batch == -1 && elapsed < 0.05) { iterations *= 2; continue; } \
        if (batch >= 0) times[batch] = elapsed * 1e6 / iterations; \
        ++batch; \
    } \
    if (times[0] > times[1]) { double t = times[0]; times[0] = times[1]; times[1] = t; } \
    if (times[1] > times[2]) times[1] = times[2]; \
    printf("%s %zu %.3f us\n", #suffix, n, times[0] > times[1] ? times[0] : times[1]); \
    free(a); free(b); free(c); free(out); \
}

PROFILE(f32, float)
PROFILE(f64, double)

int main(void) {
    printf("Hardware FMA: %d\n", ts_fma_supported());
    size_t lengths[] = {32, 1024, 65536};
    for (size_t i = 0; i < 3; ++i) { profile_f32(lengths[i]); profile_f64(lengths[i]); }
    return 0;
}
