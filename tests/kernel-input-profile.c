#include <assert.h>
#include <stdio.h>
#include <time.h>

#ifndef TS_PROFILE_VM
#define TS_PROFILE_VM "../native/trivial_simd.c"
#endif
#include TS_PROFILE_VM

static volatile double profile_result;

#define PROFILE_INPUTS(suffix, type, precision) \
static void (*volatile prepare_##suffix)(const ts_kernel_input *, size_t, size_t, size_t, \
                                        size_t, type *, const type **) = ts_prepare_inputs_##suffix; \
static double profile_##suffix(size_t source, size_t rows, size_t n, size_t phase, unsigned mode) { \
    size_t count = rows * n, blocks = (phase + n + 31) / 32; \
    type *x = malloc(n * sizeof(type)), *scales = malloc(rows * blocks * sizeof(type)); \
    type *q_float = malloc(count * sizeof(type)), *scale_float = malloc(count * sizeof(type)); \
    type *out = malloc(rows * sizeof(type)); \
    void *q = malloc((count + 5) * ts_input_size(source)); \
    assert(x && scales && q_float && scale_float && out && q); \
    for (size_t i = 0; i < n; ++i) x[i] = (type)((int)(i % 17) - 8) / 16; \
    for (size_t i = 0; i < rows * blocks; ++i) scales[i] = (type)((int)(i % 7) - 3) / 4; \
    for (size_t i = 0; i < count + 5; ++i) { \
        int value = (int)(i % 127) - 63; \
        switch (source) { \
        case 2: ((int8_t *)q)[i] = (int8_t)value; break; \
        case 5: ((uint16_t *)q)[i] = (uint16_t)(value + 63); break; \
        case 6: ((int32_t *)q)[i] = value; break; \
        case 8: ((int64_t *)q)[i] = value; break; \
        case 9: ((uint64_t *)q)[i] = (uint64_t)(value + 63); break; \
        default: abort(); \
        } \
    } \
    ts_kernel_input inputs[] = {{x, n, 0, precision, 1, 0, 0}, \
        {q, count + 5, 5, source, 1, 0, (int64_t)n}, \
        {scales, rows * blocks, 0, precision, 32, phase, (int64_t)blocks}}; \
    uint8_t code[] = {TS_OP_MULTIPLY, 0, 9, 10, TS_OP_MULTIPLY, TS_KERNEL_OUTPUT, 8, 0}; \
    type buffers[3 * TS_KERNEL_BLOCK]; \
    const type *prepared[3]; \
    for (size_t row = 0; row < rows; ++row) { \
        for (size_t base = 0; base < n; base += TS_KERNEL_BLOCK) { \
            size_t m = n - base < TS_KERNEL_BLOCK ? n - base : TS_KERNEL_BLOCK; \
            prepare_##suffix(inputs, 3, row, base, m, buffers, prepared); \
            memcpy(q_float + row * n + base, prepared[1], m * sizeof(type)); \
            memcpy(scale_float + row * n + base, prepared[2], m * sizeof(type)); \
        } \
    } \
    const type *expanded[] = {x, q_float, scale_float}; \
    int64_t strides[] = {0, (int64_t)n, (int64_t)n}; \
    assert(ts_kernel_inputs_##suffix(code, sizeof(code), NULL, inputs, 3, out, rows, n, \
                                     0, TS_INPUT_SUM, 1, NULL, NULL) == 0); \
    type *reference = malloc(rows * sizeof(type)); \
    assert(reference); \
    memcpy(reference, out, rows * sizeof(type)); \
    size_t iterations = 1; \
    double times[3]; \
    for (int batch = -1; batch < 3;) { \
        clock_t start = clock(); \
        for (size_t iteration = 0; iteration < iterations; ++iteration) { \
            if (mode == 0) { \
                for (size_t row = 0; row < rows; ++row) { \
                    for (size_t base = 0; base < n; base += TS_KERNEL_BLOCK) { \
                        size_t m = n - base < TS_KERNEL_BLOCK ? n - base : TS_KERNEL_BLOCK; \
                        prepare_##suffix(inputs, 3, row, base, m, buffers, prepared); \
                    } \
                } \
                profile_result = prepared[1][0] + prepared[2][0]; \
            } else { \
                int status = mode == 1 \
                    ? ts_kernel_sum_rows_##suffix(code, sizeof(code), NULL, expanded, 3, strides, out, rows, n, 0) \
                    : ts_kernel_inputs_##suffix(code, sizeof(code), NULL, inputs, 3, out, rows, n, \
                                               0, TS_INPUT_SUM, 1, NULL, NULL); \
                assert(status == 0); \
                profile_result = out[0]; \
            } \
        } \
        double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC; \
        if (batch == -1 && elapsed < 0.05) { iterations *= 2; continue; } \
        if (batch >= 0) times[batch] = elapsed * 1e6 / iterations; \
        ++batch; \
    } \
    if (mode != 0) assert(memcmp(out, reference, rows * sizeof(type)) == 0); \
    free(x); free(scales); free(q_float); free(scale_float); free(q); free(out); free(reference); \
    if (times[0] > times[1]) { double t = times[0]; times[0] = times[1]; times[1] = t; } \
    if (times[1] > times[2]) times[1] = times[2]; \
    return times[0] > times[1] ? times[0] : times[1]; \
}

PROFILE_INPUTS(f32, float, 0)
PROFILE_INPUTS(f64, double, 1)

int main(void) {
    const char *modes[] = {"PREPARE", "VM", "COMPLETE"};
    size_t sources[] = {2, 5, 6, 8, 9};
    puts("PROFILE columns: precision source rows width phase mode microseconds");
    for (size_t source = 0; source < 5; ++source) {
        for (size_t shape = 0; shape < 2; ++shape) {
            size_t n = shape ? 257 : 1024, rows = shape ? 64 : 1024, phase = shape ? 5 : 0;
            for (unsigned mode = 0; mode < 3; ++mode) {
                double us = source < 2 ? profile_f32(sources[source], rows, n, phase, mode)
                                      : profile_f64(sources[source], rows, n, phase, mode);
                printf("PROFILE %s %zu %zu %zu %zu %s %.6f\n", source < 2 ? "f32" : "f64",
                       sources[source], rows, n, phase, modes[mode], us);
                fflush(stdout);
            }
        }
    }
}
