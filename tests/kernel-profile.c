#include <time.h>

#ifdef TS_PROFILE_BASELINE
#define ts_kernel_f32 ts_baseline_kernel_f32
#define ts_kernel_f64 ts_baseline_kernel_f64
#define ts_kernel_sum_f32 ts_baseline_kernel_sum_f32
#define ts_kernel_sum_f64 ts_baseline_kernel_sum_f64
#define ts_profile_f32 ts_profile_baseline_f32
#define ts_profile_f64 ts_profile_baseline_f64
#endif

#ifndef TS_PROFILE_VM
#define TS_PROFILE_VM "../native/trivial_simd.c"
#endif
#include TS_PROFILE_VM

static volatile double profile_result;

#define PROFILE(suffix, type) \
double ts_profile_##suffix(const uint8_t *code, size_t code_length, \
                          const type *constants, const type *const *inputs, \
                          type *out, size_t n, size_t slots, int sum, int reuse) { \
    type *scratch = slots ? malloc(slots * TS_KERNEL_BLOCK * sizeof(type)) : NULL; \
    if (slots && !scratch) return -1; \
    size_t iterations = 1; \
    double timings[3]; \
    for (int batch = -1; batch < 3;) { \
        clock_t start = clock(); \
        for (size_t i = 0; i < iterations; ++i) { \
            int status = profile_call_##suffix(code, code_length, constants, inputs, out, n, slots, sum, reuse, scratch); \
            if (status) { free(scratch); return -1; } \
        } \
        double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC; \
        profile_result = out[0]; \
        if (batch == -1) { \
            if (elapsed < 0.05) { iterations *= 2; continue; } \
        } else timings[batch] = elapsed * 1e6 / iterations; \
        ++batch; \
    } \
    free(scratch); \
    if (timings[0] > timings[1]) { double t = timings[0]; timings[0] = timings[1]; timings[1] = t; } \
    if (timings[1] > timings[2]) timings[1] = timings[2]; \
    return timings[0] > timings[1] ? timings[0] : timings[1]; \
}

#define PROFILE_CALL(suffix, type) \
static int profile_call_##suffix(const uint8_t *code, size_t length, const type *constants, \
                                const type *const *inputs, type *out, size_t n, size_t slots, \
                                int sum, int reuse, type *scratch) { \
    if (reuse) return sum \
        ? ts_kernel_sum_with_scratch_##suffix(code, length, constants, inputs, out, n, slots, scratch, slots) \
        : ts_kernel_with_scratch_##suffix(code, length, constants, inputs, out, n, slots, scratch, slots); \
    return sum ? ts_kernel_sum_##suffix(code, length, constants, inputs, out, n, slots) \
               : ts_kernel_##suffix(code, length, constants, inputs, out, n, slots); \
}

#ifdef TS_PROFILE_BASELINE
#undef PROFILE_CALL
#define PROFILE_CALL(suffix, type) \
static int profile_call_##suffix(const uint8_t *code, size_t length, const type *constants, \
                                const type *const *inputs, type *out, size_t n, size_t slots, \
                                int sum, int reuse, type *scratch) { \
    (void)reuse; (void)scratch; \
    return sum ? ts_kernel_sum_##suffix(code, length, constants, inputs, out, n, slots) \
               : ts_kernel_##suffix(code, length, constants, inputs, out, n, slots); \
}
#endif

PROFILE_CALL(f32, float)
PROFILE_CALL(f64, double)
PROFILE(f32, float)
PROFILE(f64, double)
