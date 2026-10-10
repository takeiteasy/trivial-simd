#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../native/extended.c"

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif
static void *output;
static size_t length;
static volatile uint64_t observed;
static double timed(void (*run)(void)) {
    size_t iterations = 1;
    run();
    for (;;) {
        clock_t start = clock();
        for (size_t i = 0; i < iterations; ++i) run();
        if ((double)(clock() - start) / CLOCKS_PER_SEC >= 0.05) break;
        iterations *= 2;
    }
    double samples[3];
    for (size_t trial = 0; trial < 3; ++trial) {
        clock_t start = clock();
        for (size_t i = 0; i < iterations; ++i) run();
        samples[trial] = 1e6 * (double)(clock() - start) / CLOCKS_PER_SEC / iterations;
        observed += ((uint8_t *)output)[length - 1];
    }
    for (size_t i = 0; i < 3; ++i) for (size_t j = i + 1; j < 3; ++j)
        if (samples[i] > samples[j]) { double x = samples[i]; samples[i] = samples[j]; samples[j] = x; }
    return samples[1];
}
#define PROFILE_CONVERT(name, source, target, ds, is, value) \
static source input_##name[65536]; \
static target output_##name[65536], reference_##name[65536]; \
static NOINLINE void scalar_##name(void) { \
    for (size_t i = 0; i < length; ++i) { source x = input_##name[i]; reference_##name[i] = (target)(value); } \
    output = reference_##name; \
} \
static NOINLINE void packed_##name(void) { \
    assert(ts_extended_convert_numeric(output_##name, (const void *)input_##name, length, ds, is) == 0); \
    output = output_##name; \
} \
static void profile_##name(void) { \
    for (size_t i = 0; i < 65536; ++i) input_##name[i] = (source)(i * UINT64_C(1234567891234567)); \
    scalar_##name(); packed_##name(); \
    assert(memcmp(output_##name, reference_##name, length * sizeof(target)) == 0); \
    double scalar = timed(scalar_##name), packed = timed(packed_##name); \
    printf("convert,%s,%zu,%.6f,%.6f,%.3f\n", #name, length, scalar, packed, scalar / packed); \
}
#define PROFILE_FLOAT(name, source, tag) \
    PROFILE_CONVERT(name##_f32, source, float, 0, tag, x) \
    PROFILE_CONVERT(name##_f64, source, double, 1, tag, x)
PROFILE_FLOAT(s8, int8_t, 2)
PROFILE_FLOAT(u8, uint8_t, 3)
PROFILE_FLOAT(s16, int16_t, 4)
PROFILE_FLOAT(u16, uint16_t, 5)
PROFILE_FLOAT(s32, int32_t, 6)
PROFILE_FLOAT(u32, uint32_t, 7)
PROFILE_CONVERT(s64_f32, volatile int64_t, float, 0, 8, x)
PROFILE_CONVERT(s64_f64, int64_t, double, 1, 8, x)
PROFILE_CONVERT(u64_f32, volatile uint64_t, float, 0, 9, x)
PROFILE_CONVERT(u64_f64, uint64_t, double, 1, 9, x)
PROFILE_CONVERT(s16_s8, int16_t, int8_t, 2, 4, x < INT8_MIN ? INT8_MIN : x > INT8_MAX ? INT8_MAX : x)
PROFILE_CONVERT(u16_s8, uint16_t, int8_t, 2, 5, x > INT8_MAX ? INT8_MAX : x)
PROFILE_CONVERT(s16_u8, int16_t, uint8_t, 3, 4, x < 0 ? 0 : x > UINT8_MAX ? UINT8_MAX : x)
PROFILE_CONVERT(u16_u8, uint16_t, uint8_t, 3, 5, x > UINT8_MAX ? UINT8_MAX : x)
PROFILE_CONVERT(s32_s16, int32_t, int16_t, 4, 6, x < INT16_MIN ? INT16_MIN : x > INT16_MAX ? INT16_MAX : x)
PROFILE_CONVERT(u32_s16, uint32_t, int16_t, 4, 7, x > INT16_MAX ? INT16_MAX : x)
PROFILE_CONVERT(s32_u16, int32_t, uint16_t, 5, 6, x < 0 ? 0 : x > UINT16_MAX ? UINT16_MAX : x)
PROFILE_CONVERT(u32_u16, uint32_t, uint16_t, 5, 7, x > UINT16_MAX ? UINT16_MAX : x)

#define PROFILE_MASK(name, type) \
static type a_##name[65536], b_##name[65536]; \
static uint8_t mask_##name[65536], reference_mask_##name[65536]; \
static NOINLINE void scalar_mask_##name(void) { \
    for (size_t i = 0; i < length; ++i) reference_mask_##name[i] = a_##name[i] > b_##name[i]; \
    output = reference_mask_##name; \
} \
static NOINLINE void packed_mask_##name(void) { \
    assert(ts_extended_compare_##name(4, mask_##name, a_##name, b_##name, 0, 0, length) == 0); \
    output = mask_##name; \
} \
static void profile_mask_##name(void) { \
    for (size_t i = 0; i < 65536; ++i) { a_##name[i] = (type)(i * 13 % 251); b_##name[i] = (type)(i * 17 % 251); } \
    scalar_mask_##name(); packed_mask_##name(); assert(memcmp(mask_##name, reference_mask_##name, length) == 0); \
    double scalar = timed(scalar_mask_##name), packed = timed(packed_mask_##name); \
    printf("compare,%s,%zu,%.6f,%.6f,%.3f\n", #name, length, scalar, packed, scalar / packed); \
}
PROFILE_MASK(f32, float)
PROFILE_MASK(f64, double)
PROFILE_MASK(s8, int8_t)
PROFILE_MASK(u8, uint8_t)
PROFILE_MASK(s16, int16_t)
PROFILE_MASK(u16, uint16_t)
PROFILE_MASK(s32, int32_t)
PROFILE_MASK(u32, uint32_t)
PROFILE_MASK(s64, int64_t)
PROFILE_MASK(u64, uint64_t)

static double encoded_input[65536];
static uint16_t encoded_output[65536], encoded_reference[65536];
static int half_format, mode;
static NOINLINE void scalar_encoding(void) {
    for (size_t i = 0; i < length; ++i) { uint64_t bits; memcpy(&bits, encoded_input + i, sizeof bits);
        encoded_reference[i] = ts_f64_bits_to_encoded(bits, half_format, mode); }
    output = encoded_reference;
}
static NOINLINE void packed_encoding(void) {
    ts_extended_convert_encoded(encoded_output, encoded_input, length, half_format ? 3 : 2, 1, mode);
    output = encoded_output;
}

static double widened_output[65536], widened_reference[65536];
static NOINLINE void scalar_widening(void) {
    for (size_t i = 0; i < length; ++i) {
        float value = half_format ? ts_f16_to_f32_scalar(encoded_output[i]) : ts_bf16_to_f32_scalar(encoded_output[i]);
        uint64_t bits = ts_f32_bits_to_f64_bits(ts_float_bits(value));
        memcpy(widened_reference + i, &bits, sizeof bits);
    }
    output = widened_reference;
}
static NOINLINE void packed_widening(void) {
    ts_extended_convert_encoded(widened_output, encoded_output, length, 1, half_format ? 3 : 2, 0);
    output = widened_output;
}

int main(void) {
    puts("operation,pair,elements,scalar_us,candidate_us,speedup");
    size_t lengths[] = {32, 1024, 65536};
    for (size_t k = 0; k < 3; ++k) {
        length = lengths[k];
#define RUN_FLOAT(name) profile_##name##_f32(); profile_##name##_f64();
        RUN_FLOAT(s8) RUN_FLOAT(u8) RUN_FLOAT(s16) RUN_FLOAT(u16)
        RUN_FLOAT(s32) RUN_FLOAT(u32) RUN_FLOAT(s64) RUN_FLOAT(u64)
#define RUN_NARROW(name) profile_##name();
        RUN_NARROW(s16_s8) RUN_NARROW(u16_s8) RUN_NARROW(s16_u8) RUN_NARROW(u16_u8)
        RUN_NARROW(s32_s16) RUN_NARROW(u32_s16) RUN_NARROW(s32_u16) RUN_NARROW(u32_u16)
#define RUN_MASK(name) profile_mask_##name();
        RUN_MASK(f32) RUN_MASK(f64) RUN_MASK(s8) RUN_MASK(u8) RUN_MASK(s16) RUN_MASK(u16)
        RUN_MASK(s32) RUN_MASK(u32) RUN_MASK(s64) RUN_MASK(u64)
        for (size_t i = 0; i < 65536; ++i) encoded_input[i] = (double)((int)(i % 8192) - 4096) / 19.0;
        for (half_format = 0; half_format < 2; ++half_format) for (mode = 0; mode < 4; ++mode) {
            scalar_encoding(); packed_encoding(); assert(memcmp(encoded_output, encoded_reference, length * 2) == 0);
            double scalar = timed(scalar_encoding), packed = timed(packed_encoding);
            printf("encoded,f64_%s_mode%d,%zu,%.6f,%.6f,%.3f\n", half_format ? "f16" : "bf16", mode, length, scalar, packed, scalar / packed);
        }
        for (half_format = 0; half_format < 2; ++half_format) {
            for (size_t i = 0; i < length; ++i) encoded_output[i] = half_format
                ? ts_f64_bits_to_encoded(UINT64_C(0x3ff0000000000000) + ((uint64_t)(i % 1024) << 42), 1, 0)
                : (uint16_t)(0x3f80 + i % 128);
            scalar_widening(); packed_widening();
            assert(memcmp(widened_output, widened_reference, length * sizeof(double)) == 0);
            double scalar = timed(scalar_widening), packed = timed(packed_widening);
            printf("encoded,%s_f64,%zu,%.6f,%.6f,%.3f\n", half_format ? "f16" : "bf16", length, scalar, packed, scalar / packed);
        }
        fflush(stdout);
    }
    return observed == UINT64_MAX;
}
