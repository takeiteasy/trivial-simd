#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t allocations, releases;
static int fail_allocation;
static void *checked_malloc(size_t n) { ++allocations; return fail_allocation ? NULL : malloc(n); }
static void checked_free(void *p) { if (p) ++releases; free(p); }
#define malloc checked_malloc
#define free checked_free
#include "../native/trivial_simd.c"
#undef malloc
#undef free
#include "../native/extended.c"

#define CHECK_MASK(suffix, type, unsigned_type) \
static void check_mask_##suffix(void) { \
    type a[601], b[601], out[603]; uint8_t mask[603]; \
    const unsigned_type *inputs[] = {(const unsigned_type *)a, (const unsigned_type *)b}; \
    for (size_t i = 0; i < 601; ++i) { \
        a[i] = (type)((int)((i * 13) % 251) - 125); b[i] = (type)((int)((i * 17) % 251) - 125); \
    } \
    a[1] = (type)0; b[1] = (type)-0.0; \
    a[2] = (type)UINT64_C(0x800000007fffffff); b[2] = (type)UINT64_C(0x8000000080000000); \
    a[3] = (type)UINT64_C(0x8000000000000000); b[3] = (type)UINT64_C(0x7fffffffffffffff); \
    a[4] = (type)UINT64_C(0x7fffffff00000000); b[4] = (type)UINT64_C(0x8000000000000000); \
    for (size_t n = 0; n <= 601; n += n < 20 ? 1 : n < 255 ? 235 : n < 259 ? 1 : 342) { \
        for (unsigned op = 0; op < 6; ++op) { \
            memset(mask, 7, sizeof mask); \
            assert(ts_extended_compare_##suffix(op, mask + 1, a, b, 0, 0, n) == 0); \
            assert(mask[0] == 7 && mask[n + 1] == 7); \
            for (size_t i = 0; i < n; ++i) assert(mask[i + 1] == TS_COMPARE(op, a[i], b[i])); \
            uint8_t code[] = {(uint8_t)(TS_OP_EQ + op), TS_KERNEL_OUTPUT, 8, 9}; \
            assert(ts_kernel_mask_##suffix(code, sizeof code, NULL, inputs, 2, mask + 1, n, 0, 0, NULL) == 0); \
            for (size_t i = 0; i < n; ++i) assert(mask[i + 1] == TS_COMPARE(op, a[i], b[i])); \
            assert(ts_extended_compare_##suffix(op, mask + 1, NULL, b, a[0], 0, n) == 0); \
            for (size_t i = 0; i < n; ++i) assert(mask[i + 1] == TS_COMPARE(op, a[0], b[i])); \
        } \
        for (size_t i = 0; i < n; ++i) mask[i + 1] = (uint8_t)(i % 4 ? i % 255 + 1 : 0); \
        assert(ts_extended_select_##suffix(out + 1, mask + 1, a, b, 0, 0, n) == 0); \
        for (size_t i = 0; i < n; ++i) assert(memcmp(out + i + 1, mask[i + 1] ? a + i : b + i, sizeof(type)) == 0); \
        assert(ts_extended_select_##suffix(out + 1, mask + 1, NULL, b, a[0], 0, n) == 0); \
        for (size_t i = 0; i < n; ++i) assert(memcmp(out + i + 1, mask[i + 1] ? a : b + i, sizeof(type)) == 0); \
        size_t expected = 0, result; \
        for (size_t i = 0; i < n; ++i) expected += mask[i + 1] != 0; \
        assert(ts_extended_mask_reduce(0, mask + 1, n, &result) == 0 && result == expected); \
        uint8_t spill[] = {TS_OP_LT, 0, 8, 9, TS_OP_SPILL, 0, 0, 0, \
                           TS_OP_RELOAD, 1, 0, 0, TS_OP_COPY, TS_KERNEL_OUTPUT, 1, 0}; \
        size_t before = allocations; \
        assert(ts_kernel_mask_##suffix(spill, sizeof spill, NULL, inputs, 2, mask + 1, n, 1, 0, NULL) == 0); \
        assert(allocations == before + (n != 0) && allocations == releases); \
        for (size_t i = 0; i < n; ++i) assert(mask[i + 1] == (a[i] < b[i])); \
    } \
    uint8_t code[] = {TS_OP_LT, TS_KERNEL_OUTPUT, 8, 9}; \
    memset(mask, 7, sizeof mask); fail_allocation = 1; size_t before = allocations; \
    assert(ts_kernel_mask_##suffix(code, sizeof code, NULL, inputs, 2, mask, 601, 1, 0, NULL) == -1); \
    assert(allocations == before + 1 && releases == before && mask[0] == 7); \
    --allocations; fail_allocation = 0; \
    assert(ts_kernel_mask_##suffix(code, sizeof code, NULL, inputs, 2, mask, 601, SIZE_MAX, 0, NULL) == -1); \
    assert(allocations == before); \
}
CHECK_MASK(f32, float, float)
CHECK_MASK(f64, double, double)
CHECK_MASK(s8, int8_t, uint8_t)
CHECK_MASK(u8, uint8_t, uint8_t)
CHECK_MASK(s16, int16_t, uint16_t)
CHECK_MASK(u16, uint16_t, uint16_t)
CHECK_MASK(s32, int32_t, uint32_t)
CHECK_MASK(u32, uint32_t, uint32_t)
CHECK_MASK(s64, int64_t, uint64_t)
CHECK_MASK(u64, uint64_t, uint64_t)

#define CHECK_CONVERT(name, source_type, tag) \
static void check_convert_##name(void) { \
    source_type input[603]; float f32[604]; double f64[604]; \
    uint64_t state = 0x123456789abcdef0; \
    for (size_t i = 0; i < 603; ++i) { state ^= state << 13; state ^= state >> 7; state ^= state << 17; input[i] = (source_type)state; } \
    input[1] = (source_type)0; input[2] = (source_type)-1; \
    input[3] = (source_type)(UINT64_C(1) << 63); \
    input[4] = (source_type)((UINT64_C(1) << 62) + (UINT64_C(1) << 38) + 1); \
    for (size_t n = 0; n <= 601; n += n < 20 ? 1 : n < 255 ? 235 : n < 259 ? 1 : 342) { \
        f32[0] = f32[n + 1] = -9; f64[0] = f64[n + 1] = -9; \
        assert(ts_extended_convert_numeric(f32 + 1, input + 1, n, 0, tag) == 0); \
        assert(ts_extended_convert_numeric(f64 + 1, input + 1, n, 1, tag) == 0); \
        assert(f32[0] == -9 && f32[n + 1] == -9 && f64[0] == -9 && f64[n + 1] == -9); \
        for (size_t i = 0; i < n; ++i) { volatile source_type value = input[i + 1]; \
            assert(f32[i + 1] == (float)value); assert(f64[i + 1] == (double)value); \
        } \
    } \
}
CHECK_CONVERT(s8, int8_t, 2)
CHECK_CONVERT(u8, uint8_t, 3)
CHECK_CONVERT(s16, int16_t, 4)
CHECK_CONVERT(u16, uint16_t, 5)
CHECK_CONVERT(s32, int32_t, 6)
CHECK_CONVERT(u32, uint32_t, 7)
CHECK_CONVERT(s64, int64_t, 8)
CHECK_CONVERT(u64, uint64_t, 9)

#define CHECK_NARROW(name, source, target, ds, is, low, high) \
static void check_narrow_##name(void) { \
    source input[603]; target out[604]; \
    int64_t boundaries[] = {INT32_MIN, -65536, -32769, -32768, -129, -128, -1, 0, 1, 127, 128, 255, 256, 32767, 32768, 65535, 65536, UINT32_MAX}; \
    for (size_t i = 0; i < 603; ++i) input[i] = (source)boundaries[i % 18]; \
    for (size_t n = 0; n <= 601; n += n < 20 ? 1 : n < 255 ? 235 : n < 259 ? 1 : 342) { \
        out[0] = out[n + 1] = 7; \
        assert(ts_extended_convert_numeric(out + 1, input + 1, n, ds, is) == 0); \
        assert(out[0] == 7 && out[n + 1] == 7); \
        for (size_t i = 0; i < n; ++i) { int64_t value = input[i + 1]; \
            assert(out[i + 1] == (value < low ? low : value > high ? high : value)); \
        } \
    } \
}
CHECK_NARROW(s16_s8, int16_t, int8_t, 2, 4, INT8_MIN, INT8_MAX)
CHECK_NARROW(u16_s8, uint16_t, int8_t, 2, 5, INT8_MIN, INT8_MAX)
CHECK_NARROW(s16_u8, int16_t, uint8_t, 3, 4, 0, UINT8_MAX)
CHECK_NARROW(u16_u8, uint16_t, uint8_t, 3, 5, 0, UINT8_MAX)
CHECK_NARROW(s32_s16, int32_t, int16_t, 4, 6, INT16_MIN, INT16_MAX)
CHECK_NARROW(u32_s16, uint32_t, int16_t, 4, 7, INT16_MIN, INT16_MAX)
CHECK_NARROW(s32_u16, int32_t, uint16_t, 5, 6, 0, UINT16_MAX)
CHECK_NARROW(u32_u16, uint32_t, uint16_t, 5, 7, 0, UINT16_MAX)

static void check_float_selection_bits(void) {
    uint64_t values[] = {UINT64_C(0x8000000000000000), UINT64_C(0x7ff0000000000001),
                         UINT64_C(0xfff8123456789abc), UINT64_C(0x0000000000000001)};
    double input[19], out[19], other[19]; uint8_t mask[19];
    for (size_t i = 0; i < 19; ++i) { memcpy(input + i, values + i % 4, sizeof(double));
        other[i] = 0; mask[i] = (uint8_t)(i % 3 ? 255 : 0); }
    assert(ts_extended_select_f64(out, mask, input, other, 0, 0, 19) == 0);
    for (size_t i = 0; i < 19; ++i) assert(memcmp(out + i, mask[i] ? input + i : other + i, sizeof(double)) == 0);
    uint8_t invalid[] = {TS_OP_SQRT, TS_KERNEL_OUTPUT, 8, 0};
    float negative[257]; const float *inputs[] = {negative}; uint8_t bits[257];
    for (size_t i = 0; i < 257; ++i) negative[i] = -1;
    size_t before = allocations;
    assert(ts_kernel_mask_f32(invalid, sizeof invalid, NULL, inputs, 1, bits, 257, 1, 0, NULL) == -2);
    assert(allocations == before + 1 && allocations == releases);
}

int main(void) {
    check_float_selection_bits();
#define RUN_MASK(name) check_mask_##name();
    RUN_MASK(f32) RUN_MASK(f64) RUN_MASK(s8) RUN_MASK(u8) RUN_MASK(s16) RUN_MASK(u16)
    RUN_MASK(s32) RUN_MASK(u32) RUN_MASK(s64) RUN_MASK(u64)
#define RUN_CONVERT(name) check_convert_##name();
    RUN_CONVERT(s8) RUN_CONVERT(u8) RUN_CONVERT(s16) RUN_CONVERT(u16)
    RUN_CONVERT(s32) RUN_CONVERT(u32) RUN_CONVERT(s64) RUN_CONVERT(u64)
#define RUN_NARROW(name) check_narrow_##name();
    RUN_NARROW(s16_s8) RUN_NARROW(u16_s8) RUN_NARROW(s16_u8) RUN_NARROW(u16_u8)
    RUN_NARROW(s32_s16) RUN_NARROW(u32_s16) RUN_NARROW(s32_u16) RUN_NARROW(u32_u16)
    puts("mask and conversion checks passed");
}
