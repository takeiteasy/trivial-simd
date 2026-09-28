#include <stdint.h>
#include <stdlib.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

static int fail_allocation;
static size_t allocations, releases;

static void *test_malloc(size_t size) {
    ++allocations;
    return fail_allocation ? NULL : malloc(size);
}

static void test_free(void *pointer) {
    if (pointer) ++releases;
    free(pointer);
}

#define malloc test_malloc
#define free test_free
#include "../native/trivial_simd.c"
#undef malloc
#undef free

#define CHECK_KERNEL(suffix, type) \
static void check_##suffix(void) { \
    type out[600], constants[] = {42, 9}; \
    size_t lengths[] = {0, 1, 3, 4, 5, 255, 256, 257, 600}; \
    size_t slots[] = {0, 256, 65535}; \
    for (size_t s = 0; s < sizeof(slots) / sizeof(slots[0]); ++s) { \
        size_t slot = slots[s]; \
        uint8_t code[] = {TS_OP_CONSTANT, 0, 0, 0, \
                          TS_OP_SPILL, 0, slot & 255, slot >> 8, \
                          TS_OP_CONSTANT, 0, 1, 0, \
                          TS_OP_RELOAD, 1, slot & 255, slot >> 8, \
                          TS_OP_COPY, TS_KERNEL_OUTPUT, 1, 0}; \
        for (size_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) { \
            size_t n = lengths[l], before = allocations; \
            for (size_t i = 0; i < 600; ++i) out[i] = -1; \
            assert(ts_kernel_##suffix(code, sizeof(code), constants, NULL, out, n, slot + 1) == 0); \
            assert(allocations == before + (n != 0)); \
            assert(allocations == releases); \
            for (size_t i = 0; i < 600; ++i) assert(out[i] == (i < n ? 42 : -1)); \
            type sum_out[2] = {-1, -9}; \
            before = allocations; \
            assert(ts_kernel_sum_##suffix(code, sizeof(code), constants, NULL, sum_out, n, slot + 1) == 0); \
            assert(sum_out[0] == (type)(42 * n) && sum_out[1] == -9); \
            assert(allocations == before + (n != 0) && allocations == releases); \
        } \
    } \
    uint8_t constant[] = {TS_OP_CONSTANT, TS_KERNEL_OUTPUT, 0, 0}; \
    type scalar = -1; \
    size_t sum_before = allocations; \
    assert(ts_kernel_sum_##suffix(NULL, 0, NULL, NULL, &scalar, 0, SIZE_MAX) == 0); \
    assert(scalar == 0 && allocations == sum_before); \
    scalar = -1; \
    assert(ts_kernel_sum_##suffix(constant, sizeof(constant), constants, NULL, &scalar, 600, 0) == 0); \
    assert(scalar == 25200 && allocations == sum_before); \
    scalar = -1; \
    assert(ts_kernel_sum_##suffix(constant, sizeof(constant), constants, NULL, &scalar, 1, SIZE_MAX) != 0); \
    assert(scalar == -1 && allocations == sum_before); \
    size_t before = allocations; \
    assert(ts_kernel_##suffix(constant, sizeof(constant), constants, NULL, out, 1, 0) == 0); \
    assert(allocations == before); \
    out[0] = -1; \
    assert(ts_kernel_##suffix(constant, sizeof(constant), constants, NULL, out, 1, SIZE_MAX) != 0); \
    assert(allocations == before && out[0] == -1); \
    fail_allocation = 1; \
    assert(ts_kernel_##suffix(constant, sizeof(constant), constants, NULL, out, 1, 1) != 0); \
    assert(allocations == before + 1 && releases == before && out[0] == -1); \
    --allocations; \
    scalar = -1; \
    assert(ts_kernel_sum_##suffix(constant, sizeof(constant), constants, NULL, &scalar, 1, 1) != 0); \
    assert(scalar == -1 && allocations == before + 1 && releases == before); \
    --allocations; \
    fail_allocation = 0; \
}

CHECK_KERNEL(f32, float)
CHECK_KERNEL(f64, double)

#define CHECK_MATH(suffix, type, epsilon, root, absolute, fused) \
static void check_math_##suffix(void) { \
    type a[600], b[600], c[600], out[600]; \
    const type *inputs[] = {a, b, c}; \
    uint8_t sqrt_code[] = {TS_OP_SQRT, TS_KERNEL_OUTPUT, 8, 0}; \
    uint8_t abs_code[] = {TS_OP_ABS, TS_KERNEL_OUTPUT, 8, 0}; \
    uint8_t min_code[] = {TS_OP_MIN, TS_KERNEL_OUTPUT, 8, 9}; \
    uint8_t max_code[] = {TS_OP_MAX, TS_KERNEL_OUTPUT, 8, 9}; \
    uint8_t fma_code[] = {TS_OP_COPY, 0, 10, 0, TS_OP_FMA, 0, 8, 9, \
                          TS_OP_COPY, TS_KERNEL_OUTPUT, 0, 0}; \
    size_t lengths[] = {0, 1, 3, 4, 5, 255, 256, 257, 600}; \
    for (size_t i = 0; i < 600; ++i) { a[i] = (type)(i + 1); b[i] = 42; c[i] = -3; } \
    for (size_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) { \
        size_t n = lengths[l]; \
        assert(ts_kernel_##suffix(sqrt_code, sizeof(sqrt_code), NULL, inputs, out, n, 0) == 0); \
        for (size_t i = 0; i < n; ++i) assert(out[i] == root(a[i])); \
        assert(ts_kernel_##suffix(min_code, sizeof(min_code), NULL, inputs, out, n, 0) == 0); \
        for (size_t i = 0; i < n; ++i) assert(out[i] == (a[i] <= b[i] ? a[i] : b[i])); \
        assert(ts_kernel_##suffix(max_code, sizeof(max_code), NULL, inputs, out, n, 0) == 0); \
        for (size_t i = 0; i < n; ++i) assert(out[i] == (a[i] >= b[i] ? a[i] : b[i])); \
        assert(ts_kernel_##suffix(fma_code, sizeof(fma_code), NULL, inputs, out, n, 0) == 0); \
        for (size_t i = 0; i < n; ++i) assert(out[i] == fused(a[i], b[i], c[i])); \
        type expected = 0, scalar[2] = {-1, -9}; \
        for (size_t i = 0; i < n; ++i) expected += fused(a[i], b[i], c[i]); \
        assert(ts_kernel_sum_##suffix(fma_code, sizeof(fma_code), NULL, inputs, scalar, n, 0) == 0); \
        assert(scalar[0] == expected && scalar[1] == -9); \
    } \
    for (size_t i = 0; i < 600; ++i) a[i] = -(type)i; \
    assert(ts_kernel_##suffix(abs_code, sizeof(abs_code), NULL, inputs, out, 600, 0) == 0); \
    for (size_t i = 0; i < 600; ++i) assert(out[i] == absolute(a[i]) && !signbit(out[i])); \
    for (size_t i = 0; i < 600; ++i) { a[i] = (type)(1 + epsilon); b[i] = (type)(1 - epsilon); c[i] = -1; } \
    assert(ts_kernel_##suffix(fma_code, sizeof(fma_code), NULL, inputs, out, 600, 0) == 0); \
    for (size_t i = 0; i < 600; ++i) assert(out[i] == -(type)(epsilon * epsilon)); \
    for (size_t i = 0; i < 600; ++i) { a[i] = -(type)0.0; b[i] = 0; c[i] = -(type)0.0; } \
    assert(ts_kernel_##suffix(sqrt_code, sizeof(sqrt_code), NULL, inputs, out, 600, 0) == 0); \
    for (size_t i = 0; i < 600; ++i) assert(signbit(out[i])); \
    assert(ts_kernel_##suffix(min_code, sizeof(min_code), NULL, inputs, out, 600, 0) == 0); \
    for (size_t i = 0; i < 600; ++i) assert(signbit(out[i])); \
    assert(ts_kernel_##suffix(max_code, sizeof(max_code), NULL, inputs, out, 600, 0) == 0); \
    for (size_t i = 0; i < 600; ++i) assert(signbit(out[i])); \
    size_t errors[] = {0, 3, 256, 599}; \
    for (size_t e = 0; e < sizeof(errors) / sizeof(errors[0]); ++e) { \
        a[errors[e]] = -1; \
        assert(ts_kernel_##suffix(sqrt_code, sizeof(sqrt_code), NULL, inputs, out, 600, 1) == -2); \
        assert(allocations == releases); \
        type scalar = -1; \
        assert(ts_kernel_sum_##suffix(sqrt_code, sizeof(sqrt_code), NULL, inputs, &scalar, 600, 1) == -2); \
        assert(scalar == -1 && allocations == releases); \
        a[errors[e]] = 0; \
    } \
}

CHECK_MATH(f32, float, 0x1p-23f, sqrtf, fabsf, fmaf)
CHECK_MATH(f64, double, 0x1p-52, sqrt, fabs, fma)

int main(void) {
    check_f32();
    check_f64();
    check_math_f32();
    check_math_f64();
    return 0;
}
