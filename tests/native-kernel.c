#include <stdint.h>
#include <stdio.h>
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

static void check_zero_sign(double value, int negative, const char *operation) {
    if (value != 0 || !!signbit(value) != negative) {
        fprintf(stderr, "%s: expected %s zero, got %.17g (sign %d)\n",
                operation, negative ? "negative" : "positive", value, !!signbit(value));
        abort();
    }
}

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
    size_t zero_lengths[] = {1, 5, 600}; \
    for (size_t z = 0; z < sizeof(zero_lengths) / sizeof(zero_lengths[0]); ++z) { \
        size_t n = zero_lengths[z]; \
        assert(ts_kernel_##suffix(sqrt_code, sizeof(sqrt_code), NULL, inputs, out, n, 0) == 0); \
        for (size_t i = 0; i < n; ++i) check_zero_sign(out[i], 1, "sqrt"); \
        assert(ts_kernel_##suffix(min_code, sizeof(min_code), NULL, inputs, out, n, 0) == 0); \
        for (size_t i = 0; i < n; ++i) check_zero_sign(out[i], 1, "min"); \
        assert(ts_kernel_##suffix(max_code, sizeof(max_code), NULL, inputs, out, n, 0) == 0); \
        for (size_t i = 0; i < n; ++i) check_zero_sign(out[i], 1, "max"); \
    } \
    for (size_t i = 0; i < 600; ++i) b[i] = 1; \
    assert(ts_kernel_##suffix(fma_code, sizeof(fma_code), NULL, inputs, out, 600, 0) == 0); \
    for (size_t i = 0; i < 600; ++i) check_zero_sign(out[i], 1, "fma"); \
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

#define CHECK_FINAL_OUTPUT(suffix, type, width, epsilon) \
static void check_final_output_##suffix(void) { \
    type a[600], b[600], out[600], scratch[TS_KERNEL_BLOCK], constants[] = {(type)1.25}; \
    const type *inputs[] = {a, b}; \
    size_t lengths[] = {0, 1, 2, 3, 4, 5, 255, 256, 257, 600}; \
    for (uint8_t op = TS_OP_COPY; op <= TS_OP_FMA; ++op) { \
        if (op == TS_OP_SPILL || op == TS_OP_RELOAD) continue; \
        uint8_t code[] = {TS_OP_COPY, 0, 8, 0, TS_OP_SPILL, 0, 0, 0, \
                          TS_OP_RELOAD, 1, 0, 0, op, TS_KERNEL_OUTPUT, 8, 9}; \
        if (op == TS_OP_CONSTANT) code[14] = 0; \
        if (op == TS_OP_FMA) { \
            code[12] = TS_OP_FMA; code[13] = 1; \
        } \
        uint8_t fma_code[20]; \
        memcpy(fma_code, code, sizeof(code)); \
        fma_code[16] = TS_OP_COPY; fma_code[17] = TS_KERNEL_OUTPUT; \
        fma_code[18] = 1; fma_code[19] = 0; \
        const uint8_t *program = op == TS_OP_FMA ? fma_code : code; \
        size_t size = op == TS_OP_FMA ? sizeof(fma_code) : sizeof(code); \
        for (size_t i = 0; i < 600; ++i) { \
            a[i] = i % 3 == 0 ? (type)0x1p30 : (i % 3 == 1 ? (type)-0x1p30 : (type)0.125); \
            if (op == TS_OP_SQRT) a[i] = (type)(i % 17); \
            b[i] = (type)(1 + i % 7); \
        } \
        for (size_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) { \
            size_t n = lengths[l], before = allocations; \
            assert(ts_kernel_with_scratch_##suffix(program, size, constants, inputs, out, n, 1, scratch, 1) == 0); \
            type expected = 0, actual[2] = {-1, -9}; \
            for (size_t base = 0; base < n; base += TS_KERNEL_BLOCK) { \
                size_t count = n - base < TS_KERNEL_BLOCK ? n - base : TS_KERNEL_BLOCK; \
                expected += ts_sum_##suffix(out + base, count); \
            } \
            assert(ts_kernel_sum_with_scratch_##suffix(program, size, constants, inputs, actual, n, 1, scratch, 1) == 0); \
            assert(memcmp(actual, &expected, sizeof(type)) == 0 && actual[1] == -9); \
            actual[0] = -1; \
            assert(ts_kernel_sum_##suffix(program, size, constants, inputs, actual, n, 1) == 0); \
            assert(memcmp(actual, &expected, sizeof(type)) == 0); \
            assert(allocations == before + (n != 0) && allocations == releases); \
        } \
    } \
    uint8_t copy[] = {TS_OP_COPY, TS_KERNEL_OUTPUT, 8, 0}; \
    type scalar = -1; \
    size_t before = allocations; \
    assert(ts_kernel_sum_with_scratch_##suffix(copy, sizeof(copy), NULL, inputs, &scalar, 1, 1, scratch, 0) == -1); \
    assert(scalar == -1); \
    assert(ts_kernel_with_scratch_##suffix(copy, sizeof(copy), NULL, inputs, out, 1, 1, NULL, 1) == -1); \
    assert(ts_kernel_sum_with_scratch_##suffix(NULL, 0, NULL, NULL, &scalar, 0, SIZE_MAX, NULL, 0) == 0); \
    assert(scalar == 0); \
    scalar = -1; \
    assert(ts_kernel_sum_with_scratch_##suffix(copy, sizeof(copy), NULL, inputs, &scalar, 1, SIZE_MAX, scratch, SIZE_MAX) == -1); \
    assert(scalar == -1); \
    uint8_t root[] = {TS_OP_SQRT, TS_KERNEL_OUTPUT, 8, 0}; \
    for (size_t i = 0; i < 600; ++i) a[i] = 1; \
    a[599] = -1; \
    assert(ts_kernel_sum_with_scratch_##suffix(root, sizeof(root), NULL, inputs, &scalar, 600, 1, scratch, 1) == -2); \
    assert(scalar == -1 && allocations == before); \
    a[599] = 1; \
    assert(ts_kernel_sum_with_scratch_##suffix(root, sizeof(root), NULL, inputs, &scalar, 600, 1, scratch, 1) == 0); \
    assert(scalar == 600 && allocations == before); \
    uint8_t product[] = {TS_OP_MULTIPLY, TS_KERNEL_OUTPUT, 8, 9}; \
    size_t rounding_lengths[] = {3, 2 * width}; \
    for (size_t l = 0; l < 2; ++l) { \
        size_t n = rounding_lengths[l]; \
        for (size_t i = 0; i < n; ++i) { a[i] = 0; b[i] = 1; } \
        a[0] = -1; \
        size_t last = l == 0 ? n - 1 : width; \
        a[last] = (type)(1 + epsilon); b[last] = (type)(1 - epsilon); \
        assert(ts_kernel_sum_##suffix(product, sizeof(product), NULL, inputs, &scalar, n, 0) == 0); \
        assert(scalar == 0); \
    } \
}

CHECK_FINAL_OUTPUT(f32, float, TS_F32_WIDTH, 0x1p-23f)
CHECK_FINAL_OUTPUT(f64, double, TS_F64_WIDTH, 0x1p-52)

int main(void) {
    check_f32();
    check_f64();
    check_math_f32();
    check_math_f64();
    check_final_output_f32();
    check_final_output_f64();
    return 0;
}
