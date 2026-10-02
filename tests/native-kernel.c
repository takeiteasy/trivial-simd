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

#define CHECK_KERNEL_REDUCTION(suffix, type) \
static void check_kernel_reduction_##suffix(void) { \
    type a[600], value = 0; \
    const type *inputs[] = {a}; \
    double wide = 0; \
    size_t index = 99; \
    uint8_t copy[] = {TS_OP_COPY, TS_KERNEL_OUTPUT, 8, 0}; \
    uint8_t spill[] = {TS_OP_COPY, 0, 8, 0, TS_OP_SPILL, 0, 0, 0, \
                       TS_OP_RELOAD, 1, 0, 0, TS_OP_COPY, TS_KERNEL_OUTPUT, 1, 0}; \
    size_t lengths[] = {1, 2, 255, 256, 257, 600}; \
    for (size_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) { \
        size_t n = lengths[l], least = 0, greatest = 0; \
        double squares = 0, magnitude = 0, largest = 0; \
        for (size_t i = 0; i < 600; ++i) a[i] = (type)((int)((i * 7 + 3) % 9) - 4); \
        for (size_t i = 0; i < n; ++i) { \
            if (a[i] < a[least]) least = i; \
            if (a[i] > a[greatest]) greatest = i; \
            squares += (double)a[i] * a[i]; \
            magnitude += fabs((double)a[i]); \
            if (fabs((double)a[i]) > largest) largest = fabs((double)a[i]); \
        } \
        for (size_t program = 0; program < 2; ++program) { \
            const uint8_t *code = program ? spill : copy; \
            size_t size = program ? sizeof(spill) : sizeof(copy), before = allocations; \
            assert(ts_kernel_reduction_##suffix(code, size, NULL, inputs, 1, n, program, \
                                                TS_REDUCE_ARGMIN, 0, &value, &wide, &index) == 0); \
            assert(index == least && value == a[least]); \
            assert(ts_kernel_reduction_##suffix(code, size, NULL, inputs, 1, n, program, \
                                                TS_REDUCE_ARGMAX, 0, &value, &wide, &index) == 0); \
            assert(index == greatest && value == a[greatest]); \
            assert(ts_kernel_reduction_##suffix(code, size, NULL, inputs, 1, n, program, \
                                                TS_REDUCE_ASUM, 0, &value, &wide, &index) == 0); \
            assert(value == (type)magnitude); \
            assert(ts_kernel_reduction_##suffix(code, size, NULL, inputs, 1, n, program, \
                                                TS_REDUCE_SUMSQ, 0, &value, &wide, &index) == 0); \
            assert(wide == squares); \
            assert(ts_kernel_reduction_##suffix(code, size, NULL, inputs, 1, n, program, \
                                                TS_REDUCE_MAXABS, 0, &value, &wide, &index) == 0); \
            assert(wide == largest); \
            assert(ts_kernel_reduction_##suffix(code, size, NULL, inputs, 1, n, program, \
                                                TS_REDUCE_SCALED_SUMSQ, 2, &value, &wide, &index) == 0); \
            assert(wide == squares / 4); \
            assert(allocations == before + 6 * program && allocations == releases); \
        } \
    } \
    assert(ts_kernel_reduction_##suffix(copy, sizeof(copy), NULL, inputs, 1, 4, 0, 0, 0, \
                                        &value, &wide, &index) == -1); \
    assert(ts_kernel_reduction_##suffix(copy, sizeof(copy), NULL, inputs, 1, 4, 0, \
                                        TS_REDUCE_SCALED_SUMSQ + 1, 0, &value, &wide, &index) == -1); \
    assert(ts_kernel_reduction_##suffix(copy, sizeof(copy), NULL, inputs, 257, 4, 0, \
                                        TS_REDUCE_ARGMIN, 0, &value, &wide, &index) == -1); \
    size_t before = allocations; \
    fail_allocation = 1; \
    assert(ts_kernel_reduction_##suffix(spill, sizeof(spill), NULL, inputs, 1, 4, 1, \
                                        TS_REDUCE_ARGMIN, 0, &value, &wide, &index) == -1); \
    assert(allocations == before + 1 && releases == before); \
    --allocations; \
    fail_allocation = 0; \
    uint8_t root[] = {TS_OP_SQRT, TS_KERNEL_OUTPUT, 8, 0}; \
    for (size_t i = 0; i < 600; ++i) a[i] = 1; \
    a[599] = -1; \
    assert(ts_kernel_reduction_##suffix(root, sizeof(root), NULL, inputs, 1, 600, 1, \
                                        TS_REDUCE_ASUM, 0, &value, &wide, &index) == -2); \
    assert(allocations == releases); \
}

CHECK_KERNEL_REDUCTION(f32, float)
CHECK_KERNEL_REDUCTION(f64, double)

static void check_kernel_reduction_range(void) {
    double a[300], value = 0, wide = 0;
    const double *inputs[] = {a};
    size_t index = 0;
    uint8_t copy[] = {TS_OP_COPY, TS_KERNEL_OUTPUT, 8, 0};
    for (size_t i = 0; i < 300; ++i) a[i] = 1e200;
    assert(ts_kernel_reduction_f64(copy, sizeof(copy), NULL, inputs, 1, 300, 0,
                                   TS_REDUCE_SUMSQ, 0, &value, &wide, &index) == 0);
    assert(isinf(wide));
    assert(ts_kernel_reduction_f64(copy, sizeof(copy), NULL, inputs, 1, 300, 0,
                                   TS_REDUCE_SCALED_SUMSQ, 1e200, &value, &wide, &index) == 0);
    assert(wide == 300);
}

#define CHECK_INTEGER_REDUCTION(suffix, type, bits, signedp) \
static void check_integer_reduction_##suffix(void) { \
    type a[300], value = 0; \
    const type *inputs[] = {a}; \
    double wide = 0; \
    size_t index = 99; \
    type low = (type)(UINT64_C(1) << (bits - 1)), high = (type)(low - 1); \
    type pattern[] = {high, low, 1, low}; \
    uint8_t copy[] = {TS_OP_COPY, TS_KERNEL_OUTPUT, 8, 0}; \
    for (size_t i = 0; i < 300; ++i) a[i] = pattern[i % 4]; \
    assert(ts_kernel_reduction_##suffix(copy, sizeof(copy), NULL, inputs, 1, 300, 0, \
                                        TS_REDUCE_ARGMIN, 0, &value, &wide, &index) == 0); \
    assert(index == (signedp ? 1 : 2) && value == pattern[index]); \
    assert(ts_kernel_reduction_##suffix(copy, sizeof(copy), NULL, inputs, 1, 300, 0, \
                                        TS_REDUCE_ARGMAX, 0, &value, &wide, &index) == 0); \
    assert(index == (signedp ? 0 : 1) && value == pattern[index]); \
    assert(ts_kernel_reduction_##suffix(copy, sizeof(copy), NULL, inputs, 1, 300, 0, \
                                        TS_REDUCE_ASUM, 0, &value, &wide, &index) == 0); \
    assert(value == (type)(UINT64_C(75) * ((uint64_t)high + low + 1 + low))); \
    assert(ts_kernel_reduction_##suffix(copy, sizeof(copy), NULL, inputs, 1, 300, 0, \
                                        TS_REDUCE_SUMSQ, 0, &value, &wide, &index) == -1); \
}

CHECK_INTEGER_REDUCTION(s8, uint8_t, 8, 1)
CHECK_INTEGER_REDUCTION(u8, uint8_t, 8, 0)
CHECK_INTEGER_REDUCTION(s16, uint16_t, 16, 1)
CHECK_INTEGER_REDUCTION(u16, uint16_t, 16, 0)
CHECK_INTEGER_REDUCTION(s32, uint32_t, 32, 1)
CHECK_INTEGER_REDUCTION(u32, uint32_t, 32, 0)
CHECK_INTEGER_REDUCTION(s64, uint64_t, 64, 1)
CHECK_INTEGER_REDUCTION(u64, uint64_t, 64, 0)

static void check_scalar_fma(void) {
    assert(ts_fma_scalar_f32(0x1.000002p0f, 0x1.fffffcp-1f, -1.0f) == -0x1p-46f);
    assert(ts_fma_scalar_f64(0x1.0000000000001p0, 0x1.ffffffffffffep-1, -1.0) == -0x1p-104);
    assert(signbit(ts_fma_scalar_f32(-0.0f, 1.0f, -0.0f)));
    assert(signbit(ts_fma_scalar_f64(-0.0, 1.0, -0.0)));
    assert(ts_fma_supported() == 0 || ts_fma_supported() == 1);
#ifdef TS_FORCE_SOFTWARE_FMA
    assert(ts_fma_supported() == 0);
#endif
}

#define CHECK_INTEGER(suffix, type, high, low) \
static void check_integer_##suffix(void) { \
    type a[257], b[257], out[257], result = 0; \
    const type *inputs[] = {a, b}; \
    for (size_t i = 0; i < 257; ++i) { a[i] = high; b[i] = 1; out[i] = 0; } \
    a[256] = low; \
    assert(ts_add_##suffix(out, a, b, 257) == 0); \
    assert(out[0] == (type)(high + 1) && out[256] == (type)(low + 1)); \
    assert(ts_subtract_##suffix(out, a, b, 257) == 0); \
    assert(out[0] == (type)(high - 1)); \
    assert(ts_multiply_##suffix(out, a, b, 257) == 0); \
    assert(out[256] == low); \
    assert(ts_sum_##suffix(a, &result, 257) == 0); \
    assert(result == (type)((uint64_t)high * 256 + low)); \
    assert(ts_dot_##suffix(a, b, &result, 257) == 0); \
    assert(result == (type)((uint64_t)high * 256 + low)); \
    b[256] = 0; \
    assert(ts_divide_##suffix(out, a, b, 257) == -3); \
    uint8_t code[] = {TS_OP_DIVIDE, TS_KERNEL_OUTPUT, 8, 9}; \
    assert(ts_kernel_##suffix(code, sizeof(code), NULL, inputs, out, 257, 0) == -3); \
    b[256] = 1; \
    assert(ts_kernel_##suffix(code, sizeof(code), NULL, inputs, out, 257, 0) == 0); \
    assert(out[256] == low); \
    assert(ts_kernel_sum_##suffix(code, sizeof(code), NULL, inputs, &result, 257, 0) == 0); \
    assert(result == (type)((uint64_t)high * 256 + low)); \
    uint8_t spill[] = {TS_OP_CONSTANT, 0, 0, 0, TS_OP_SPILL, 0, 0, 0, \
                       TS_OP_RELOAD, 0, 0, 0, TS_OP_COPY, TS_KERNEL_OUTPUT, 0, 0}; \
    type constant[] = {high}; \
    size_t before = allocations; \
    fail_allocation = 1; \
    assert(ts_kernel_##suffix(spill, sizeof(spill), constant, inputs, out, 257, 1) == -1); \
    assert(allocations == before + 1 && releases == before); \
    --allocations; \
    fail_allocation = 0; \
    assert(ts_kernel_##suffix(spill, sizeof(spill), constant, inputs, out, 257, 1) == 0); \
    assert(allocations == releases && out[256] == high); \
}

CHECK_INTEGER(s8, uint8_t, UINT8_MAX, 0x80)
CHECK_INTEGER(u8, uint8_t, UINT8_MAX, 0)
CHECK_INTEGER(s16, uint16_t, UINT16_MAX, 0x8000)
CHECK_INTEGER(u16, uint16_t, UINT16_MAX, 0)
CHECK_INTEGER(s32, uint32_t, UINT32_MAX, UINT32_C(0x80000000))
CHECK_INTEGER(u32, uint32_t, UINT32_MAX, 0)
CHECK_INTEGER(s64, uint64_t, UINT64_MAX, UINT64_C(0x8000000000000000))
CHECK_INTEGER(u64, uint64_t, UINT64_MAX, 0)

static void check_integer_mul_add_64(void) {
    uint8_t code[] = {TS_OP_MULTIPLY, 0, 8, 9,
                      TS_OP_CONSTANT, 1, 0, 0,
                      TS_OP_ADD, TS_KERNEL_OUTPUT, 0, 1};
    uint64_t constants[] = {UINT64_MAX};
    uint64_t a[258], b[257], out[257];
    const uint64_t *inputs[] = {a, b};
    for (size_t i = 0; i < 257; ++i) {
        a[i] = UINT64_MAX - i;
        b[i] = i + 2;
    }
    a[257] = 17;
    assert(ts_mul_add_constant_pattern(code, sizeof(code)));
    assert(ts_kernel_u64(code, sizeof(code), constants, inputs, out, 257, 0) == 0);
    for (size_t i = 0; i < 257; ++i)
        assert(out[i] == a[i] * b[i] + constants[0]);
    assert(ts_kernel_s64(code, sizeof(code), constants, inputs, out, 257, 0) == 0);
    for (size_t i = 0; i < 257; ++i)
        assert(out[i] == a[i] * b[i] + constants[0]);
    assert(ts_kernel_u64(code, sizeof(code), constants, inputs, a, 257, 0) == 0);
    for (size_t i = 0; i < 257; ++i)
        assert(a[i] == (UINT64_MAX - i) * b[i] + constants[0]);
    for (size_t i = 0; i < 257; ++i) a[i] = UINT64_MAX - i;
    assert(ts_kernel_u64(code, sizeof(code), constants, inputs, a + 1, 257, 0) == 0);
    assert(a[1] == UINT64_MAX * b[0] + constants[0]);
    assert(a[257] == a[256] * b[256] + constants[0]);
    code[4] = TS_OP_COPY;
    assert(!ts_mul_add_constant_pattern(code, sizeof(code)));
}

static void check_mask_kernels(void) {
    uint8_t compare[] = {TS_OP_GT, TS_KERNEL_OUTPUT, 8, 9};
    uint8_t choose[] = {TS_OP_COPY, 0, 10, 0,
                        TS_OP_GT, 1, 8, 9,
                        TS_OP_SELECT, 0, 1, 8,
                        TS_OP_COPY, TS_KERNEL_OUTPUT, 0, 0};
    float a[259], b[259], fallback[259], output[259];
    uint8_t mask[259];
    const float *inputs[] = {a, b, fallback};
    size_t result;
    for (size_t i = 0; i < 259; ++i) {
        a[i] = (float)i;
        b[i] = 129.0f;
        fallback[i] = -1.0f;
    }
    assert(ts_kernel_mask_f32(compare, sizeof(compare), NULL, inputs, 2,
                              mask, 259, 0, 0, NULL) == 0);
    for (size_t i = 0; i < 259; ++i) assert(mask[i] == (i > 129));
    assert(ts_kernel_mask_f32(compare, sizeof(compare), NULL, inputs, 2,
                              NULL, 259, 0, 1, &result) == 0);
    assert(result == 129);
    assert(ts_kernel_mask_f32(compare, sizeof(compare), NULL, inputs, 2,
                              NULL, 259, 0, 2, &result) == 0);
    assert(result == 1);
    assert(ts_kernel_mask_f32(compare, sizeof(compare), NULL, inputs, 2,
                              NULL, 259, 0, 3, &result) == 0);
    assert(result == 0);
    assert(ts_kernel_f32(choose, sizeof(choose), NULL, inputs, output, 259, 0) == 0);
    for (size_t i = 0; i < 259; ++i) assert(output[i] == (i > 129 ? a[i] : -1.0f));

    uint8_t signed_compare[] = {TS_OP_LT, TS_KERNEL_OUTPUT, 8, 9};
    uint8_t x[] = {0x80, 0x7f}, y[] = {0x7f, 0x80};
    const uint8_t *signed_inputs[] = {x, y};
    assert(ts_kernel_mask_s8(signed_compare, sizeof(signed_compare), NULL,
                             signed_inputs, 2, mask, 2, 0, 0, NULL) == 0);
    assert(mask[0] == 1 && mask[1] == 0);
}

#define CHECK_REDUCTIONS(suffix, type, wide_check) \
static void check_reductions_##suffix(void) { \
    type input[300]; \
    for (size_t n = 0; n <= 300; n += n < 70 ? 1 : 37) { \
        for (size_t i = 0; i < n; ++i) input[i] = (type)((int)((i * 7 + 3) % 9) - 4); \
        size_t least = 0, greatest = 0, largest = 0; \
        double sum = 0, magnitude = 0; \
        for (size_t i = 1; i < n; ++i) { \
            if (input[i] < input[least]) least = i; \
            if (input[i] > input[greatest]) greatest = i; \
            if (fabs((double)input[i]) > fabs((double)input[largest])) largest = i; \
        } \
        for (size_t i = 0; i < n; ++i) { sum += input[i]; magnitude += fabs((double)input[i]); } \
        assert(ts_argmin_##suffix(input, n) == least); \
        assert(ts_argmax_##suffix(input, n) == greatest); \
        assert(ts_iamax_##suffix(input, n) == largest); \
        assert(ts_asum_##suffix(input, n) == (type)magnitude); \
        wide_check \
    } \
}

CHECK_REDUCTIONS(f32, float, \
    assert(ts_sum_acc_f32(input, n) == sum); \
    assert(ts_asum_acc_f32(input, n) == magnitude); \
    assert(ts_dot_acc_f32(input, input, n) == (double)ts_dot_f32(input, input, n));)
CHECK_REDUCTIONS(f64, double, )

static void check_first_extreme_wins(void) {
    float input[100];
    for (size_t i = 0; i < 100; ++i) input[i] = 1.0f;
    input[40] = -0.0f; input[41] = 0.0f;
    assert(ts_argmin_f32(input, 100) == 40 && ts_argmax_f32(input, 100) == 0);
    input[40] = 0.0f; input[41] = -0.0f;
    for (size_t i = 0; i < 100; ++i) input[i] = -input[i];
    assert(ts_argmax_f32(input, 100) == 40 && ts_argmin_f32(input, 100) == 0);
    {
        float widened[3] = {16777216.0f, 1.0f, 1.0f};
        assert(ts_sum_acc_f32(widened, 3) == 16777218.0);
    }
}

static void check_swap_and_fill(void) {
    size_t sizes[] = {0, 1, 15, 16, 17, 31, 32, 33, 90};
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
        size_t n = sizes[s];
        uint8_t a[100], b[100];
        for (size_t i = 0; i < 100; ++i) { a[i] = (uint8_t)i; b[i] = (uint8_t)(200 + i); }
        ts_swap_bytes(a + 3, b + 5, n);
        for (size_t i = 0; i < 100; ++i) {
            int moved_a = i >= 3 && i < 3 + n, moved_b = i >= 5 && i < 5 + n;
            assert(a[i] == (moved_a ? (uint8_t)(200 + i - 3 + 5) : (uint8_t)i));
            assert(b[i] == (moved_b ? (uint8_t)(i - 5 + 3) : (uint8_t)(200 + i)));
        }
    }
    float f[9] = {0};
    ts_fill_f32(f + 1, -0.0f, 7);
    assert(f[0] == 0 && !signbit(f[0]) && f[8] == 0 && !signbit(f[8]));
    for (size_t i = 1; i < 8; ++i) assert(f[i] == 0 && signbit(f[i]));
    uint64_t wide[5] = {1, 1, 1, 1, 1};
    ts_fill_u64(wide + 1, UINT64_MAX, 3);
    assert(wide[0] == 1 && wide[1] == UINT64_MAX && wide[3] == UINT64_MAX && wide[4] == 1);
    uint8_t bytes[4] = {1, 1, 1, 1};
    ts_fill_s8(bytes, 0xFE, 0);
    assert(bytes[0] == 1);
    ts_fill_s8(bytes, 0xFE, 4);
    assert(bytes[3] == 0xFE);
}

int main(void) {
    check_swap_and_fill();
    check_reductions_f32();
    check_reductions_f64();
    check_first_extreme_wins();
    check_integer_s8();
    check_integer_u8();
    check_integer_s16();
    check_integer_u16();
    check_integer_s32();
    check_integer_u32();
    check_integer_s64();
    check_integer_u64();
    check_integer_mul_add_64();
    check_mask_kernels();
    check_kernel_reduction_f32();
    check_kernel_reduction_f64();
    check_kernel_reduction_range();
    check_integer_reduction_s8();
    check_integer_reduction_u8();
    check_integer_reduction_s16();
    check_integer_reduction_u16();
    check_integer_reduction_s32();
    check_integer_reduction_u32();
    check_integer_reduction_s64();
    check_integer_reduction_u64();
    check_scalar_fma();
    check_f32();
    check_f64();
    check_math_f32();
    check_math_f64();
    check_final_output_f32();
    check_final_output_f64();
    return 0;
}
