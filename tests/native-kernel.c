#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

static int fail_allocation;
static size_t fail_allocation_at;
static size_t allocations, releases;

static void *test_malloc(size_t size) {
    ++allocations;
    return fail_allocation || allocations == fail_allocation_at ? NULL : malloc(size);
}

static void test_free(void *pointer) {
    if (pointer) ++releases;
    free(pointer);
}

#define malloc test_malloc
#define free test_free
#include "../native/trivial_simd.c"
#include "../native/blas.c"
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

#define CHECK_BLAS(suffix, type) \
static type blas_value_##suffix(int seed) { return (type)((seed * 7 + 3) % 17 - 8) / 8; } \
static void check_blas_##suffix(void) { \
    enum { M = 137, N = 29, K = 270 }; \
    static type a[M * K + 5], b[K * N + 5], c[M * N + 5], expected[M * N + 5]; \
    for (int i = 0; i < M * K + 5; ++i) a[i] = blas_value_##suffix(i); \
    for (int i = 0; i < K * N + 5; ++i) b[i] = blas_value_##suffix(i + 11); \
    type scalars[][2] = {{1, 0}, {(type)0.75, (type)0.5}, {0, 1}, {0, (type)0.5}, {(type)-1, 1}}; \
    for (int s = 0; s < 5; ++s) { \
        type alpha = scalars[s][0], beta = scalars[s][1]; \
        for (int i = 0; i < M * N + 5; ++i) c[i] = expected[i] = blas_value_##suffix(i + 5); \
        for (int i = 0; i < M; ++i) \
            for (int j = 0; j < N; ++j) { \
                type sum = 0; \
                for (int p = 0; p < K; ++p) sum += a[i * K + p] * b[p + j * K]; \
                expected[i * N + j] = (beta == 0 ? 0 : beta * expected[i * N + j]) + alpha * sum; \
            } \
        assert(ts_blas_gemm_##suffix(M, N, K, alpha, a, K, 1, b, 1, K, beta, c, N, 1) == 0); \
        for (int i = 0; i < M * N; ++i) assert(fabs((double)(c[i] - expected[i])) < 1e-2); \
    } \
    type nan = (type)NAN; \
    for (int i = 0; i < M * N; ++i) c[i] = nan; \
    assert(ts_blas_gemm_##suffix(M, N, K, 1, a, K, 1, b, 1, K, 0, c, N, 1) == 0); \
    for (int i = 0; i < M * N; ++i) assert(c[i] == c[i]); \
    for (int i = 0; i < M * N; ++i) c[i] = 1; \
    for (int i = 0; i < M * K; ++i) a[i] = nan; \
    assert(ts_blas_gemm_##suffix(M, N, K, 0, a, K, 1, b, 1, K, 1, c, N, 1) == 0); \
    for (int i = 0; i < M * N; ++i) assert(c[i] == 1); \
    fail_allocation = 1; \
    assert(ts_blas_gemm_##suffix(M, N, K, 1, a, K, 1, b, 1, K, 1, c, N, 1) == 1); \
    fail_allocation = 0; \
} \
static void check_blas_rank_##suffix(void) { \
    enum { N = 150, K = 270 }; \
    static type a[N * K], b[N * K], c[N * N], expected[N * N]; \
    for (int i = 0; i < N * K; ++i) { a[i] = blas_value_##suffix(i); b[i] = blas_value_##suffix(i + 5); } \
    for (int second = 0; second < 2; ++second) \
        for (int upper = 0; upper < 2; ++upper) { \
            for (int i = 0; i < N * N; ++i) c[i] = expected[i] = blas_value_##suffix(i + 9); \
            for (int i = 0; i < N; ++i) \
                for (int j = upper ? i : 0; j < (upper ? N : i + 1); ++j) { \
                    type sum = 0; \
                    for (int p = 0; p < K; ++p) { \
                        sum += a[i * K + p] * (second ? b[j * K + p] : a[j * K + p]); \
                        if (second) sum += b[i * K + p] * a[j * K + p]; \
                    } \
                    expected[i * N + j] = (type)0.5 * expected[i * N + j] + (type)0.75 * sum; \
                } \
            assert(ts_blas_rank_##suffix(N, K, (type)0.75, a, K, 1, second ? b : a, K, 1, (type)0.5, \
                                         c, N, 1, upper, second) == 0); \
            for (int i = 0; i < N * N; ++i) assert(fabs((double)(c[i] - expected[i])) < 1e-2); \
        } \
} \
static void check_blas_solve_##suffix(void) { \
    enum { N = 300, C = 7 }; \
    static type a[N * N], at[N * N], x[N * C], b[N * C], y[N], z[N]; \
    for (int i = 0; i < N; ++i) \
        for (int j = 0; j < N; ++j) a[i * N + j] = blas_value_##suffix(i * N + j) / N + (i == j ? 2 : 0); \
    for (int i = 0; i < N * C; ++i) x[i] = b[i] = blas_value_##suffix(i + 1); \
    for (int upper = 0; upper < 2; ++upper) { \
        for (int unit = 0; unit < 2; ++unit) { \
            for (int i = 0; i < N * C; ++i) b[i] = x[i]; \
            assert(ts_blas_triangular_##suffix(0, 1, upper, unit, 1, a, N, 1, b, C, 1, N, C) == 0); \
            assert(ts_blas_triangular_##suffix(1, 1, upper, unit, 1, a, N, 1, b, C, 1, N, C) == 0); \
            for (int i = 0; i < N * C; ++i) assert(fabs((double)(b[i] - x[i])) < 1e-3); \
        } \
        for (int i = 0; i < N; ++i) y[i] = x[i * C]; \
        assert(ts_blas_trsv_##suffix(upper, 0, a, N, 1, N, y, 1) == 0); \
        for (int i = 0; i < N; ++i) { \
            type sum = 0; \
            for (int j = upper ? i : 0; j < (upper ? N : i + 1); ++j) sum += a[i * N + j] * y[j]; \
            assert(fabs((double)(sum - x[i * C])) < 1e-3); \
        } \
        for (int i = 0; i < N; ++i) \
            for (int j = 0; j < N; ++j) at[j * N + i] = a[i * N + j]; \
        for (int unit = 0; unit < 2; ++unit) { \
            for (int i = 0; i < N; ++i) y[i] = z[i] = x[i * C]; \
            assert(ts_blas_trsv_##suffix(upper, unit, a, N, 1, N, y, 1) == 0); \
            assert(ts_blas_trsv_##suffix(upper, unit, at, 1, N, N, z, 1) == 0); \
            for (int i = 0; i < N; ++i) assert(fabs((double)(y[i] - z[i])) < 1e-3); \
        } \
    } \
} \
static void check_blas_level2_##suffix(void) { \
    enum { M = 23, N = 41 }; \
    static type a[M * N], x[N], y[M], expected[M], row[N]; \
    for (int i = 0; i < M * N; ++i) a[i] = blas_value_##suffix(i); \
    for (int i = 0; i < N; ++i) x[i] = blas_value_##suffix(i + 3); \
    for (int i = 0; i < M; ++i) y[i] = expected[i] = blas_value_##suffix(i + 9); \
    for (int i = 0; i < M; ++i) { \
        type sum = 0; \
        for (int j = 0; j < N; ++j) sum += a[i * N + j] * x[j]; \
        expected[i] = (type)0.5 * expected[i] + (type)0.75 * sum; \
    } \
    assert(ts_blas_gemv_##suffix(M, N, (type)0.75, a, N, 1, x, 1, (type)0.5, y, 1) == 0); \
    for (int i = 0; i < M; ++i) assert(fabs((double)(y[i] - expected[i])) < 1e-3); \
    for (int i = 0; i < M; ++i) y[i] = blas_value_##suffix(i + 9); \
    assert(ts_blas_gemv_##suffix(M, N, (type)0.75, a, N, 1, x, 1, (type)0.5, y, 1) == 0); \
    for (int i = 0; i < M; ++i) assert(y[i] == y[i]); \
    for (int j = 0; j < N; ++j) row[j] = a[j]; \
    assert(ts_blas_ger_##suffix(M, N, 2, y, 1, x, 1, a, N, 1) == 0); \
    for (int j = 0; j < N; ++j) assert(fabs((double)(a[j] - (row[j] + 2 * y[0] * x[j]))) < 1e-3); \
}

CHECK_BLAS(f32, float)
CHECK_BLAS(f64, double)
#ifdef TS_BLAS_DISPATCH
CHECK_BLAS(f32_sse, float)
CHECK_BLAS(f64_sse, double)
CHECK_BLAS(f32_fma, float)
CHECK_BLAS(f64_fma, double)
#endif

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

#define CHECK_ROWS(suffix, type) \
static void check_rows_##suffix(void) { \
    type a[1800], b[600], out[4] = {-9, -9, -9, -9}; \
    for (size_t i = 0; i < 1800; ++i) a[i] = (type)((int)(i % 13) - 6); \
    for (size_t i = 0; i < 600; ++i) b[i] = (type)(i % 7 + 1); \
    const type *inputs[] = {a, b}; \
    int64_t strides[] = {600, 0}; \
    uint8_t code[] = {TS_OP_MULTIPLY, 0, 8, 9, TS_OP_SPILL, 0, 0, 0, \
                      TS_OP_RELOAD, 1, 0, 0, TS_OP_COPY, TS_KERNEL_OUTPUT, 1, 0}; \
    size_t lengths[] = {0, 1, 3, 4, 5, 255, 256, 257, 600}; \
    for (size_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) { \
        size_t before = allocations; \
        assert(ts_kernel_sum_rows_##suffix(code, sizeof(code), NULL, inputs, 2, strides, out, 3, lengths[l], 1) == 0); \
        assert(allocations == before + (lengths[l] != 0) && allocations == releases); \
        for (size_t r = 0; r < 3; ++r) { \
            const type *row[] = {a + r * 600, b}; \
            type expected; \
            assert(ts_kernel_sum_##suffix(code, sizeof(code), NULL, row, &expected, lengths[l], 1) == 0); \
            assert(memcmp(out + r, &expected, sizeof(type)) == 0); \
        } \
        assert(out[3] == -9); \
    } \
    inputs[0] = a + 1200; strides[0] = -600; \
    assert(ts_kernel_sum_rows_##suffix(code, sizeof(code), NULL, inputs, 2, strides, out, 3, 600, 1) == 0); \
    for (size_t r = 0; r < 3; ++r) { \
        const type *row[] = {a + (2 - r) * 600, b}; type expected; \
        assert(ts_kernel_sum_##suffix(code, sizeof(code), NULL, row, &expected, 600, 1) == 0); \
        assert(out[r] == expected); \
    } \
    size_t before = allocations; \
    assert(ts_kernel_sum_rows_##suffix(NULL, 0, NULL, NULL, 0, NULL, NULL, 0, SIZE_MAX, SIZE_MAX) == 0); \
    assert(ts_kernel_sum_rows_##suffix(NULL, 0, NULL, NULL, 0, NULL, out, 3, 0, SIZE_MAX) == 0); \
    assert(allocations == before); \
    uint8_t plain[] = {TS_OP_MULTIPLY, TS_KERNEL_OUTPUT, 8, 9}; \
    assert(ts_kernel_sum_rows_##suffix(plain, sizeof(plain), NULL, inputs, 2, strides, out, 3, 3, 0) == 0); \
    assert(allocations == before); \
    fail_allocation = 1; \
    assert(ts_kernel_sum_rows_##suffix(code, sizeof(code), NULL, inputs, 2, strides, out, 3, 3, 1) == -1); \
    fail_allocation = 0; \
    --allocations; \
    assert(allocations == releases); \
    before = allocations; \
    assert(ts_kernel_sum_rows_##suffix(code, sizeof(code), NULL, inputs, 2, strides, out, 3, 3, SIZE_MAX) == -1); \
    strides[0] = INT64_MIN; \
    assert(ts_kernel_sum_rows_##suffix(code, sizeof(code), NULL, inputs, 2, strides, out, 3, 3, 1) == -1); \
    assert(allocations == before); \
    strides[0] = 600; inputs[0] = a; a[600] = -1; a[0] = 1; \
    uint8_t root[] = {TS_OP_SQRT, 0, 8, 8, TS_OP_SPILL, 0, 0, 0, \
                      TS_OP_RELOAD, 1, 0, 0, TS_OP_COPY, TS_KERNEL_OUTPUT, 1, 0}; \
    assert(ts_kernel_sum_rows_##suffix(root, sizeof(root), NULL, inputs, 2, strides, out, 3, 1, 1) == -2); \
    assert(allocations == releases); \
    a[600] = 1; a[1200] = 1; \
    assert(ts_kernel_sum_rows_##suffix(root, sizeof(root), NULL, inputs, 2, strides, out, 3, 1, 1) == 0); \
    assert(allocations == releases); \
}

CHECK_ROWS(f32, float)
CHECK_ROWS(f64, double)

#define CHECK_INPUTS(suffix, type, precision) \
static void check_inputs_##suffix(void) { \
    type x[600], scale[20], out[600], value; \
    int8_t q[600]; \
    for (size_t i = 0; i < 600; ++i) { x[i] = 0.5; q[i] = (int8_t)((int)(i % 17) - 8); } \
    for (size_t i = 0; i < 20; ++i) scale[i] = (type)(i + 1); \
    ts_kernel_input inputs[] = {{x, 600, 0, precision, 1, 0, 0}, \
                                {q, 600, 0, 2, 1, 0, 0}, \
                                {scale, 20, 0, precision, 32, 0, 0}}; \
    uint8_t code[] = {TS_OP_MULTIPLY, 0, 8, 9, TS_OP_MULTIPLY, TS_KERNEL_OUTPUT, 0, 10}; \
    size_t lengths[] = {0, 1, 3, 4, 5, 31, 32, 33, 255, 256, 257, 600}; \
    size_t index; double wide; \
    for (size_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) { \
        size_t n = lengths[l], before = allocations; \
        assert(ts_kernel_inputs_##suffix(code, sizeof(code), NULL, inputs, 3, out, 1, n, \
                                         0, TS_INPUT_OUTPUT, 1, &wide, &index) == 0); \
        assert(allocations == before + (n != 0) && allocations == releases); \
        type expected = 0; \
        for (size_t i = 0; i < n; ++i) { \
            type term = (x[i] * (type)q[i]) * scale[i / 32]; \
            assert(out[i] == term); expected += term; \
        } \
        assert(ts_kernel_inputs_##suffix(code, sizeof(code), NULL, inputs, 3, &value, 1, n, \
                                         0, TS_INPUT_SUM, 1, &wide, &index) == 0); \
        assert(value == expected && allocations == releases); \
    } \
    inputs[0].start = 5; inputs[1].start = 5; inputs[2].phase = 5; \
    assert(ts_kernel_inputs_##suffix(code, sizeof(code), NULL, inputs, 3, out, 1, 285, \
                                     0, TS_INPUT_OUTPUT, 1, &wide, &index) == 0); \
    for (size_t i = 0; i < 285; ++i) assert(out[i] == (x[5+i] * (type)q[5+i]) * scale[(5+i)/32]); \
    inputs[0].start = 0; inputs[1].start = 0; inputs[2].phase = 0; \
    inputs[0].stride = 257; inputs[1].stride = 257; inputs[2].stride = 9; \
    size_t before = allocations; \
    assert(ts_kernel_inputs_##suffix(code, sizeof(code), NULL, inputs, 3, out, 2, 257, \
                                     1, TS_INPUT_SUM, 1, &wide, &index) == 0); \
    assert(allocations == before + 2 && allocations == releases); \
    for (size_t row = 0; row < 2; ++row) { \
        type expected = 0; \
        for (size_t i = 0; i < 257; ++i) expected += x[row*257+i] * (type)q[row*257+i] * scale[row*9+i/32]; \
        assert(out[row] == expected); \
    } \
    inputs[0].start = 257; inputs[1].start = 257; inputs[2].start = 9; \
    inputs[0].stride = -257; inputs[1].stride = -257; inputs[2].stride = -9; \
    assert(ts_kernel_inputs_##suffix(code, sizeof(code), NULL, inputs, 3, out, 2, 257, \
                                     0, TS_INPUT_SUM, 1, &wide, &index) == 0); \
    before = allocations; inputs[2].repeat = 0; \
    assert(ts_kernel_inputs_##suffix(code, sizeof(code), NULL, inputs, 3, out, 2, 257, \
                                     0, TS_INPUT_SUM, 1, &wide, &index) == -1); \
    assert(allocations == before); inputs[2].repeat = 32; \
    inputs[2].length = 1; \
    assert(ts_kernel_inputs_##suffix(code, sizeof(code), NULL, inputs, 3, out, 2, 257, \
                                     0, TS_INPUT_SUM, 1, &wide, &index) == -1); \
    inputs[2].length = 20; inputs[2].stride = INT64_MIN; \
    assert(ts_kernel_inputs_##suffix(code, sizeof(code), NULL, inputs, 3, out, 2, 257, \
                                     0, TS_INPUT_SUM, 1, &wide, &index) == -1); \
    inputs[2].stride = -9; \
    before = releases; fail_allocation = 1; \
    assert(ts_kernel_inputs_##suffix(code, sizeof(code), NULL, inputs, 3, out, 2, 257, \
                                     1, TS_INPUT_SUM, 1, &wide, &index) == -1); \
    assert(releases == before); allocations -= 2; fail_allocation = 0; \
    fail_allocation_at = allocations + 2; before = releases; \
    assert(ts_kernel_inputs_##suffix(code, sizeof(code), NULL, inputs, 3, out, 2, 257, \
                                     1, TS_INPUT_SUM, 1, &wide, &index) == -1); \
    assert(releases == before + 1); --allocations; fail_allocation_at = 0; \
    type negative = -1; ts_kernel_input bad = {&negative, 1, 0, precision, 1, 0, 0}; \
    uint8_t root[] = {TS_OP_SQRT, TS_KERNEL_OUTPUT, 8, 8}; \
    assert(ts_kernel_inputs_##suffix(root, sizeof(root), NULL, &bad, 1, out, 1, 1, \
                                     1, TS_INPUT_OUTPUT, 1, &wide, &index) == -2); \
    assert(allocations == releases); \
}

CHECK_INPUTS(f32, float, 0)
CHECK_INPUTS(f64, double, 1)

#define CHECK_LOADER_SOURCE(suffix, type, id, source) \
static void check_loader_##id##_##suffix(void) { \
    source values[2048]; \
    uint64_t boundaries[] = {0, 1, 127, 128, 255, 32767, 32768, 65535, \
        UINT64_C(16777217), UINT64_C(4294967295), UINT64_C(9007199254740993), \
        (UINT64_C(1) << 62) + (UINT64_C(1) << 38) + 1, \
        (UINT64_C(1) << 63) - 1, UINT64_C(1) << 63, UINT64_MAX}; \
    for (size_t i = 0; i < 2048; ++i) values[i] = (source)boundaries[i % 15]; \
    size_t repeats[] = {1, 3, 32, 257}; \
    size_t lengths[] = {1, 2, 3, 4, 7, 8, 9, 31, 32, 33, 255, 256, 257, 600}; \
    for (size_t r = 0; r < 4; ++r) { \
        for (size_t phase = 0; phase < repeats[r]; phase += repeats[r] > 1 ? repeats[r] - 1 : 1) { \
            for (size_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) { \
                ts_kernel_input input = {values, 2048, 700, id, repeats[r], phase, -650}; \
                for (size_t row = 0; row < 2; ++row) { \
                    for (size_t base = 0; base < lengths[l]; base += TS_KERNEL_BLOCK) { \
                        size_t n = lengths[l] - base; \
                        if (n > TS_KERNEL_BLOCK) n = TS_KERNEL_BLOCK; \
                        type storage[TS_KERNEL_BLOCK + 2], *buffer = storage + 1; \
                        storage[0] = storage[n + 1] = (type)-12345; \
                        const type *prepared; \
                        ts_prepare_inputs_##suffix(&input, 1, row, base, n, buffer, &prepared); \
                        assert(prepared == buffer); \
                        for (size_t i = 0; i < n; ++i) { \
                            size_t index = 700 - row * 650 + (phase + base + i) / repeats[r]; \
                            type expected = (type)*(volatile source *)(values + index); \
                            assert(memcmp(prepared + i, &expected, sizeof(type)) == 0); \
                        } \
                        assert(storage[0] == (type)-12345 && storage[n + 1] == (type)-12345); \
                    } \
                } \
            } \
        } \
    } \
}

#define CHECK_LOADER_TYPES(suffix, type) \
CHECK_LOADER_SOURCE(suffix, type, 2, int8_t) \
CHECK_LOADER_SOURCE(suffix, type, 3, uint8_t) \
CHECK_LOADER_SOURCE(suffix, type, 4, int16_t) \
CHECK_LOADER_SOURCE(suffix, type, 5, uint16_t) \
CHECK_LOADER_SOURCE(suffix, type, 6, int32_t) \
CHECK_LOADER_SOURCE(suffix, type, 7, uint32_t) \
CHECK_LOADER_SOURCE(suffix, type, 8, int64_t) \
CHECK_LOADER_SOURCE(suffix, type, 9, uint64_t)

CHECK_LOADER_TYPES(f32, float)
CHECK_LOADER_TYPES(f64, double)

#define CHECK_FLOAT_LOADER(suffix, type, precision) \
static void check_float_loader_##suffix(void) { \
    type values[] = {0.0, -0.0, 1.0, -1.0}; \
    ts_kernel_input input = {values, 4, 0, precision, 32, 31, 0}; \
    type buffer[TS_KERNEL_BLOCK]; \
    const type *prepared; \
    ts_prepare_inputs_##suffix(&input, 1, 0, 0, 65, buffer, &prepared); \
    for (size_t i = 0; i < 65; ++i) \
        assert(memcmp(prepared + i, values + (31 + i) / 32, sizeof(type)) == 0); \
    input.repeat = 1; input.phase = 0; \
    ts_prepare_inputs_##suffix(&input, 1, 0, 1, 3, buffer, &prepared); \
    assert(prepared == values + 1); \
}

CHECK_FLOAT_LOADER(f32, float, 0)
CHECK_FLOAT_LOADER(f64, double, 1)

#define CHECK_TRANSCENDENTALS(suffix, type, precision, exps, sins, coss) \
static void check_transcendentals_##suffix(void) { \
    type x[513], out[513], sum; \
    const type *inputs[] = {x}; \
    const uint8_t ops[] = {TS_OP_EXP, TS_OP_SIN, TS_OP_COS}; \
    size_t lengths[] = {0, 1, 3, 5, 255, 256, 257, 513}; \
    for (size_t i = 0; i < 513; ++i) x[i] = (type)((int)(i % 17) - 8) / 4; \
    for (size_t op = 0; op < 3; ++op) { \
        uint8_t code[] = {ops[op], TS_KERNEL_OUTPUT, 8, 0}; \
        for (size_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) { \
            size_t n = lengths[l]; type expected_sum = 0; \
            assert(ts_kernel_##suffix(code, sizeof(code), NULL, inputs, out, n, 0) == 0); \
            for (size_t i = 0; i < n; ++i) { \
                type expected = op == 0 ? exps(x[i]) : op == 1 ? sins(x[i]) : coss(x[i]); \
                assert(out[i] == expected); expected_sum += expected; \
            } \
            assert(ts_kernel_sum_##suffix(code, sizeof(code), NULL, inputs, &sum, n, 0) == 0); \
            assert(fabs((double)sum - expected_sum) <= (precision ? 1e-12 : 1e-5) * fmax(1, fabs(expected_sum))); \
            ts_kernel_input descriptor = {x, 513, 0, precision, 1, 0, 0}; \
            double wide; size_t index; \
            assert(ts_kernel_inputs_##suffix(code, sizeof(code), NULL, &descriptor, 1, out, 1, n, \
                                             0, TS_INPUT_OUTPUT, 1, &wide, &index) == 0); \
            for (size_t i = 0; i < n; ++i) \
                assert(out[i] == (op == 0 ? exps(x[i]) : op == 1 ? sins(x[i]) : coss(x[i]))); \
        } \
    } \
    x[0] = -(type)0; x[1] = (type)1; x[2] = (type)INFINITY; x[3] = (type)NAN; \
    uint8_t sine[] = {TS_OP_SIN, TS_KERNEL_OUTPUT, 8, 0}; \
    assert(ts_kernel_##suffix(sine, sizeof(sine), NULL, inputs, out, 4, 0) == 0); \
    assert(signbit(out[0]) && isnan(out[2]) && isnan(out[3])); \
    assert(fabs((double)out[1] - 0.84147098480789650665) < (precision ? 1e-15 : 1e-7)); \
    uint8_t cosine[] = {TS_OP_COS, TS_KERNEL_OUTPUT, 8, 0}; \
    assert(ts_kernel_##suffix(cosine, sizeof(cosine), NULL, inputs, out, 4, 0) == 0); \
    assert(out[0] == 1 && isnan(out[2]) && isnan(out[3])); \
    uint8_t exponential[] = {TS_OP_EXP, TS_KERNEL_OUTPUT, 8, 0}; \
    assert(ts_kernel_##suffix(exponential, sizeof(exponential), NULL, inputs, out, 4, 0) == 0); \
    assert(out[0] == 1 && isinf(out[2]) && isnan(out[3])); \
    assert(fabs((double)out[1] - 2.71828182845904523536) < (precision ? 1e-15 : 1e-7)); \
    assert(allocations == releases); \
}
CHECK_TRANSCENDENTALS(f32, float, 0, expf, sinf, cosf)
CHECK_TRANSCENDENTALS(f64, double, 1, exp, sin, cos)

#include "native-complex-kernel.h"

int main(void) {
    check_complex_c32();
    check_complex_c64();
    check_loader_2_f32();
    check_loader_3_f32();
    check_loader_4_f32();
    check_loader_5_f32();
    check_loader_6_f32();
    check_loader_7_f32();
    check_loader_8_f32();
    check_loader_9_f32();
    check_loader_2_f64();
    check_loader_3_f64();
    check_loader_4_f64();
    check_loader_5_f64();
    check_loader_6_f64();
    check_loader_7_f64();
    check_loader_8_f64();
    check_loader_9_f64();
    check_float_loader_f32();
    check_float_loader_f64();
    check_transcendentals_f32();
    check_transcendentals_f64();
    check_inputs_f32();
    check_inputs_f64();
    check_rows_f32();
    check_rows_f64();
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
    check_blas_f32();
    check_blas_f64();
    check_blas_solve_f32();
    check_blas_rank_f32();
    check_blas_solve_f64();
    check_blas_rank_f64();
    check_blas_level2_f32();
    check_blas_level2_f64();
#ifdef TS_BLAS_DISPATCH
    check_blas_f32_sse();
    check_blas_f64_sse();
    check_blas_solve_f32_sse();
    check_blas_rank_f32_sse();
    check_blas_solve_f64_sse();
    check_blas_rank_f64_sse();
    check_blas_level2_f32_sse();
    check_blas_level2_f64_sse();
    printf("BLAS AVX+FMA checks: %s\n", ts_fma_supported() ? "run" : "skipped");
    if (ts_fma_supported()) {
        check_blas_f32_fma();
        check_blas_f64_fma();
        check_blas_solve_f32_fma();
        check_blas_rank_f32_fma();
        check_blas_solve_f64_fma();
        check_blas_rank_f64_fma();
        check_blas_level2_f32_fma();
        check_blas_level2_f64_fma();
    }
#endif
    return 0;
}
