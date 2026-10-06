#define CHECK_COMPLEX_KERNEL(suffix, type, tolerance, large, tiny) \
static void check_complex_##suffix(void) { \
    type a[1200], b[1200], out[1200], value[2], constants[] = {2, 0}; \
    const type *inputs[] = {a, b}; uint8_t mask[600]; \
    const uint8_t equal[] = {TS_OP_EQ, TS_KERNEL_OUTPUT, 8, 9}; \
    const uint8_t unequal[] = {TS_OP_NE, TS_KERNEL_OUTPUT, 8, 9}; \
    const uint8_t choice[] = {TS_OP_EQ, 0, 8, 9, TS_OP_COPY, 1, 9, 0, \
                             TS_OP_SELECT, 1, 0, 8, TS_OP_COPY, TS_KERNEL_OUTPUT, 1, 0}; \
    const uint8_t spilled[] = {TS_OP_EQ, 0, 8, 9, TS_OP_SPILL, 0, 0, 1, \
                              TS_OP_RELOAD, 1, 0, 1, TS_OP_COPY, TS_KERNEL_OUTPUT, 1, 0}; \
    size_t lengths[] = {0, 1, 2, 3, 4, 5, 255, 256, 257, 600}; \
    for (size_t i = 0; i < 600; ++i) { \
        a[2 * i] = 3; a[2 * i + 1] = 4; b[2 * i] = 3; b[2 * i + 1] = i % 3 ? -4 : 4; \
    } \
    for (size_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) { \
        size_t n = lengths[l], result = 9, before = allocations; double norm; \
        memset(mask, 9, sizeof(mask)); \
        assert(ts_kernel_mask_##suffix(equal, sizeof(equal), NULL, inputs, 2, mask, n, 0, 0, NULL) == 0); \
        for (size_t i = 0; i < 600; ++i) assert(mask[i] == (i < n ? i % 3 == 0 : 9)); \
        assert(ts_kernel_mask_##suffix(unequal, sizeof(unequal), NULL, inputs, 2, mask, n, 0, 0, NULL) == 0); \
        for (size_t i = 0; i < n; ++i) assert(mask[i] == (i % 3 != 0)); \
        assert(ts_kernel_mask_##suffix(spilled, sizeof(spilled), NULL, inputs, 2, NULL, n, 257, 1, &result) == 0); \
        assert(result == (n + 2) / 3 && allocations == before + (n != 0)); \
        assert(allocations == releases); \
        assert(ts_kernel_mask_##suffix(equal, sizeof(equal), NULL, inputs, 2, NULL, n, 0, 2, &result) == 0); \
        assert(result == (n != 0)); \
        assert(ts_kernel_mask_##suffix(equal, sizeof(equal), NULL, inputs, 2, NULL, n, 0, 3, &result) == 0); \
        assert(result == (n < 2)); \
        for (size_t i = 0; i < 1200; ++i) out[i] = -9; \
        assert(ts_kernel_##suffix(choice, sizeof(choice), NULL, inputs, out, n, 0) == 0); \
        for (size_t i = 0; i < 1200; ++i) assert(out[i] == (i < 2 * n ? b[i] : -9)); \
        assert(ts_kernel_reduction_##suffix(choice, sizeof(choice), NULL, inputs, 2, n, 0, 0, 1, value, &norm, NULL) == 0); \
        type re = 0, im = 0; \
        for (size_t i = 0; i < n; ++i) { re += b[2 * i]; im += b[2 * i + 1]; } \
        assert(value[0] == re && value[1] == im); \
        assert(ts_kernel_reduction_##suffix(choice, sizeof(choice), NULL, inputs, 2, n, 0, TS_REDUCE_ASUM, 1, value, &norm, NULL) == 0); \
        assert(value[0] == 5 * n); \
        assert(ts_kernel_reduction_##suffix(choice, sizeof(choice), NULL, inputs, 2, n, 0, TS_REDUCE_SUMSQ, 1, value, &norm, NULL) == 0); \
        assert(fabs(norm - 5 * sqrt((double)n)) <= tolerance * fmax(1, norm)); \
    } \
    const uint8_t multiply[] = {TS_OP_MULTIPLY, TS_KERNEL_OUTPUT, 8, 9}; \
    assert(ts_kernel_##suffix(multiply, sizeof(multiply), NULL, inputs, out, 600, 0) == 0); \
    for (size_t i = 0; i < 600; ++i) { \
        assert(out[2 * i] == (i % 3 ? 25 : -7)); assert(out[2 * i + 1] == (i % 3 ? 0 : 24)); \
    } \
    const uint8_t root[] = {TS_OP_SQRT, TS_KERNEL_OUTPUT, 8, 0}; \
    assert(ts_kernel_##suffix(root, sizeof(root), NULL, inputs, out, 1, 0) == 0); \
    assert(out[0] == 2 && out[1] == 1); \
    a[0] = -3; a[1] = -4; \
    assert(ts_kernel_##suffix(root, sizeof(root), NULL, inputs, out, 1, 0) == 0); \
    assert(out[0] == 1 && out[1] == -2); \
    a[0] = -0.0; a[1] = -0.0; \
    assert(ts_kernel_##suffix(root, sizeof(root), NULL, inputs, out, 1, 0) == 0); \
    assert(out[0] == 0 && out[1] == 0 && signbit(out[1])); \
    const uint8_t divide[] = {TS_OP_DIVIDE, TS_KERNEL_OUTPUT, 8, 9}; \
    for (size_t trial = 0; trial < 2; ++trial) { \
        a[0] = trial ? tiny : large; a[1] = -a[0]; b[0] = a[0]; b[1] = a[0]; \
        assert(ts_kernel_##suffix(divide, sizeof(divide), NULL, inputs, out, 1, 0) == 0); \
        assert(out[0] == 0 && fabs(out[1] + 1) < tolerance); \
        double norm; \
        const uint8_t copy[] = {TS_OP_COPY, TS_KERNEL_OUTPUT, 8, 0}; \
        assert(ts_kernel_reduction_##suffix(copy, sizeof(copy), NULL, inputs, 1, 1, 0, TS_REDUCE_SUMSQ, 1, value, &norm, NULL) == 0); \
        assert(fabs(norm / hypot((double)a[0], (double)a[1]) - 1) < tolerance); \
    } \
    b[0] = b[1] = 0; \
    out[0] = out[1] = -9; \
    assert(ts_kernel_##suffix(divide, sizeof(divide), NULL, inputs, out, 1, 0) == -3); \
    assert(out[0] == -9 && out[1] == -9); \
    uint8_t invalid[] = {TS_OP_LT, TS_KERNEL_OUTPUT, 8, 9}; \
    assert(ts_kernel_##suffix(invalid, sizeof(invalid), NULL, inputs, out, 1, 0) == -4); \
    assert(ts_kernel_mask_##suffix(equal, sizeof(equal), NULL, inputs, 1, mask, 1, 0, 0, NULL) == -4); \
    assert(ts_kernel_mask_##suffix(equal, sizeof(equal) - 1, NULL, inputs, 2, mask, 1, 0, 0, NULL) == -4); \
    assert(ts_kernel_mask_##suffix(equal, sizeof(equal), NULL, inputs, 249, mask, 1, 0, 0, NULL) == -4); \
    assert(ts_kernel_mask_##suffix(spilled, sizeof(spilled), NULL, inputs, 2, mask, 1, 256, 0, NULL) == -4); \
    assert(ts_kernel_mask_##suffix(equal, sizeof(equal), NULL, inputs, 2, mask, 1, SIZE_MAX, 0, NULL) == -1); \
    size_t before = allocations; fail_allocation = 1; \
    assert(ts_kernel_mask_##suffix(spilled, sizeof(spilled), NULL, inputs, 2, mask, 1, 257, 0, NULL) == -1); \
    assert(allocations == before + 1 && releases == before); \
    --allocations; fail_allocation = 0; \
    assert(ts_kernel_mask_##suffix(spilled, sizeof(spilled), NULL, inputs, 2, mask, 600, 257, 0, NULL) == 0); \
    assert(allocations == releases); \
    const uint8_t constant[] = {TS_OP_CONSTANT, TS_KERNEL_OUTPUT, 0, 0}; \
    assert(ts_kernel_##suffix(constant, sizeof(constant), constants, NULL, out, 3, 0) == 0); \
    assert(out[0] == 2 && out[1] == 0 && out[4] == 2 && out[5] == 0); \
}

CHECK_COMPLEX_KERNEL(c32, float, 1e-6, 1e30f, 1e-30f)
CHECK_COMPLEX_KERNEL(c64, double, 1e-12, 1e300, 1e-300)
