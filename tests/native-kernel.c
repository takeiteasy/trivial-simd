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
        } \
    } \
    uint8_t constant[] = {TS_OP_CONSTANT, TS_KERNEL_OUTPUT, 0, 0}; \
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
    fail_allocation = 0; \
}

CHECK_KERNEL(f32, float)
CHECK_KERNEL(f64, double)

int main(void) {
    check_f32();
    check_f64();
    return 0;
}
