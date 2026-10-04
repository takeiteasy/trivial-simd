#include <fenv.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif
#include <stdio.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdlib.h>
#include "../native/extended.c"

static double encoded_value(unsigned bits, unsigned fraction_bits, unsigned exponent_bits, int bias) {
    unsigned fraction_mask = (1u << fraction_bits) - 1u;
    unsigned exponent = bits >> fraction_bits;
    unsigned fraction = bits & fraction_mask;
    if (exponent == (1u << exponent_bits) - 1u)
        return ldexp(1.0, (int)exponent - bias);
    return ldexp(exponent ? 1.0 + (double)fraction / (1u << fraction_bits) :
                           (double)fraction / (1u << fraction_bits),
                 (exponent ? (int)exponent : 1) - bias);
}

/* Search representable values rather than reproducing the bit-rounding algorithm. */
static uint16_t reference_encode(uint32_t bits, int f16, int rounding) {
    unsigned fraction_bits = f16 ? 10 : 7, exponent_bits = f16 ? 5 : 8;
    int bias = f16 ? 15 : 127;
    unsigned infinity = ((1u << exponent_bits) - 1u) << fraction_bits;
    unsigned sign = (bits >> 16) & 0x8000u, magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u)
        return (uint16_t)(sign | infinity | (magnitude > 0x7f800000u
            ? (1u << (fraction_bits - 1)) | ((magnitude >> (23 - fraction_bits)) & ((1u << fraction_bits) - 1u))
            : 0u));
    double value = (double)ts_bits_float(magnitude);
    unsigned low = 0, high = infinity;
    while (low < high) {
        unsigned middle = (low + high + 1) / 2;
        if (encoded_value(middle, fraction_bits, exponent_bits, bias) <= value) low = middle;
        else high = middle - 1;
    }
    double lower = encoded_value(low, fraction_bits, exponent_bits, bias);
    unsigned result = low;
    if (rounding == TS_ROUND_TRUNCATE || (rounding == TS_ROUND_FLOOR && !sign) ||
        (rounding == TS_ROUND_CEILING && sign)) {
        if (result == infinity) --result;
    } else if (low < infinity && value != lower) {
        if (rounding != TS_ROUND_NEAREST_EVEN) ++result;
        else {
            double upper = encoded_value(low + 1, fraction_bits, exponent_bits, bias);
            double midpoint = (lower + upper) / 2;
            if (value > midpoint || (value == midpoint && (low & 1))) ++result;
        }
    }
    return (uint16_t)(sign | result);
}

static void check_results(const float *input, const uint16_t *expected, size_t n, int f16, int rounding) {
    uint16_t actual[1026];
    for (size_t i = 0; i < n + 2; ++i) actual[i] = 0x1234;
    if (f16) ts_extended_f32_to_f16(actual + 1, input, n, rounding);
    else ts_extended_f32_to_bf16(actual + 1, input, n, rounding);
    assert(actual[0] == 0x1234 && actual[n + 1] == 0x1234);
    for (size_t i = 0; i < n; ++i) {
        if (actual[i + 1] != expected[i]) {
            fprintf(stderr, "%s mode %d lane %zu input %08x: got %04x expected %04x\n",
                    f16 ? "f16" : "bf16", rounding, i, ts_float_bits(input[i]), actual[i + 1], expected[i]);
            abort();
        }
    }
}

static void check_block(const float *input, size_t n) {
    uint16_t expected[1024];
    for (int f16 = 0; f16 <= 1; ++f16) {
        for (int rounding = 0; rounding < 4; ++rounding) {
            for (size_t i = 0; i < n; ++i) {
                uint32_t bits = ts_float_bits(input[i]);
                expected[i] = reference_encode(bits, f16, rounding);
                uint16_t scalar = f16 ? ts_f32_to_f16_scalar(input[i], rounding) :
                                        ts_f32_to_bf16_scalar(input[i], rounding);
                assert(scalar == expected[i]);
            }
            check_results(input, expected, n, f16, rounding);
        }
    }
}

#if defined(__aarch64__)
static uint64_t read_fpcr(void) {
    uint64_t value;
    __asm__ volatile("mrs %0, fpcr" : "=r"(value) :: "memory");
    return value;
}
static uint64_t read_fpsr(void) {
    uint64_t value;
    __asm__ volatile("mrs %0, fpsr" : "=r"(value) :: "memory");
    return value;
}
#endif

static void check_environment(const float *input, size_t n) {
    uint16_t expected[1024];
    fenv_t saved;
    assert(fegetenv(&saved) == 0);
    const int caller_modes[] = {FE_TONEAREST, FE_TOWARDZERO, FE_DOWNWARD, FE_UPWARD};
    for (int f16 = 0; f16 <= 1; ++f16) {
        for (int rounding = 0; rounding < 4; ++rounding) {
            assert(fesetenv(&saved) == 0);
            for (size_t i = 0; i < n; ++i)
                expected[i] = reference_encode(ts_float_bits(input[i]), f16, rounding);
            for (size_t mode = 0; mode < 4; ++mode) {
                assert(fesetenv(&saved) == 0);
                assert(fesetround(caller_modes[mode]) == 0);
                assert(feraiseexcept(FE_DIVBYZERO) == 0);
#if defined(__aarch64__)
                uint64_t fpcr = read_fpcr() | (1ull << 24) | (1ull << 19) | (0x1full << 8);
                __asm__ volatile("msr fpcr, %0" :: "r"(fpcr) : "memory");
                fpcr = read_fpcr();
                uint64_t fpsr = read_fpsr();
#elif defined(__x86_64__) || defined(_M_X64)
                unsigned mxcsr = (_mm_getcsr() | 0x8040u) & ~0x1f80u;
                _mm_setcsr(mxcsr);
                /* Rosetta retains exception masks; compare the accepted settings. */
                mxcsr = _mm_getcsr();
#endif
                int flags = fetestexcept(FE_ALL_EXCEPT);
                check_results(input, expected, n, f16, rounding);
#if defined(__aarch64__)
                assert(read_fpcr() == fpcr && read_fpsr() == fpsr);
#elif defined(__x86_64__) || defined(_M_X64)
                assert(_mm_getcsr() == mxcsr);
#endif
                assert(fegetround() == caller_modes[mode]);
                assert(fetestexcept(FE_ALL_EXCEPT) == flags);
            }
        }
    }
    assert(fesetenv(&saved) == 0);
}

int main(void) {
    assert(ts_extended_float_encoding_rounding_modes() == 0xfu);
    const uint32_t edges[] = {
        0, 0x80000000u, 1, 0x80000001u, 0x007fffffu, 0x807fffffu,
        0x00008000u, 0x80008000u, 0x33000000u, 0xb3000000u, 0x33800000u, 0xb3800000u,
        0x387fffffu, 0xb87fffffu, 0x38800000u, 0xb8800000u,
        0x3f800001u, 0xbf800001u, 0x477fe000u, 0xc77fe000u, 0x477fe001u, 0xc77fe001u,
        0x477ff000u, 0xc77ff000u, 0x47800000u, 0xc7800000u, 0x7f7fffffu, 0xff7fffffu,
        0x7f800000u, 0xff800000u, 0x7f800001u, 0xff800001u, 0x7fc0ffffu, 0xffc0ffffu
    };
    float input[1024];
    for (size_t i = 0; i < 1024; ++i) input[i] = ts_bits_float(edges[i % (sizeof edges / sizeof *edges)]);
    for (size_t n = 0; n <= 19; ++n) check_block(input + 1, n);
    check_block(input, 1023);
    check_environment(input, 521);
    uint32_t state = 119;
    for (size_t batch = 0; batch < 128; ++batch) {
        for (size_t i = 0; i < 1024; ++i) {
            state = state * 1664525u + 1013904223u;
            input[i] = ts_bits_float(state);
        }
        check_block(input, 1024);
    }
    puts("Float encoding rounding and environment checks passed.");
    return 0;
}
