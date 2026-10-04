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
static uint16_t reference_encode_wide(uint64_t bits, int f64, int f16, int rounding) {
    unsigned fraction_bits = f16 ? 10 : 7, exponent_bits = f16 ? 5 : 8;
    int bias = f16 ? 15 : 127;
    unsigned infinity = ((1u << exponent_bits) - 1u) << fraction_bits;
    unsigned sign_shift = f64 ? 48 : 16, precision = f64 ? 52 : 23;
    uint64_t magnitude = bits & (f64 ? UINT64_C(0x7fffffffffffffff) : UINT64_C(0x7fffffff));
    uint64_t input_infinity = f64 ? UINT64_C(0x7ff0000000000000) : UINT64_C(0x7f800000);
    unsigned sign = (unsigned)(bits >> sign_shift) & 0x8000u;
    if (magnitude >= input_infinity)
        return (uint16_t)(sign | infinity | (magnitude > input_infinity
            ? (1u << (fraction_bits - 1)) | (unsigned)((magnitude >> (precision - fraction_bits)) & ((1u << fraction_bits) - 1u))
            : 0u));
    double value;
    if (f64) memcpy(&value, &magnitude, sizeof value);
    else value = (double)ts_bits_float((uint32_t)magnitude);
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

static uint16_t reference_encode(uint32_t bits, int f16, int rounding) {
    return reference_encode_wide(bits, 0, f16, rounding);
}

static uint64_t double_bits(double value) {
    uint64_t bits; memcpy(&bits, &value, sizeof bits); return bits;
}

static void check_f64_block(const double *input, size_t n, int f16, int rounding) {
    uint16_t output[1026];
    for (size_t i = 0; i < n + 2; ++i) output[i] = 0x1234;
    ts_extended_convert_encoded(output + 1, input, n, f16 ? 3 : 2, 1, rounding);
    assert(output[0] == 0x1234 && output[n + 1] == 0x1234);
    for (size_t i = 0; i < n; ++i)
        assert(output[i + 1] == reference_encode_wide(double_bits(input[i]), 1, f16, rounding));
}

static void check_extended(void) {
    assert(ts_extended_encoded_conversion_rounding_modes() == 0xfu);
    uint16_t input[1024], output[1024];
    double wide[1024];
    for (int f16 = 0; f16 <= 1; ++f16) {
        unsigned precision = f16 ? 10 : 7, exponent_bits = f16 ? 5 : 8;
        int bias = f16 ? 15 : 127;
        unsigned infinity = ((1u << exponent_bits) - 1) << precision;
        for (unsigned base = 0; base < 65536; base += 1024) {
            for (unsigned i = 0; i < 1024; ++i) input[i] = (uint16_t)(base + i);
            ts_extended_convert_encoded(wide, input, 1024, 1, f16 ? 3 : 2, 0);
            for (unsigned i = 0; i < 1024; ++i) {
                unsigned magnitude = input[i] & 0x7fff, fraction = magnitude & ((1u << precision) - 1);
                uint64_t sign = (uint64_t)(input[i] & 0x8000) << 48;
                uint64_t expected = sign | (magnitude >= infinity ?
                    UINT64_C(0x7ff0000000000000) | ((uint64_t)fraction << (52 - precision)) |
                    (fraction ? UINT64_C(0x8000000000000) : 0) :
                    double_bits(encoded_value(magnitude, precision, exponent_bits, bias)));
                assert(double_bits(wide[i]) == expected);
            }
            for (int rounding = 0; rounding < 4; ++rounding) {
                ts_extended_convert_encoded(output, input, 1024, f16 ? 2 : 3, f16 ? 3 : 2, rounding);
                for (unsigned i = 0; i < 1024; ++i)
                    assert(output[i] == reference_encode_wide(double_bits(wide[i]), 1, !f16, rounding));
                ts_extended_convert_encoded(output, input, 1024, f16 ? 3 : 2, f16 ? 3 : 2, rounding);
                assert(memcmp(input, output, sizeof input) == 0);
            }
        }
        for (unsigned base = 0; base < infinity; base += 128) {
            size_t n = 0;
            for (unsigned index = base; index < infinity && index < base + 128; ++index) {
                double midpoint = (encoded_value(index, precision, exponent_bits, bias) +
                                   encoded_value(index + 1, precision, exponent_bits, bias)) / 2;
                uint64_t middle = double_bits(midpoint);
                for (int delta = -1; delta <= 1; ++delta) {
                    uint64_t bits = middle + delta;
                    memcpy(wide + n++, &bits, sizeof bits);
                    bits |= UINT64_C(0x8000000000000000);
                    memcpy(wide + n++, &bits, sizeof bits);
                }
            }
            for (int rounding = 0; rounding < 4; ++rounding) check_f64_block(wide, n, f16, rounding);
        }
        const uint64_t edges[] = {0, UINT64_C(0x8000000000000000), 1, UINT64_C(0x8000000000000001),
            UINT64_C(0x7fefffffffffffff), UINT64_C(0xffefffffffffffff),
            UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
            UINT64_C(0x7ff0000000000001), UINT64_C(0xfff923456789abcd)};
        for (size_t i = 0; i < 1024; ++i) memcpy(wide + i, edges + i % (sizeof edges / sizeof *edges), sizeof(double));
        for (int rounding = 0; rounding < 4; ++rounding) {
            for (size_t n = 0; n <= 19; ++n) check_f64_block(wide + 1, n, f16, rounding);
            check_f64_block(wide, 513, f16, rounding);
        }
    }
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
    uint16_t expected[1024], encoded[1024], cross[1024], cross_expected[1024];
    double wide[1024], decoded[1024];
    float decoded32[1024];
    uint32_t decoded32_expected[1024];
    fenv_t saved;
    assert(fegetenv(&saved) == 0);
    const int caller_modes[] = {FE_TONEAREST, FE_TOWARDZERO, FE_DOWNWARD, FE_UPWARD};
    for (int f16 = 0; f16 <= 1; ++f16) {
        for (int rounding = 0; rounding < 4; ++rounding) {
            assert(fesetenv(&saved) == 0);
            for (size_t i = 0; i < n; ++i) {
                uint64_t bits = ts_f32_bits_to_f64_bits(ts_float_bits(input[i]));
                memcpy(wide + i, &bits, sizeof bits);
                expected[i] = reference_encode(ts_float_bits(input[i]), f16, rounding);
                encoded[i] = expected[i];
                float value = f16 ? ts_f16_to_f32_scalar(encoded[i]) : ts_bf16_to_f32_scalar(encoded[i]);
                decoded32_expected[i] = ts_float_bits(value);
                cross_expected[i] = reference_encode(decoded32_expected[i], !f16, rounding);
            }
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
                ts_extended_convert_encoded(encoded, wide, n, f16 ? 3 : 2, 1, rounding);
                assert(memcmp(encoded, expected, n * sizeof(uint16_t)) == 0);
                ts_extended_convert_encoded(decoded, encoded, n, 1, f16 ? 3 : 2, rounding);
                if (f16) ts_extended_f16_to_f32(decoded32, encoded, n);
                else ts_extended_bf16_to_f32(decoded32, encoded, n);
                for (size_t i = 0; i < n; ++i) assert(ts_float_bits(decoded32[i]) == decoded32_expected[i]);
                ts_extended_convert_encoded(cross, encoded, n, f16 ? 2 : 3, f16 ? 3 : 2, rounding);
                for (size_t i = 0; i < n; ++i) {
                    if (cross[i] != cross_expected[i]) {
                        fprintf(stderr, "cross source=%s rounding=%d caller=%zu lane=%zu bits=%04x got=%04x expected=%04x\n",
                                f16 ? "f16" : "bf16", rounding, mode, i, encoded[i], cross[i], cross_expected[i]);
                        abort();
                    }
                }
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
    check_extended();
    puts("Float encoding rounding and environment checks passed.");
    return 0;
}
