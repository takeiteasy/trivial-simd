#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
#include <arm_neon.h>
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

#include "masks.h"
#include "conversion.h"

#define TS_COMPARE(op, a, b) ((op) == 0 ? (a) == (b) : (op) == 1 ? (a) != (b) : \
                              (op) == 2 ? (a) < (b) : (op) == 3 ? (a) <= (b) : \
                              (op) == 4 ? (a) > (b) : (a) >= (b))

#define TS_EXTENDED(suffix, type, unsigned_type, signed_type) \
int ts_extended_unary_##suffix(unsigned op, type *out, const type *input, size_t n) { \
    for (size_t i = 0; i < n; ++i) { \
        type x = input[i]; \
        if (op == 0) out[i] = (type)((unsigned_type)0 - (unsigned_type)x); \
        else if (op == 1) out[i] = signed_type && x < 0 \
            ? (type)((unsigned_type)0 - (unsigned_type)x) : x; \
        else return -1; \
    } \
    return 0; \
} \
int ts_extended_minmax_##suffix(unsigned op, type *out, const type *left, const type *right, \
                                 type left_scalar, type right_scalar, size_t n) { \
    for (size_t i = 0; i < n; ++i) { \
        type a = left ? left[i] : left_scalar, b = right ? right[i] : right_scalar; \
        out[i] = op == 0 ? (b < a ? b : a) : (b > a ? b : a); \
    } \
    return 0; \
} \
int ts_extended_clamp_##suffix(type *out, const type *input, const type *lower, \
                                const type *upper, type lower_scalar, type upper_scalar, size_t n) { \
    for (size_t i = 0; i < n; ++i) { \
        type lo = lower ? lower[i] : lower_scalar, hi = upper ? upper[i] : upper_scalar; \
        if (lo > hi) return -2; \
        type x = input[i]; \
        out[i] = x < lo ? lo : x > hi ? hi : x; \
    } \
    return 0; \
} \
int ts_extended_compare_##suffix(unsigned op, uint8_t *mask, const type *left, \
                                  const type *right, type left_scalar, type right_scalar, size_t n) { \
    size_t i = ts_mask_compare_##suffix(op, mask, left, right, left_scalar, right_scalar, n, 1); \
    for (; i < n; ++i) { \
        type a = left ? left[i] : left_scalar, b = right ? right[i] : right_scalar; \
        mask[i] = (uint8_t)TS_COMPARE(op, a, b); \
    } \
    return 0; \
} \
int ts_extended_select_##suffix(type *out, const uint8_t *mask, const type *on_true, \
                                 const type *on_false, type true_scalar, type false_scalar, size_t n) { \
    size_t i = ts_mask_select_##suffix(out, mask, on_true, on_false, true_scalar, false_scalar, n, 1); \
    for (; i < n; ++i) \
        out[i] = mask[i] ? (on_true ? on_true[i] : true_scalar) \
                         : (on_false ? on_false[i] : false_scalar); \
    return 0; \
}

TS_EXTENDED(s8, int8_t, uint8_t, 1)
TS_EXTENDED(u8, uint8_t, uint8_t, 0)
TS_EXTENDED(s16, int16_t, uint16_t, 1)
TS_EXTENDED(u16, uint16_t, uint16_t, 0)
TS_EXTENDED(s32, int32_t, uint32_t, 1)
TS_EXTENDED(u32, uint32_t, uint32_t, 0)
TS_EXTENDED(s64, int64_t, uint64_t, 1)
TS_EXTENDED(u64, uint64_t, uint64_t, 0)

#define TS_EXTENDED_FLOAT(suffix, type, neg, absolute, square_root) \
int ts_extended_unary_##suffix(unsigned op, type *out, const type *input, size_t n) { \
    for (size_t i = 0; i < n; ++i) \
        if ((op == 2 && input[i] < 0) || (op == 3 && input[i] == 0)) return -2; \
    size_t i = 0; \
    i = ts_extended_unary_packed_##suffix(op, out, input, n); \
    switch (op) { \
    case 0: for (; i < n; ++i) out[i] = -input[i]; break; \
    case 1: for (; i < n; ++i) out[i] = absolute(input[i]); break; \
    case 2: for (; i < n; ++i) out[i] = input[i] == 0 ? input[i] : square_root(input[i]); break; \
    default: for (; i < n; ++i) out[i] = (type)1 / input[i]; break; \
    } \
    return 0; \
} \
int ts_extended_minmax_##suffix(unsigned op, type *out, const type *left, const type *right, \
                                 type left_scalar, type right_scalar, size_t n) { \
    for (size_t i = 0; i < n; ++i) { \
        type a = left ? left[i] : left_scalar, b = right ? right[i] : right_scalar; \
        out[i] = op == 0 ? (b < a ? b : a) : (b > a ? b : a); \
    } \
    return 0; \
} \
int ts_extended_clamp_##suffix(type *out, const type *input, const type *lower, \
                                const type *upper, type lower_scalar, type upper_scalar, size_t n) { \
    for (size_t i = 0; i < n; ++i) { \
        type lo = lower ? lower[i] : lower_scalar, hi = upper ? upper[i] : upper_scalar; \
        if (lo > hi) return -2; \
        type x = input[i]; \
        x = x < lo ? lo : x; \
        out[i] = x > hi ? hi : x; \
    } \
    return 0; \
} \
int ts_extended_compare_##suffix(unsigned op, uint8_t *mask, const type *left, \
                                  const type *right, type left_scalar, type right_scalar, size_t n) { \
    size_t i = ts_mask_compare_##suffix(op, mask, left, right, left_scalar, right_scalar, n, 1); \
    for (; i < n; ++i) { \
        type a = left ? left[i] : left_scalar, b = right ? right[i] : right_scalar; \
        mask[i] = (uint8_t)TS_COMPARE(op, a, b); \
    } \
    return 0; \
} \
int ts_extended_select_##suffix(type *out, const uint8_t *mask, const type *on_true, \
                                 const type *on_false, type true_scalar, type false_scalar, size_t n) { \
    size_t i = ts_mask_select_##suffix(out, mask, on_true, on_false, true_scalar, false_scalar, n, 1); \
    for (; i < n; ++i) \
        out[i] = mask[i] ? (on_true ? on_true[i] : true_scalar) \
                         : (on_false ? on_false[i] : false_scalar); \
    return 0; \
}

#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
static size_t ts_extended_unary_packed_f32(unsigned op, float *out, const float *input, size_t n) {
    size_t i = 0;
    switch (op) {
    case 0:
        for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vnegq_f32(vld1q_f32(input + i)));
        break;
    case 1:
        for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vabsq_f32(vld1q_f32(input + i)));
        break;
    case 2:
        for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vsqrtq_f32(vld1q_f32(input + i)));
        break;
    default:
        for (; i + 4 <= n; i += 4)
            vst1q_f32(out + i, vdivq_f32(vdupq_n_f32(1.0f), vld1q_f32(input + i)));
        break;
    }
    return i;
}
static size_t ts_extended_unary_packed_f64(unsigned op, double *out, const double *input, size_t n) {
    size_t i = 0;
    switch (op) {
    case 0:
        for (; i + 2 <= n; i += 2) vst1q_f64(out + i, vnegq_f64(vld1q_f64(input + i)));
        break;
    case 1:
        for (; i + 2 <= n; i += 2) vst1q_f64(out + i, vabsq_f64(vld1q_f64(input + i)));
        break;
    case 2:
        for (; i + 2 <= n; i += 2) vst1q_f64(out + i, vsqrtq_f64(vld1q_f64(input + i)));
        break;
    default:
        for (; i + 2 <= n; i += 2)
            vst1q_f64(out + i, vdivq_f64(vdupq_n_f64(1.0), vld1q_f64(input + i)));
        break;
    }
    return i;
}
void ts_extended_f32_to_f64(double *out, const float *input, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);
        vst1q_f64(out + i, vcvt_f64_f32(vget_low_f32(x)));
        vst1q_f64(out + i + 2, vcvt_f64_f32(vget_high_f32(x)));
    }
    for (; i < n; ++i) out[i] = input[i];
}
void ts_extended_f64_to_f32(float *out, const double *input, size_t n) {
    size_t i = 0;
    for (; i + 2 <= n; i += 2)
        vst1_f32(out + i, vcvt_f32_f64(vld1q_f64(input + i)));
    for (; i < n; ++i) out[i] = (float)input[i];
}
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
static size_t ts_extended_unary_packed_f32(unsigned op, float *out, const float *input, size_t n) {
    size_t i = 0;
    switch (op) {
    case 0:
        for (; i + 4 <= n; i += 4)
            _mm_storeu_ps(out + i, _mm_xor_ps(_mm_loadu_ps(input + i), _mm_set1_ps(-0.0f)));
        break;
    case 1:
        for (; i + 4 <= n; i += 4)
            _mm_storeu_ps(out + i, _mm_andnot_ps(_mm_set1_ps(-0.0f), _mm_loadu_ps(input + i)));
        break;
    case 2:
        for (; i + 4 <= n; i += 4)
            _mm_storeu_ps(out + i, _mm_sqrt_ps(_mm_loadu_ps(input + i)));
        break;
    default:
        for (; i + 4 <= n; i += 4)
            _mm_storeu_ps(out + i, _mm_div_ps(_mm_set1_ps(1.0f), _mm_loadu_ps(input + i)));
        break;
    }
    return i;
}
static size_t ts_extended_unary_packed_f64(unsigned op, double *out, const double *input, size_t n) {
    size_t i = 0;
    switch (op) {
    case 0:
        for (; i + 2 <= n; i += 2)
            _mm_storeu_pd(out + i, _mm_xor_pd(_mm_loadu_pd(input + i), _mm_set1_pd(-0.0)));
        break;
    case 1:
        for (; i + 2 <= n; i += 2)
            _mm_storeu_pd(out + i, _mm_andnot_pd(_mm_set1_pd(-0.0), _mm_loadu_pd(input + i)));
        break;
    case 2:
        for (; i + 2 <= n; i += 2)
            _mm_storeu_pd(out + i, _mm_sqrt_pd(_mm_loadu_pd(input + i)));
        break;
    default:
        for (; i + 2 <= n; i += 2)
            _mm_storeu_pd(out + i, _mm_div_pd(_mm_set1_pd(1.0), _mm_loadu_pd(input + i)));
        break;
    }
    return i;
}
void ts_extended_f32_to_f64(double *out, const float *input, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 x = _mm_loadu_ps(input + i);
        _mm_storeu_pd(out + i, _mm_cvtps_pd(x));
        _mm_storeu_pd(out + i + 2, _mm_cvtps_pd(_mm_movehl_ps(x, x)));
    }
    for (; i < n; ++i) out[i] = input[i];
}
void ts_extended_f64_to_f32(float *out, const double *input, size_t n) {
    size_t i = 0;
    for (; i + 2 <= n; i += 2)
        _mm_storel_pi((__m64 *)(out + i), _mm_cvtpd_ps(_mm_loadu_pd(input + i)));
    for (; i < n; ++i) out[i] = (float)input[i];
}
#else
static size_t ts_extended_unary_packed_f32(unsigned op, float *out, const float *input, size_t n) {
    (void)op; (void)out; (void)input; (void)n; return 0;
}
static size_t ts_extended_unary_packed_f64(unsigned op, double *out, const double *input, size_t n) {
    (void)op; (void)out; (void)input; (void)n; return 0;
}
void ts_extended_f32_to_f64(double *out, const float *input, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = input[i];
}
void ts_extended_f64_to_f32(float *out, const double *input, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = (float)input[i];
}
#endif

/* Integer narrowing preserves the caller's floating-point environment.
   NaNs keep their sign and leading payload bits and become quiet. */
enum { TS_ROUND_NEAREST_EVEN, TS_ROUND_TRUNCATE, TS_ROUND_FLOOR, TS_ROUND_CEILING };

unsigned ts_extended_float_encoding_rounding_modes(void) { return 0xfu; }

static int ts_round_away(uint32_t bits, int rounding) {
    return rounding == TS_ROUND_FLOOR ? (int)(bits >> 31) :
           rounding == TS_ROUND_CEILING && !(bits >> 31);
}

static uint32_t ts_round_bias(uint32_t bits, int rounding, unsigned shift) {
    if (rounding == TS_ROUND_NEAREST_EVEN)
        return (1u << (shift - 1)) - 1u + ((bits >> shift) & 1u);
    return ts_round_away(bits, rounding) ? (1u << shift) - 1u : 0u;
}

static uint32_t ts_float_bits(float x) { uint32_t b; memcpy(&b, &x, sizeof b); return b; }
static float ts_bits_float(uint32_t b) { float x; memcpy(&x, &b, sizeof x); return x; }

static float ts_bf16_to_f32_scalar(uint16_t h) {
    uint32_t b = (uint32_t)h << 16;
    return ts_bits_float((b & 0x7fffffffu) > 0x7f800000u ? b | 0x400000u : b);
}

static uint16_t ts_f32_to_bf16_scalar(float x, int rounding) {
    uint32_t b = ts_float_bits(x);
    if ((b & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((b >> 16) | 0x40u);
    return (uint16_t)((b + ts_round_bias(b, rounding, 16)) >> 16);
}

static float ts_f16_to_f32_scalar(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, mantissa = h & 0x3ffu;
    int exponent = (h >> 10) & 0x1f;
    if (exponent == 0x1f)
        return ts_bits_float(sign | 0x7f800000u | (mantissa << 13) | (mantissa ? 0x400000u : 0));
    if (exponent == 0) {
        if (mantissa == 0) return ts_bits_float(sign);
        for (exponent = 1; !(mantissa & 0x400u); --exponent) mantissa <<= 1;
        mantissa &= 0x3ffu;
    }
    return ts_bits_float(sign | ((uint32_t)(exponent + 112) << 23) | (mantissa << 13));
}

static uint16_t ts_f32_to_f16_scalar(float x, int rounding) {
    uint32_t b = ts_float_bits(x), sign = (b >> 16) & 0x8000u, a = b & 0x7fffffffu;
    if (a > 0x7f800000u) return (uint16_t)(sign | 0x7e00u | ((a >> 13) & 0x3ffu));
    if (a >= 0x477ff000u) {                                    /* 65520 and above */
        int toward = rounding != TS_ROUND_NEAREST_EVEN && !ts_round_away(b, rounding);
        return (uint16_t)(sign | (toward && a < 0x7f800000u ? 0x7bffu : 0x7c00u));
    }
    if (a >= 0x38800000u) {                                     /* f16 normal range */
        a -= 112u << 23;
        return (uint16_t)(sign | ((a + ts_round_bias(b, rounding, 13)) >> 13));
    }
    int exponent = (int)(a >> 23);
    if (exponent < 102) return (uint16_t)(sign | (a != 0 && ts_round_away(b, rounding)));
    uint32_t mantissa = (a & 0x7fffffu) | 0x800000u, shift = (uint32_t)(126 - exponent);
    uint32_t quotient = mantissa >> shift, remainder = mantissa & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1u);
    quotient += rounding == TS_ROUND_NEAREST_EVEN
        ? remainder > half || (remainder == half && (quotient & 1u))
        : remainder != 0 && ts_round_away(b, rounding);
    return (uint16_t)(sign | quotient);
}

#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
static float32x4_t ts_bf16_to_f32_neon(uint16x4_t h) {
    uint32x4_t b = vshll_n_u16(h, 16);
    uint32x4_t nan = vcgtq_u32(vandq_u32(b, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    return vreinterpretq_f32_u32(vorrq_u32(b, vandq_u32(nan, vdupq_n_u32(0x400000))));
}
void ts_extended_bf16_to_f32(float *out, const uint16_t *input, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t h = vld1q_u16(input + i);
        vst1q_f32(out + i, ts_bf16_to_f32_neon(vget_low_u16(h)));
        vst1q_f32(out + i + 4, ts_bf16_to_f32_neon(vget_high_u16(h)));
    }
    for (; i < n; ++i) out[i] = ts_bf16_to_f32_scalar(input[i]);
}
static uint16x4_t ts_f32_to_bf16_neon(float32x4_t x, int rounding) {
    uint32x4_t b = vreinterpretq_u32_f32(x);
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(b, 16), vdupq_n_u32(1));
    uint32x4_t negative = vceqq_u32(vshrq_n_u32(b, 31), vdupq_n_u32(1));
    uint32x4_t away = rounding == TS_ROUND_FLOOR ? negative :
                     rounding == TS_ROUND_CEILING ? vmvnq_u32(negative) : vdupq_n_u32(0);
    uint32x4_t bias = rounding == TS_ROUND_NEAREST_EVEN
        ? vaddq_u32(vdupq_n_u32(0x7fff), lsb)
        : vandq_u32(away, vdupq_n_u32(0xffff));
    uint32x4_t rounded = vaddq_u32(b, bias);
    uint32x4_t nan = vcgtq_u32(vandq_u32(b, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    return vshrn_n_u32(vbslq_u32(nan, vorrq_u32(b, vdupq_n_u32(0x400000)), rounded), 16);
}
void ts_extended_f32_to_bf16(uint16_t *out, const float *input, size_t n, int rounding) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        vst1q_u16(out + i, vcombine_u16(ts_f32_to_bf16_neon(vld1q_f32(input + i), rounding),
                                        ts_f32_to_bf16_neon(vld1q_f32(input + i + 4), rounding)));
    for (; i < n; ++i) out[i] = ts_f32_to_bf16_scalar(input[i], rounding);
}
#if defined(__aarch64__)
/* Conversion instructions set FPSR flags; isolate them from caller traps. */
static uint64_t ts_round_fpcr(int rounding) {
    static const uint64_t modes[] = {0, 3ull << 22, 2ull << 22, 1ull << 22};
    return modes[rounding];
}
typedef struct { uint64_t fpcr, fpsr; } ts_fp_env;
static ts_fp_env ts_fp_enter(uint64_t fpcr) {
    ts_fp_env saved;
    __asm__ volatile("mrs %0, fpcr" : "=r"(saved.fpcr) :: "memory");
    __asm__ volatile("mrs %0, fpsr" : "=r"(saved.fpsr) :: "memory");
    __asm__ volatile("msr fpcr, %0" :: "r"(fpcr) : "memory");
    return saved;
}
static void ts_fp_leave(ts_fp_env saved) {
    __asm__ volatile("msr fpsr, %0" :: "r"(saved.fpsr) : "memory");
    __asm__ volatile("msr fpcr, %0" :: "r"(saved.fpcr) : "memory");
}
void ts_extended_f16_to_f32(float *out, const uint16_t *input, size_t n) {
    size_t i = 0;
    ts_fp_env saved = ts_fp_enter(0);
    for (; i + 4 <= n; i += 4)
        vst1q_f32(out + i, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(input + i))));
    ts_fp_leave(saved);
    for (; i < n; ++i) out[i] = ts_f16_to_f32_scalar(input[i]);
}
void ts_extended_f32_to_f16(uint16_t *out, const float *input, size_t n, int rounding) {
    size_t i = 0;
    ts_fp_env saved = ts_fp_enter(ts_round_fpcr(rounding));
    for (; i + 4 <= n; i += 4)
        vst1_u16(out + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(input + i))));
    ts_fp_leave(saved);
    for (; i < n; ++i) out[i] = ts_f32_to_f16_scalar(input[i], rounding);
}
#else
void ts_extended_f16_to_f32(float *out, const uint16_t *input, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = ts_f16_to_f32_scalar(input[i]);
}
void ts_extended_f32_to_f16(uint16_t *out, const float *input, size_t n, int rounding) {
    for (size_t i = 0; i < n; ++i) out[i] = ts_f32_to_f16_scalar(input[i], rounding);
}
#endif
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
/* B holds one bf16 in the high half of each lane. */
static __m128 ts_bf16_to_f32_sse2(__m128i b) {
    __m128i nan = _mm_cmpgt_epi32(_mm_and_si128(b, _mm_set1_epi32(0x7fffffff)),
                                  _mm_set1_epi32(0x7f800000));
    return _mm_castsi128_ps(_mm_or_si128(b, _mm_and_si128(nan, _mm_set1_epi32(0x400000))));
}
void ts_extended_bf16_to_f32(float *out, const uint16_t *input, size_t n) {
    size_t i = 0;
    const __m128i zero = _mm_setzero_si128();
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(input + i));
        _mm_storeu_ps(out + i, ts_bf16_to_f32_sse2(_mm_unpacklo_epi16(zero, h)));
        _mm_storeu_ps(out + i + 4, ts_bf16_to_f32_sse2(_mm_unpackhi_epi16(zero, h)));
    }
    for (; i < n; ++i) out[i] = ts_bf16_to_f32_scalar(input[i]);
}
static __m128i ts_round_away_sse2(__m128i bits, int rounding) {
    __m128i negative = _mm_srai_epi32(bits, 31);
    return rounding == TS_ROUND_FLOOR ? negative :
           rounding == TS_ROUND_CEILING ? _mm_xor_si128(negative, _mm_set1_epi32(-1)) :
           _mm_setzero_si128();
}

static __m128i ts_round_bias_sse2(__m128i bits, int rounding, unsigned shift) {
    if (rounding == TS_ROUND_NEAREST_EVEN)
        return _mm_add_epi32(_mm_set1_epi32((1u << (shift - 1)) - 1u),
                            _mm_and_si128(_mm_srl_epi32(bits, _mm_cvtsi32_si128(shift)),
                                          _mm_set1_epi32(1)));
    return _mm_and_si128(ts_round_away_sse2(bits, rounding), _mm_set1_epi32((1u << shift) - 1u));
}

/* Sign-extend each result for _mm_packs_epi32. */
static __m128i ts_f32_to_bf16_sse2(__m128 x, int rounding) {
    __m128i b = _mm_castps_si128(x);
    __m128i bias = ts_round_bias_sse2(b, rounding, 16);
    __m128i rounded = _mm_add_epi32(b, bias);
    __m128i nan = _mm_cmpgt_epi32(_mm_and_si128(b, _mm_set1_epi32(0x7fffffff)),
                                  _mm_set1_epi32(0x7f800000));
    __m128i quiet = _mm_or_si128(b, _mm_set1_epi32(0x400000));
    return _mm_srai_epi32(_mm_or_si128(_mm_and_si128(nan, quiet),
                                       _mm_andnot_si128(nan, rounded)), 16);
}
void ts_extended_f32_to_bf16(uint16_t *out, const float *input, size_t n, int rounding) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i *)(out + i),
                         _mm_packs_epi32(ts_f32_to_bf16_sse2(_mm_loadu_ps(input + i), rounding),
                                         ts_f32_to_bf16_sse2(_mm_loadu_ps(input + i + 4), rounding)));
    for (; i < n; ++i) out[i] = ts_f32_to_bf16_scalar(input[i], rounding);
}
/* H holds one zero-extended f16 per lane. Subnormals become normal floats
   through an exact subtraction of 2^-14, so DAZ and FTZ cannot affect them. */
static __m128 ts_f16_to_f32_sse2(__m128i h) {
    const __m128i exponent_mask = _mm_set1_epi32(0x7c00 << 13);
    const __m128i magic = _mm_set1_epi32(113 << 23);
    __m128i magnitude = _mm_slli_epi32(_mm_and_si128(h, _mm_set1_epi32(0x7fff)), 13);
    __m128i exponent = _mm_and_si128(magnitude, exponent_mask);
    __m128i special = _mm_cmpeq_epi32(exponent, exponent_mask);
    __m128i nan = _mm_cmpgt_epi32(magnitude, exponent_mask);
    __m128i subnormal = _mm_cmpeq_epi32(exponent, _mm_setzero_si128());
    __m128i bits = _mm_add_epi32(magnitude, _mm_set1_epi32(112 << 23));
    bits = _mm_add_epi32(bits, _mm_and_si128(special, _mm_set1_epi32(112 << 23)));
    bits = _mm_or_si128(bits, _mm_and_si128(nan, _mm_set1_epi32(0x400000)));
    __m128i scaled = _mm_or_si128(_mm_and_si128(subnormal, _mm_add_epi32(magnitude, magic)),
                                  _mm_andnot_si128(subnormal, magic));
    __m128i small = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(scaled),
                                                _mm_castsi128_ps(magic)));
    /* Equal operands subtract to -0 under downward rounding. */
    small = _mm_and_si128(small, _mm_set1_epi32(0x7fffffff));
    bits = _mm_or_si128(_mm_and_si128(subnormal, small), _mm_andnot_si128(subnormal, bits));
    __m128i sign = _mm_slli_epi32(_mm_and_si128(h, _mm_set1_epi32(0x8000)), 16);
    return _mm_castsi128_ps(_mm_or_si128(bits, sign));
}
void ts_extended_f16_to_f32(float *out, const uint16_t *input, size_t n) {
    size_t i = 0;
    const __m128i zero = _mm_setzero_si128();
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(input + i));
        _mm_storeu_ps(out + i, ts_f16_to_f32_sse2(_mm_unpacklo_epi16(h, zero)));
        _mm_storeu_ps(out + i + 4, ts_f16_to_f32_sse2(_mm_unpackhi_epi16(h, zero)));
    }
    for (; i < n; ++i) out[i] = ts_f16_to_f32_scalar(input[i]);
}
/* Lanes that round to an f16 subnormal need a per-lane shift, which SSE2 lacks;
   *SLOW reports them so the caller converts that block with the scalar helper. */
static __m128i ts_f32_to_f16_sse2(__m128 x, int rounding, int *slow) {
    __m128i b = _mm_castps_si128(x);
    __m128i a = _mm_and_si128(b, _mm_set1_epi32(0x7fffffff));
    __m128i normal = _mm_sub_epi32(a, _mm_set1_epi32(112 << 23));
    __m128i bias = ts_round_bias_sse2(b, rounding, 13);
    __m128i r = _mm_srli_epi32(_mm_add_epi32(normal, bias), 13);
    __m128i overflow = _mm_cmpgt_epi32(a, _mm_set1_epi32(0x477fefff));
    __m128i finite = _mm_cmplt_epi32(a, _mm_set1_epi32(0x7f800000));
    __m128i toward = rounding == TS_ROUND_NEAREST_EVEN ? _mm_setzero_si128() :
                     _mm_andnot_si128(ts_round_away_sse2(b, rounding), finite);
    __m128i limit = _mm_sub_epi32(_mm_set1_epi32(0x7c00),
                                  _mm_and_si128(toward, _mm_set1_epi32(1)));
    r = _mm_or_si128(_mm_and_si128(overflow, limit), _mm_andnot_si128(overflow, r));
    __m128i nan = _mm_cmpgt_epi32(a, _mm_set1_epi32(0x7f800000));
    __m128i payload = _mm_or_si128(_mm_set1_epi32(0x7e00),
                                   _mm_and_si128(_mm_srli_epi32(a, 13), _mm_set1_epi32(0x3ff)));
    r = _mm_or_si128(_mm_and_si128(nan, payload), _mm_andnot_si128(nan, r));
    __m128i small = _mm_cmplt_epi32(a, _mm_set1_epi32(0x38800000));
    __m128i zero = rounding >= TS_ROUND_FLOOR
        ? _mm_cmpeq_epi32(a, _mm_setzero_si128())
        : _mm_cmplt_epi32(a, _mm_set1_epi32(0x33000001));
    *slow |= _mm_movemask_epi8(_mm_andnot_si128(zero, small));
    r = _mm_andnot_si128(small, r);
    r = _mm_or_si128(r, _mm_and_si128(_mm_srli_epi32(b, 16), _mm_set1_epi32(0x8000)));
    return _mm_srai_epi32(_mm_slli_epi32(r, 16), 16);
}
void ts_extended_f32_to_f16(uint16_t *out, const float *input, size_t n, int rounding) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int slow = 0;
        __m128i low = ts_f32_to_f16_sse2(_mm_loadu_ps(input + i), rounding, &slow);
        __m128i high = ts_f32_to_f16_sse2(_mm_loadu_ps(input + i + 4), rounding, &slow);
        if (slow)
            for (size_t j = i; j < i + 8; ++j) out[j] = ts_f32_to_f16_scalar(input[j], rounding);
        else
            _mm_storeu_si128((__m128i *)(out + i), _mm_packs_epi32(low, high));
    }
    for (; i < n; ++i) out[i] = ts_f32_to_f16_scalar(input[i], rounding);
}
#else
void ts_extended_bf16_to_f32(float *out, const uint16_t *input, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = ts_bf16_to_f32_scalar(input[i]);
}
void ts_extended_f32_to_bf16(uint16_t *out, const float *input, size_t n, int rounding) {
    for (size_t i = 0; i < n; ++i) out[i] = ts_f32_to_bf16_scalar(input[i], rounding);
}
void ts_extended_f16_to_f32(float *out, const uint16_t *input, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = ts_f16_to_f32_scalar(input[i]);
}
void ts_extended_f32_to_f16(uint16_t *out, const float *input, size_t n, int rounding) {
    for (size_t i = 0; i < n; ++i) out[i] = ts_f32_to_f16_scalar(input[i], rounding);
}
#endif

TS_EXTENDED_FLOAT(f32, float, -, fabsf, sqrtf)
TS_EXTENDED_FLOAT(f64, double, -, fabs, sqrt)

int ts_extended_mask_reduce(unsigned op, const uint8_t *mask, size_t n, size_t *result) {
    size_t total = ts_mask_count(mask, n);
    *result = op == 0 ? total : op == 1 ? total != 0 : total == n;
    return 0;
}

unsigned ts_extended_encoded_conversion_rounding_modes(void) { return 0xfu; }

static uint16_t ts_f64_bits_to_encoded(uint64_t bits, int f16, int rounding) {
    unsigned precision = f16 ? 10 : 7, bias = f16 ? 15 : 127;
    unsigned infinity = (f16 ? 31u : 255u) << precision;
    unsigned sign = (unsigned)(bits >> 48) & 0x8000u;
    unsigned exponent = (unsigned)(bits >> 52) & 0x7ffu;
    uint64_t fraction = bits & UINT64_C(0xfffffffffffff);
    int away = (rounding == TS_ROUND_FLOOR && sign) ||
               (rounding == TS_ROUND_CEILING && !sign);
    if (exponent == 0x7ffu)
        return (uint16_t)(sign | infinity | (fraction ?
            (1u << (precision - 1)) | (unsigned)(fraction >> (52 - precision)) : 0));
    int power = exponent ? (int)exponent - 1023 : -1022;
    uint64_t mantissa = fraction | (exponent ? UINT64_C(1) << 52 : 0);
    if (power > (int)bias)
        return (uint16_t)(sign | (rounding == TS_ROUND_NEAREST_EVEN || away ? infinity : infinity - 1));
    unsigned shift = 52 - precision;
    unsigned base = 0;
    if (power >= 1 - (int)bias) base = (unsigned)(power + (int)bias - 1) << precision;
    else shift += (unsigned)(1 - (int)bias - power);
    uint64_t quotient = shift < 64 ? mantissa >> shift : 0;
    uint64_t remainder = shift < 64 ? mantissa & ((UINT64_C(1) << shift) - 1) : mantissa;
    if (rounding == TS_ROUND_NEAREST_EVEN && shift < 64) {
        uint64_t half = UINT64_C(1) << (shift - 1);
        quotient += remainder > half || (remainder == half && (quotient & 1));
    } else if (rounding != TS_ROUND_NEAREST_EVEN) quotient += away && remainder != 0;
    unsigned result = base + (unsigned)quotient;
    if (result >= infinity && rounding != TS_ROUND_NEAREST_EVEN && !away) result = infinity - 1;
    return (uint16_t)(sign | result);
}

static uint64_t ts_f32_bits_to_f64_bits(uint32_t bits) {
    uint64_t sign = (uint64_t)(bits & 0x80000000u) << 32;
    unsigned exponent = (bits >> 23) & 0xffu;
    uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 0xffu) return sign | UINT64_C(0x7ff0000000000000) | ((uint64_t)fraction << 29);
    if (!exponent) {
        if (!fraction) return sign;
        int power = -126;
        while (!(fraction & 0x800000u)) { fraction <<= 1; --power; }
        return sign | ((uint64_t)(power + 1023) << 52) | ((uint64_t)(fraction & 0x7fffffu) << 29);
    }
    return sign | ((uint64_t)(exponent + 896) << 52) | ((uint64_t)fraction << 29);
}

/* Types: 0=f32, 1=f64, 2=bf16, 3=f16. */
/* TODO: exceptional f64 lanes remain scalar; add variable-shift bit conversion (#156). */
static size_t ts_f64_to_encoded_packed(uint16_t *out, const double *input, size_t n, int f16, int rounding) {
    size_t i = 0;
#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64) || defined(__x86_64__) || defined(_M_X64))
    unsigned precision = f16 ? 10 : 7, bias = f16 ? 15 : 127, shift = 52 - precision;
    uint64_t adjustment = (uint64_t)(1023 - bias) << precision;
    for (; n - i >= 2; i += 2) {
        uint64_t bits[2], result[2]; memcpy(bits, input + i, sizeof bits);
#if defined(__aarch64__) || defined(_M_ARM64)
        uint64x2_t v = vld1q_u64(bits), magnitude = vandq_u64(v, vdupq_n_u64(UINT64_C(0x7fffffffffffffff)));
        uint64x2_t quotient = vshlq_u64(magnitude, vdupq_n_s64(-(int64_t)shift));
        uint64x2_t increment = vdupq_n_u64(0);
        if (rounding == TS_ROUND_NEAREST_EVEN)
            increment = vaddq_u64(vdupq_n_u64((UINT64_C(1) << (shift - 1)) - 1), vandq_u64(quotient, vdupq_n_u64(1)));
        else if (rounding == TS_ROUND_FLOOR || rounding == TS_ROUND_CEILING) {
            uint64x2_t sign = vshrq_n_u64(v, 63);
            if (rounding == TS_ROUND_CEILING) sign = veorq_u64(sign, vdupq_n_u64(1));
            increment = vandq_u64(vsubq_u64(vdupq_n_u64(0), sign), vdupq_n_u64((UINT64_C(1) << shift) - 1));
        }
        quotient = vsubq_u64(vshlq_u64(vaddq_u64(magnitude, increment), vdupq_n_s64(-(int64_t)shift)), vdupq_n_u64(adjustment));
        vst1q_u64(result, vorrq_u64(quotient, vandq_u64(vshrq_n_u64(v, 48), vdupq_n_u64(0x8000))));
#else
        __m128i v = _mm_loadu_si128((const __m128i *)bits);
        __m128i magnitude = _mm_and_si128(v, _mm_set1_epi64x(INT64_MAX));
        __m128i count = _mm_cvtsi32_si128((int)shift), quotient = _mm_srl_epi64(magnitude, count);
        __m128i increment = _mm_setzero_si128();
        if (rounding == TS_ROUND_NEAREST_EVEN)
            increment = _mm_add_epi64(_mm_set1_epi64x((INT64_C(1) << (shift - 1)) - 1), _mm_and_si128(quotient, _mm_set1_epi64x(1)));
        else if (rounding == TS_ROUND_FLOOR || rounding == TS_ROUND_CEILING) {
            __m128i sign = _mm_srli_epi64(v, 63);
            if (rounding == TS_ROUND_CEILING) sign = _mm_xor_si128(sign, _mm_set1_epi64x(1));
            increment = _mm_and_si128(_mm_sub_epi64(_mm_setzero_si128(), sign), _mm_set1_epi64x((INT64_C(1) << shift) - 1));
        }
        quotient = _mm_sub_epi64(_mm_srl_epi64(_mm_add_epi64(magnitude, increment), count), _mm_set1_epi64x((int64_t)adjustment));
        _mm_storeu_si128((__m128i *)result, _mm_or_si128(quotient, _mm_and_si128(_mm_srli_epi64(v, 48), _mm_set1_epi64x(0x8000))));
#endif
        for (size_t j = 0; j < 2; ++j) {
            unsigned exponent = (unsigned)(bits[j] >> 52) & 0x7ffu;
            out[i + j] = exponent >= 1024 - bias && exponent <= 1023 + bias
                ? (uint16_t)result[j] : ts_f64_bits_to_encoded(bits[j], f16, rounding);
        }
    }
#endif
    return i;
}

static size_t ts_encoded_to_f64_packed(double *out, const uint16_t *input, size_t n, int f16) {
    size_t i = 0;
#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64) || defined(__x86_64__) || defined(_M_X64))
    unsigned precision = f16 ? 10 : 7, maximum = f16 ? 31 : 255;
    uint64_t fraction_mask = (UINT64_C(1) << precision) - 1, adjustment = f16 ? 1008 : 896;
    for (; n - i >= 4; i += 4) {
        uint64_t result[4];
#if defined(__aarch64__) || defined(_M_ARM64)
        uint32x4_t v = vmovl_u16(vld1_u16(input + i));
        for (size_t j = 0; j < 2; ++j) {
            uint64x2_t wide = vmovl_u32(j ? vget_high_u32(v) : vget_low_u32(v));
            uint64x2_t fraction = vshlq_u64(vandq_u64(wide, vdupq_n_u64(fraction_mask)), vdupq_n_s64(52 - precision));
            uint64x2_t exponent = vaddq_u64(vandq_u64(vshlq_u64(wide, vdupq_n_s64(-(int64_t)precision)),
                                                                   vdupq_n_u64(maximum)), vdupq_n_u64(adjustment));
            vst1q_u64(result + j * 2, vorrq_u64(vorrq_u64(fraction, vshlq_n_u64(exponent, 52)),
                                             vshlq_n_u64(vandq_u64(wide, vdupq_n_u64(0x8000)), 48)));
        }
#else
        __m128i v = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)(input + i)), _mm_setzero_si128());
        __m128i right = _mm_cvtsi32_si128((int)precision), left = _mm_cvtsi32_si128((int)(52 - precision));
        for (size_t j = 0; j < 2; ++j) {
            __m128i wide = j ? _mm_unpackhi_epi32(v, _mm_setzero_si128()) : _mm_unpacklo_epi32(v, _mm_setzero_si128());
            __m128i fraction = _mm_sll_epi64(_mm_and_si128(wide, _mm_set1_epi64x((int64_t)fraction_mask)), left);
            __m128i exponent = _mm_add_epi64(_mm_and_si128(_mm_srl_epi64(wide, right), _mm_set1_epi64x(maximum)),
                                           _mm_set1_epi64x((int64_t)adjustment));
            _mm_storeu_si128((__m128i *)(result + j * 2), _mm_or_si128(_mm_or_si128(fraction, _mm_slli_epi64(exponent, 52)),
                              _mm_slli_epi64(_mm_and_si128(wide, _mm_set1_epi64x(0x8000)), 48)));
        }
#endif
        for (size_t j = 0; j < 4; ++j) {
            unsigned exponent = (input[i + j] >> precision) & maximum;
            if (!exponent || exponent == maximum) {
                float value = f16 ? ts_f16_to_f32_scalar(input[i + j]) : ts_bf16_to_f32_scalar(input[i + j]);
                result[j] = ts_f32_bits_to_f64_bits(ts_float_bits(value));
            }
        }
        memcpy(out + i, result, sizeof result);
    }
#endif
    return i;
}

void ts_extended_convert_encoded(void *out, const void *input, size_t n,
                                unsigned destination_type, unsigned input_type, int rounding) {
    if (destination_type == input_type) {
        size_t size = input_type == 0 ? sizeof(float) : input_type == 1 ? sizeof(double) : sizeof(uint16_t);
        memmove(out, input, n * size);
    } else if (input_type == 0) {
        if (destination_type == 2) ts_extended_f32_to_bf16(out, input, n, rounding);
        else ts_extended_f32_to_f16(out, input, n, rounding);
    } else if (destination_type == 0) {
        if (input_type == 2) ts_extended_bf16_to_f32(out, input, n);
        else ts_extended_f16_to_f32(out, input, n);
    } else if (input_type >= 2 && destination_type >= 2) {
        float scratch[256];
        for (size_t i = 0; i < n; i += 256) {
            size_t count = n - i < 256 ? n - i : 256;
            if (input_type == 2) ts_extended_bf16_to_f32(scratch, (const uint16_t *)input + i, count);
            else ts_extended_f16_to_f32(scratch, (const uint16_t *)input + i, count);
            if (destination_type == 2) ts_extended_f32_to_bf16((uint16_t *)out + i, scratch, count, rounding);
            else ts_extended_f32_to_f16((uint16_t *)out + i, scratch, count, rounding);
        }
    } else if (input_type == 1) {
        size_t i = ts_f64_to_encoded_packed(out, input, n, destination_type == 3, rounding);
        for (; i < n; ++i) {
            uint64_t bits;
            memcpy(&bits, (const double *)input + i, sizeof bits);
            ((uint16_t *)out)[i] = ts_f64_bits_to_encoded(bits, destination_type == 3, rounding);
        }
    } else {
        size_t i = ts_encoded_to_f64_packed(out, input, n, input_type == 3);
        for (; i < n; ++i) {
            uint16_t half = ((const uint16_t *)input)[i];
            float value = input_type == 2 ? ts_bf16_to_f32_scalar(half) : ts_f16_to_f32_scalar(half);
            uint64_t bits = ts_f32_bits_to_f64_bits(ts_float_bits(value));
            memcpy((double *)out + i, &bits, sizeof bits);
        }
    }
}

void ts_nd_convert(void *out, const void *input, size_t n, int64_t ds, int64_t is,
                   unsigned destination_type, unsigned input_type, int rounding) {
    if (ds == 1 && is == 1 && destination_type != input_type) {
        if (destination_type == 0 && input_type == 1) {
            ts_extended_f64_to_f32(out, input, n); return;
        }
        if (destination_type == 1 && input_type == 0) {
            ts_extended_f32_to_f64(out, input, n); return;
        }
        if (input_type == 0 || destination_type == 0) {
            ts_extended_convert_encoded(out, input, n, destination_type, input_type, rounding); return;
        }
    }
    const size_t sizes[] = {4, 8, 2, 2};
    for (size_t i = 0; i < n; ++i) {
        const void *source = (const char *)input + (int64_t)i * is * (int64_t)sizes[input_type];
        void *target = (char *)out + (int64_t)i * ds * (int64_t)sizes[destination_type];
        if (destination_type == input_type) {
            memmove(target, source, sizes[input_type]); continue;
        }
        if (input_type == 1) {
            if (destination_type == 0) *(float *)target = (float)*(const double *)source;
            else {
                uint64_t bits; memcpy(&bits, source, sizeof bits);
                *(uint16_t *)target = ts_f64_bits_to_encoded(bits, destination_type == 3, rounding);
            }
        } else {
            float value = input_type == 0 ? *(const float *)source : input_type == 2
                ? ts_bf16_to_f32_scalar(*(const uint16_t *)source)
                : ts_f16_to_f32_scalar(*(const uint16_t *)source);
            if (destination_type == 0) *(float *)target = value;
            else if (destination_type == 1) {
                if (input_type == 0) *(double *)target = (double)value;
                else {
                    uint64_t bits = ts_f32_bits_to_f64_bits(ts_float_bits(value));
                    memcpy(target, &bits, sizeof bits);
                }
            } else *(uint16_t *)target = destination_type == 2
                ? ts_f32_to_bf16_scalar(value, rounding) : ts_f32_to_f16_scalar(value, rounding);
        }
    }
}

int ts_extended_conversion_supported(unsigned destination, unsigned input) {
    return (destination < 2 && input >= 2 && input < 10) ||
           ((destination == 2 || destination == 3) && (input == 4 || input == 5)) ||
           ((destination == 4 || destination == 5) && (input == 6 || input == 7));
}
#define TS_CONVERT_SOURCE(suffix, type, name, source, packed) \
    case name: { \
        const source *values = input; type *out = output; \
        size_t i = packed; \
        for (; i < n; ++i) out[i] = (type)values[i]; \
        return 0; \
    }
#define TS_CONVERT_NARROW(suffix, type) \
    TS_CONVERT_SOURCE(suffix, type, 2, int8_t, TS_LOAD_PACKED(suffix, s8, out, values, n)) \
    TS_CONVERT_SOURCE(suffix, type, 3, uint8_t, TS_LOAD_PACKED(suffix, u8, out, values, n)) \
    TS_CONVERT_SOURCE(suffix, type, 4, int16_t, TS_LOAD_PACKED(suffix, s16, out, values, n)) \
    TS_CONVERT_SOURCE(suffix, type, 5, uint16_t, TS_LOAD_PACKED(suffix, u16, out, values, n)) \
    TS_CONVERT_SOURCE(suffix, type, 6, int32_t, TS_LOAD_PACKED(suffix, s32, out, values, n)) \
    TS_CONVERT_SOURCE(suffix, type, 7, uint32_t, TS_LOAD_PACKED(suffix, u32, out, values, n))
int ts_extended_convert_numeric(void *output, const void *input, size_t n,
                                unsigned destination, unsigned source) {
    switch (destination * 10 + source) {
#define TS_NARROW_CASE(code, from, to, source_type, target_type, low, high) \
    case code: { \
        const source_type *values = input; target_type *out = output; \
        size_t i = ts_narrow_##from##_##to(out, values, n); \
        for (; i < n; ++i) { \
            int64_t value = values[i]; \
            out[i] = (target_type)(value < low ? low : value > high ? high : value); \
        } \
        return 0; \
    }
    TS_NARROW_CASE(24, s16, s8, int16_t, int8_t, INT8_MIN, INT8_MAX)
    TS_NARROW_CASE(25, u16, s8, uint16_t, int8_t, INT8_MIN, INT8_MAX)
    TS_NARROW_CASE(34, s16, u8, int16_t, uint8_t, 0, UINT8_MAX)
    TS_NARROW_CASE(35, u16, u8, uint16_t, uint8_t, 0, UINT8_MAX)
    TS_NARROW_CASE(46, s32, s16, int32_t, int16_t, INT16_MIN, INT16_MAX)
    TS_NARROW_CASE(47, u32, s16, uint32_t, int16_t, INT16_MIN, INT16_MAX)
    TS_NARROW_CASE(56, s32, u16, int32_t, uint16_t, 0, UINT16_MAX)
    TS_NARROW_CASE(57, u32, u16, uint32_t, uint16_t, 0, UINT16_MAX)
#undef TS_NARROW_CASE
    }
    if (destination == 0) {
        switch (source) {
        TS_CONVERT_NARROW(f32, float)
        TS_CONVERT_SOURCE(f32, float, 8, volatile int64_t, 0)
        TS_CONVERT_SOURCE(f32, float, 9, volatile uint64_t, 0)
        }
    } else if (destination == 1) {
        switch (source) {
        TS_CONVERT_NARROW(f64, double)
        TS_CONVERT_SOURCE(f64, double, 8, int64_t, TS_LOAD_PACKED(f64, s64, out, values, n))
        TS_CONVERT_SOURCE(f64, double, 9, uint64_t, TS_LOAD_PACKED(f64, u64, out, values, n))
        }
    }
    return -1;
}
