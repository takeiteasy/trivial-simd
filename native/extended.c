#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

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
    for (size_t i = 0; i < n; ++i) { \
        type a = left ? left[i] : left_scalar, b = right ? right[i] : right_scalar; \
        mask[i] = (uint8_t)TS_COMPARE(op, a, b); \
    } \
    return 0; \
} \
int ts_extended_select_##suffix(type *out, const uint8_t *mask, const type *on_true, \
                                 const type *on_false, type true_scalar, type false_scalar, size_t n) { \
    for (size_t i = 0; i < n; ++i) \
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
    for (size_t i = 0; i < n; ++i) { \
        type a = left ? left[i] : left_scalar, b = right ? right[i] : right_scalar; \
        mask[i] = (uint8_t)TS_COMPARE(op, a, b); \
    } \
    return 0; \
} \
int ts_extended_select_##suffix(type *out, const uint8_t *mask, const type *on_true, \
                                 const type *on_false, type true_scalar, type false_scalar, size_t n) { \
    for (size_t i = 0; i < n; ++i) \
        out[i] = mask[i] ? (on_true ? on_true[i] : true_scalar) \
                         : (on_false ? on_false[i] : false_scalar); \
    return 0; \
}

#if defined(__aarch64__) || defined(_M_ARM64)
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
#elif defined(__x86_64__) || defined(_M_X64)
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

/* bf16 and f16 vectors are stored as uint16_t bit patterns. The scalar helpers
   work on bits, so results do not depend on the FPU rounding mode, flush-to-zero
   or trap settings. Narrowing rounds to nearest even and overflows to infinity;
   NaN keeps its sign and leading payload bits and becomes quiet. */
static uint32_t ts_float_bits(float x) { uint32_t b; memcpy(&b, &x, sizeof b); return b; }
static float ts_bits_float(uint32_t b) { float x; memcpy(&x, &b, sizeof x); return x; }

static float ts_bf16_to_f32_scalar(uint16_t h) {
    uint32_t b = (uint32_t)h << 16;
    return ts_bits_float((b & 0x7fffffffu) > 0x7f800000u ? b | 0x400000u : b);
}

static uint16_t ts_f32_to_bf16_scalar(float x) {
    uint32_t b = ts_float_bits(x);
    if ((b & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((b >> 16) | 0x40u);
    return (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
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

static uint16_t ts_f32_to_f16_scalar(float x) {
    uint32_t b = ts_float_bits(x), sign = (b >> 16) & 0x8000u, a = b & 0x7fffffffu;
    if (a > 0x7f800000u) return (uint16_t)(sign | 0x7e00u | ((a >> 13) & 0x3ffu));
    if (a >= 0x477ff000u) return (uint16_t)(sign | 0x7c00u);   /* 65520 and above */
    if (a >= 0x38800000u) {                                     /* f16 normal range */
        a -= 112u << 23;
        return (uint16_t)(sign | ((a + 0xfffu + ((a >> 13) & 1u)) >> 13));
    }
    int exponent = (int)(a >> 23);
    if (exponent < 102) return (uint16_t)sign;                  /* at most 2^-25 */
    uint32_t mantissa = (a & 0x7fffffu) | 0x800000u, shift = (uint32_t)(126 - exponent);
    uint32_t quotient = mantissa >> shift, remainder = mantissa & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1u);
    quotient += remainder > half || (remainder == half && (quotient & 1u));
    return (uint16_t)(sign | quotient);
}

#if defined(__aarch64__) || defined(_M_ARM64)
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
static uint16x4_t ts_f32_to_bf16_neon(float32x4_t x) {
    uint32x4_t b = vreinterpretq_u32_f32(x);
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(b, 16), vdupq_n_u32(1));
    uint32x4_t rounded = vaddq_u32(vaddq_u32(b, vdupq_n_u32(0x7fff)), lsb);
    uint32x4_t nan = vcgtq_u32(vandq_u32(b, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    return vshrn_n_u32(vbslq_u32(nan, vorrq_u32(b, vdupq_n_u32(0x400000)), rounded), 16);
}
void ts_extended_f32_to_bf16(uint16_t *out, const float *input, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        vst1q_u16(out + i, vcombine_u16(ts_f32_to_bf16_neon(vld1q_f32(input + i)),
                                        ts_f32_to_bf16_neon(vld1q_f32(input + i + 4))));
    for (; i < n; ++i) out[i] = ts_f32_to_bf16_scalar(input[i]);
}
#if defined(__aarch64__)
/* FCVTL and FCVTN round to nearest even and quiet NaN the same way as the
   scalar helpers under the default FPCR. */
void ts_extended_f16_to_f32(float *out, const uint16_t *input, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(out + i, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(input + i))));
    for (; i < n; ++i) out[i] = ts_f16_to_f32_scalar(input[i]);
}
void ts_extended_f32_to_f16(uint16_t *out, const float *input, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1_u16(out + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(input + i))));
    for (; i < n; ++i) out[i] = ts_f32_to_f16_scalar(input[i]);
}
#else
void ts_extended_f16_to_f32(float *out, const uint16_t *input, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = ts_f16_to_f32_scalar(input[i]);
}
void ts_extended_f32_to_f16(uint16_t *out, const float *input, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = ts_f32_to_f16_scalar(input[i]);
}
#endif
#elif defined(__x86_64__) || defined(_M_X64)
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
/* Both narrowing helpers return each result sign-extended in its 32-bit lane,
   ready for _mm_packs_epi32. */
static __m128i ts_f32_to_bf16_sse2(__m128 x) {
    __m128i b = _mm_castps_si128(x);
    __m128i lsb = _mm_and_si128(_mm_srli_epi32(b, 16), _mm_set1_epi32(1));
    __m128i rounded = _mm_add_epi32(_mm_add_epi32(b, _mm_set1_epi32(0x7fff)), lsb);
    __m128i nan = _mm_cmpgt_epi32(_mm_and_si128(b, _mm_set1_epi32(0x7fffffff)),
                                  _mm_set1_epi32(0x7f800000));
    __m128i quiet = _mm_or_si128(b, _mm_set1_epi32(0x400000));
    return _mm_srai_epi32(_mm_or_si128(_mm_and_si128(nan, quiet),
                                       _mm_andnot_si128(nan, rounded)), 16);
}
void ts_extended_f32_to_bf16(uint16_t *out, const float *input, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i *)(out + i),
                         _mm_packs_epi32(ts_f32_to_bf16_sse2(_mm_loadu_ps(input + i)),
                                         ts_f32_to_bf16_sse2(_mm_loadu_ps(input + i + 4))));
    for (; i < n; ++i) out[i] = ts_f32_to_bf16_scalar(input[i]);
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
static __m128i ts_f32_to_f16_sse2(__m128 x, int *slow) {
    __m128i b = _mm_castps_si128(x);
    __m128i a = _mm_and_si128(b, _mm_set1_epi32(0x7fffffff));
    __m128i normal = _mm_sub_epi32(a, _mm_set1_epi32(112 << 23));
    normal = _mm_add_epi32(_mm_add_epi32(normal, _mm_set1_epi32(0xfff)),
                           _mm_and_si128(_mm_srli_epi32(normal, 13), _mm_set1_epi32(1)));
    __m128i r = _mm_srli_epi32(normal, 13);
    __m128i overflow = _mm_cmpgt_epi32(a, _mm_set1_epi32(0x477fefff));
    r = _mm_or_si128(_mm_and_si128(overflow, _mm_set1_epi32(0x7c00)), _mm_andnot_si128(overflow, r));
    __m128i nan = _mm_cmpgt_epi32(a, _mm_set1_epi32(0x7f800000));
    __m128i payload = _mm_or_si128(_mm_set1_epi32(0x7e00),
                                   _mm_and_si128(_mm_srli_epi32(a, 13), _mm_set1_epi32(0x3ff)));
    r = _mm_or_si128(_mm_and_si128(nan, payload), _mm_andnot_si128(nan, r));
    __m128i small = _mm_cmplt_epi32(a, _mm_set1_epi32(0x38800000));
    __m128i zero = _mm_cmplt_epi32(a, _mm_set1_epi32(0x33000001));
    *slow |= _mm_movemask_epi8(_mm_andnot_si128(zero, small));
    r = _mm_andnot_si128(small, r);
    r = _mm_or_si128(r, _mm_and_si128(_mm_srli_epi32(b, 16), _mm_set1_epi32(0x8000)));
    return _mm_srai_epi32(_mm_slli_epi32(r, 16), 16);
}
void ts_extended_f32_to_f16(uint16_t *out, const float *input, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int slow = 0;
        __m128i low = ts_f32_to_f16_sse2(_mm_loadu_ps(input + i), &slow);
        __m128i high = ts_f32_to_f16_sse2(_mm_loadu_ps(input + i + 4), &slow);
        if (slow)
            for (size_t j = i; j < i + 8; ++j) out[j] = ts_f32_to_f16_scalar(input[j]);
        else
            _mm_storeu_si128((__m128i *)(out + i), _mm_packs_epi32(low, high));
    }
    for (; i < n; ++i) out[i] = ts_f32_to_f16_scalar(input[i]);
}
#else
void ts_extended_bf16_to_f32(float *out, const uint16_t *input, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = ts_bf16_to_f32_scalar(input[i]);
}
void ts_extended_f32_to_bf16(uint16_t *out, const float *input, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = ts_f32_to_bf16_scalar(input[i]);
}
void ts_extended_f16_to_f32(float *out, const uint16_t *input, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = ts_f16_to_f32_scalar(input[i]);
}
void ts_extended_f32_to_f16(uint16_t *out, const float *input, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = ts_f32_to_f16_scalar(input[i]);
}
#endif

TS_EXTENDED_FLOAT(f32, float, -, fabsf, sqrtf)
TS_EXTENDED_FLOAT(f64, double, -, fabs, sqrt)

int ts_extended_mask_reduce(unsigned op, const uint8_t *mask, size_t n, size_t *result) {
    size_t total = 0;
    for (size_t i = 0; i < n; ++i) total += mask[i] != 0;
    *result = op == 0 ? total : op == 1 ? total != 0 : total == n;
    return 0;
}
