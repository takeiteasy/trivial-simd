#ifndef TS_CONVERSION_H
#define TS_CONVERSION_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
#include <arm_neon.h>
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
#include <emmintrin.h>
#endif
#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
static void ts_store_s32_f32(float *out, int32x4_t value) {
    vst1q_f32(out, vcvtq_f32_s32(value));
}

static void ts_store_u32_f32(float *out, uint32x4_t value) {
    vst1q_f32(out, vcvtq_f32_u32(value));
}

static void ts_store_s32_f64(double *out, int32x4_t value) {
    vst1q_f64(out, vcvtq_f64_s64(vmovl_s32(vget_low_s32(value))));
    vst1q_f64(out + 2, vcvtq_f64_s64(vmovl_s32(vget_high_s32(value))));
}

static void ts_store_u32_f64(double *out, uint32x4_t value) {
    vst1q_f64(out, vcvtq_f64_u64(vmovl_u32(vget_low_u32(value))));
    vst1q_f64(out + 2, vcvtq_f64_u64(vmovl_u32(vget_high_u32(value))));
}

#define TS_LOAD_BYTES(suffix, type, name, source, sign, tag) \
static size_t ts_load_##name##_##suffix(type *out, const source *input, size_t n) { \
    size_t i = 0; \
    for (; n - i >= 8; i += 8) { \
        sign##16x8_t value = vmovl_##name(vld1_##name(input + i)); \
        ts_store_##tag##32_##suffix(out + i, vmovl_##tag##16(vget_low_##tag##16(value))); \
        ts_store_##tag##32_##suffix(out + i + 4, vmovl_##tag##16(vget_high_##tag##16(value))); \
    } \
    return i; \
}

#define TS_LOAD_WORDS(suffix, type, name, source, tag) \
static size_t ts_load_##name##_##suffix(type *out, const source *input, size_t n) { \
    size_t i = 0; \
    for (; n - i >= 4; i += 4) \
        ts_store_##tag##32_##suffix(out + i, vmovl_##name(vld1_##name(input + i))); \
    return i; \
}

#define TS_LOAD_DWORDS(suffix, type, name, source) \
static size_t ts_load_##name##_##suffix(type *out, const source *input, size_t n) { \
    size_t i = 0; \
    for (; n - i >= 4; i += 4) ts_store_##name##_##suffix(out + i, vld1q_##name(input + i)); \
    return i; \
}

#define TS_LOAD_NARROW(suffix, type) \
TS_LOAD_BYTES(suffix, type, s8, int8_t, int, s) \
TS_LOAD_BYTES(suffix, type, u8, uint8_t, uint, u) \
TS_LOAD_WORDS(suffix, type, s16, int16_t, s) \
TS_LOAD_WORDS(suffix, type, u16, uint16_t, u) \
TS_LOAD_DWORDS(suffix, type, s32, int32_t) \
TS_LOAD_DWORDS(suffix, type, u32, uint32_t)

TS_LOAD_NARROW(f32, float)
TS_LOAD_NARROW(f64, double)

#define TS_LOAD_QWORDS(name, source) \
static size_t ts_load_##name##_f64(double *out, const source *input, size_t n) { \
    size_t i = 0; \
    for (; n - i >= 2; i += 2) vst1q_f64(out + i, vcvtq_f64_##name(vld1q_##name(input + i))); \
    return i; \
}

TS_LOAD_QWORDS(s64, int64_t)
TS_LOAD_QWORDS(u64, uint64_t)
#define TS_LOAD_PACKED(suffix, name, out, input, n) ts_load_##name##_##suffix(out, input, n)
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
static __m128i ts_input_s8(const int8_t *p) {
    uint32_t bits; memcpy(&bits, p, sizeof bits);
    __m128i v = _mm_cvtsi32_si128((int32_t)bits);
    v = _mm_unpacklo_epi8(v, _mm_cmpgt_epi8(_mm_setzero_si128(), v));
    return _mm_unpacklo_epi16(v, _mm_cmpgt_epi16(_mm_setzero_si128(), v));
}
static __m128i ts_input_u8(const uint8_t *p) {
    uint32_t bits; memcpy(&bits, p, sizeof bits);
    __m128i v = _mm_cvtsi32_si128((int32_t)bits);
    return _mm_unpacklo_epi16(_mm_unpacklo_epi8(v, _mm_setzero_si128()), _mm_setzero_si128());
}
static __m128i ts_input_s16(const int16_t *p) {
    __m128i v = _mm_loadl_epi64((const __m128i *)p);
    return _mm_unpacklo_epi16(v, _mm_cmpgt_epi16(_mm_setzero_si128(), v));
}
static __m128i ts_input_u16(const uint16_t *p) {
    return _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)p), _mm_setzero_si128());
}
static __m128i ts_input_s32(const int32_t *p) { return _mm_loadu_si128((const __m128i *)p); }
static __m128i ts_input_u32(const uint32_t *p) { return _mm_loadu_si128((const __m128i *)p); }
static __m128 ts_input_u32_f32(__m128i v) {
    /* Halve odd unsigned values with a sticky bit before the signed conversion. */
    __m128i high = _mm_srai_epi32(v, 31);
    __m128i half = _mm_or_si128(_mm_srli_epi32(v, 1), _mm_and_si128(v, _mm_set1_epi32(1)));
    __m128 x = _mm_cvtepi32_ps(_mm_or_si128(_mm_and_si128(high, half), _mm_andnot_si128(high, v)));
    return _mm_mul_ps(x, _mm_or_ps(_mm_and_ps(_mm_castsi128_ps(high), _mm_set1_ps(2)),
                                  _mm_andnot_ps(_mm_castsi128_ps(high), _mm_set1_ps(1))));
}
static __m128d ts_input_u32_f64(__m128i v) {
    __m128d x = _mm_cvtepi32_pd(_mm_and_si128(v, _mm_set1_epi32(INT32_MAX)));
    return _mm_add_pd(x, _mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_epi32(v, 31)), _mm_set1_pd(2147483648.0)));
}
#define TS_LOAD_X86(name, source, f32, f64) \
static size_t ts_load_##name##_f32(float *out, const source *input, size_t n) { \
    size_t i = 0; \
    for (; n - i >= 4; i += 4) _mm_storeu_ps(out + i, f32(ts_input_##name(input + i))); \
    return i; \
} \
static size_t ts_load_##name##_f64(double *out, const source *input, size_t n) { \
    size_t i = 0; \
    for (; n - i >= 4; i += 4) { \
        __m128i v = ts_input_##name(input + i); \
        _mm_storeu_pd(out + i, f64(v)); \
        _mm_storeu_pd(out + i + 2, f64(_mm_srli_si128(v, 8))); \
    } \
    return i; \
}
TS_LOAD_X86(s8, int8_t, _mm_cvtepi32_ps, _mm_cvtepi32_pd)
TS_LOAD_X86(u8, uint8_t, _mm_cvtepi32_ps, _mm_cvtepi32_pd)
TS_LOAD_X86(s16, int16_t, _mm_cvtepi32_ps, _mm_cvtepi32_pd)
TS_LOAD_X86(u16, uint16_t, _mm_cvtepi32_ps, _mm_cvtepi32_pd)
TS_LOAD_X86(s32, int32_t, _mm_cvtepi32_ps, _mm_cvtepi32_pd)
TS_LOAD_X86(u32, uint32_t, ts_input_u32_f32, ts_input_u32_f64)
#define ts_load_s64_f64(out, input, n) 0
#define ts_load_u64_f64(out, input, n) 0
#define TS_LOAD_PACKED(suffix, name, out, input, n) ts_load_##name##_##suffix(out, input, n)
#else
#define TS_LOAD_PACKED(suffix, name, out, input, n) 0
#endif


#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
#define TS_NARROW_PAIR(name, source, target, width, load, narrow, combine, store) \
static size_t ts_narrow_##name(target *out, const source *input, size_t n) { \
    size_t i = 0; \
    for (; n - i >= width * 2; i += width * 2) \
        store(out + i, combine(narrow(load(input + i)), narrow(load(input + i + width)))); \
    return i; \
}
#define TS_NEON_U16_S8(v) vqmovn_s16(vreinterpretq_s16_u16(vminq_u16(v, vdupq_n_u16(127))))
#define TS_NEON_U32_S16(v) vqmovn_s32(vreinterpretq_s32_u32(vminq_u32(v, vdupq_n_u32(32767))))
TS_NARROW_PAIR(s16_s8, int16_t, int8_t, 8, vld1q_s16, vqmovn_s16, vcombine_s8, vst1q_s8)
TS_NARROW_PAIR(s16_u8, int16_t, uint8_t, 8, vld1q_s16, vqmovun_s16, vcombine_u8, vst1q_u8)
TS_NARROW_PAIR(u16_s8, uint16_t, int8_t, 8, vld1q_u16, TS_NEON_U16_S8, vcombine_s8, vst1q_s8)
TS_NARROW_PAIR(u16_u8, uint16_t, uint8_t, 8, vld1q_u16, vqmovn_u16, vcombine_u8, vst1q_u8)
TS_NARROW_PAIR(s32_s16, int32_t, int16_t, 4, vld1q_s32, vqmovn_s32, vcombine_s16, vst1q_s16)
TS_NARROW_PAIR(s32_u16, int32_t, uint16_t, 4, vld1q_s32, vqmovun_s32, vcombine_u16, vst1q_u16)
TS_NARROW_PAIR(u32_s16, uint32_t, int16_t, 4, vld1q_u32, TS_NEON_U32_S16, vcombine_s16, vst1q_s16)
TS_NARROW_PAIR(u32_u16, uint32_t, uint16_t, 4, vld1q_u32, vqmovn_u32, vcombine_u16, vst1q_u16)
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
static __m128i ts_clamp_u16(__m128i x, int limit) {
    __m128i bound = _mm_set1_epi16((int16_t)limit), bias = _mm_set1_epi16(INT16_MIN);
    __m128i over = _mm_cmpgt_epi16(_mm_xor_si128(x, bias), _mm_xor_si128(bound, bias));
    return _mm_or_si128(_mm_and_si128(over, bound), _mm_andnot_si128(over, x));
}
static __m128i ts_clamp32(__m128i x, int limit, int sign) {
    __m128i bound = _mm_set1_epi32(limit), bias = _mm_set1_epi32(INT32_MIN);
    __m128i over = _mm_cmpgt_epi32(sign ? x : _mm_xor_si128(x, bias), sign ? bound : _mm_xor_si128(bound, bias));
    x = _mm_or_si128(_mm_and_si128(over, bound), _mm_andnot_si128(over, x));
    return sign ? _mm_andnot_si128(_mm_cmpgt_epi32(_mm_setzero_si128(), x), x) : x;
}
#define TS_NARROW_PAIR(name, source, target, width, prepare, pack, finish) \
static size_t ts_narrow_##name(target *out, const source *input, size_t n) { \
    size_t i = 0; \
    for (; n - i >= width * 2; i += width * 2) { \
        __m128i a = prepare(_mm_loadu_si128((const __m128i *)(input + i))); \
        __m128i b = prepare(_mm_loadu_si128((const __m128i *)(input + i + width))); \
        _mm_storeu_si128((__m128i *)(out + i), finish(pack(a, b))); \
    } \
    return i; \
}
#define TS_IDENTITY(x) (x)
#define TS_U16_S8(x) ts_clamp_u16(x, 127)
#define TS_U16_U8(x) ts_clamp_u16(x, 255)
#define TS_U32_S16(x) ts_clamp32(x, 32767, 0)
#define TS_S32_U16(x) _mm_sub_epi32(ts_clamp32(x, 65535, 1), _mm_set1_epi32(32768))
#define TS_U32_U16(x) _mm_sub_epi32(ts_clamp32(x, 65535, 0), _mm_set1_epi32(32768))
#define TS_UNSIGNED_WORD(x) _mm_xor_si128(x, _mm_set1_epi16(INT16_MIN))
TS_NARROW_PAIR(s16_s8, int16_t, int8_t, 8, TS_IDENTITY, _mm_packs_epi16, TS_IDENTITY)
TS_NARROW_PAIR(s16_u8, int16_t, uint8_t, 8, TS_IDENTITY, _mm_packus_epi16, TS_IDENTITY)
TS_NARROW_PAIR(u16_s8, uint16_t, int8_t, 8, TS_U16_S8, _mm_packs_epi16, TS_IDENTITY)
TS_NARROW_PAIR(u16_u8, uint16_t, uint8_t, 8, TS_U16_U8, _mm_packus_epi16, TS_IDENTITY)
TS_NARROW_PAIR(s32_s16, int32_t, int16_t, 4, TS_IDENTITY, _mm_packs_epi32, TS_IDENTITY)
TS_NARROW_PAIR(s32_u16, int32_t, uint16_t, 4, TS_S32_U16, _mm_packs_epi32, TS_UNSIGNED_WORD)
TS_NARROW_PAIR(u32_s16, uint32_t, int16_t, 4, TS_U32_S16, _mm_packs_epi32, TS_IDENTITY)
TS_NARROW_PAIR(u32_u16, uint32_t, uint16_t, 4, TS_U32_U16, _mm_packs_epi32, TS_UNSIGNED_WORD)
#else
#define TS_NARROW_PAIR(name, source, target) \
static size_t ts_narrow_##name(target *out, const source *input, size_t n) { \
    (void)out; (void)input; (void)n; return 0; \
}
TS_NARROW_PAIR(s16_s8, int16_t, int8_t)
TS_NARROW_PAIR(s16_u8, int16_t, uint8_t)
TS_NARROW_PAIR(u16_s8, uint16_t, int8_t)
TS_NARROW_PAIR(u16_u8, uint16_t, uint8_t)
TS_NARROW_PAIR(s32_s16, int32_t, int16_t)
TS_NARROW_PAIR(s32_u16, int32_t, uint16_t)
TS_NARROW_PAIR(u32_s16, uint32_t, int16_t)
TS_NARROW_PAIR(u32_u16, uint32_t, uint16_t)
#endif
#undef TS_NARROW_PAIR
#endif
