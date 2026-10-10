#ifndef TS_MASKS_H
#define TS_MASKS_H

#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
#include <arm_neon.h>
static void ts_mask_bytes8(uint8_t *out, uint8x16_t truth) { vst1q_u8(out, vandq_u8(truth, vdupq_n_u8(1))); }
static void ts_mask_bytes16(uint8_t *out, uint16x8_t truth) { uint8_t lanes[8]; vst1_u8(lanes, vand_u8(vmovn_u16(truth), vdup_n_u8(1))); memcpy(out, lanes, 8); }
static void ts_mask_bytes32(uint8_t *out, uint32x4_t truth) { uint8_t lanes[8]; vst1_u8(lanes, vand_u8(vmovn_u16(vcombine_u16(vmovn_u32(truth), vmovn_u32(truth))), vdup_n_u8(1))); memcpy(out, lanes, 4); }
static void ts_mask_bytes64(uint8_t *out, uint64x2_t truth) {
    uint32x2_t words = vmovn_u64(truth);
    uint16x4_t halves = vmovn_u32(vcombine_u32(words, words));
    uint8_t lanes[8];
    vst1_u8(lanes, vand_u8(vmovn_u16(vcombine_u16(halves, halves)), vdup_n_u8(1)));
    memcpy(out, lanes, 2);
}

#define TS_MASK_PACK(suffix, type, vector, mask, width, load, store, splat, eq, lt, le, invert, choose, write_bytes) \
static size_t ts_mask_compare_##suffix(unsigned op, void *output, const void *a, const void *b, \
                                     type sa, type sb, size_t n, int bytes) { \
    type *out = bytes ? NULL : output; const type *left = a, *right = b; size_t i = 0; \
    for (; n - i >= width; i += width) { \
        vector x = left ? load(left + i) : splat(sa), y = right ? load(right + i) : splat(sb); \
        mask truth = op == 0 ? eq(x, y) : op == 1 ? invert(eq(x, y)) : \
                     op == 2 ? lt(x, y) : op == 3 ? le(x, y) : op == 4 ? lt(y, x) : le(y, x); \
        if (bytes) write_bytes((uint8_t *)output + i, truth); \
        else store(out + i, choose(truth, splat(1), splat(0))); \
    } \
    return i; \
} \
static size_t ts_mask_select_##suffix(void *output, const void *bits, const void *a, const void *b, \
                                    type sa, type sb, size_t n, int bytes) { \
    type *out = output; const type *left = a, *right = b, *numeric = bits; \
    const uint8_t *mask_bytes = bits; size_t i = 0; \
    for (; n - i >= width; i += width) { \
        type lanes[width]; \
        if (bytes) for (size_t j = 0; j < width; ++j) lanes[j] = mask_bytes[i + j] != 0; \
        vector m = load(bytes ? lanes : numeric + i); \
        vector x = left ? load(left + i) : splat(sa), y = right ? load(right + i) : splat(sb); \
        store(out + i, choose(invert(eq(m, splat(0))), x, y)); \
    } \
    return i; \
}
TS_MASK_PACK(f32, float, float32x4_t, uint32x4_t, 4, vld1q_f32, vst1q_f32, vdupq_n_f32, vceqq_f32, vcltq_f32, vcleq_f32, vmvnq_u32, vbslq_f32, ts_mask_bytes32)
static uint64x2_t ts_not_u64(uint64x2_t x) { return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(x))); }
TS_MASK_PACK(f64, double, float64x2_t, uint64x2_t, 2, vld1q_f64, vst1q_f64, vdupq_n_f64, vceqq_f64, vcltq_f64, vcleq_f64, ts_not_u64, vbslq_f64, ts_mask_bytes64)
TS_MASK_PACK(s8, int8_t, int8x16_t, uint8x16_t, 16, vld1q_s8, vst1q_s8, vdupq_n_s8, vceqq_s8, vcltq_s8, vcleq_s8, vmvnq_u8, vbslq_s8, ts_mask_bytes8)
TS_MASK_PACK(u8, uint8_t, uint8x16_t, uint8x16_t, 16, vld1q_u8, vst1q_u8, vdupq_n_u8, vceqq_u8, vcltq_u8, vcleq_u8, vmvnq_u8, vbslq_u8, ts_mask_bytes8)
TS_MASK_PACK(s16, int16_t, int16x8_t, uint16x8_t, 8, vld1q_s16, vst1q_s16, vdupq_n_s16, vceqq_s16, vcltq_s16, vcleq_s16, vmvnq_u16, vbslq_s16, ts_mask_bytes16)
TS_MASK_PACK(u16, uint16_t, uint16x8_t, uint16x8_t, 8, vld1q_u16, vst1q_u16, vdupq_n_u16, vceqq_u16, vcltq_u16, vcleq_u16, vmvnq_u16, vbslq_u16, ts_mask_bytes16)
TS_MASK_PACK(s32, int32_t, int32x4_t, uint32x4_t, 4, vld1q_s32, vst1q_s32, vdupq_n_s32, vceqq_s32, vcltq_s32, vcleq_s32, vmvnq_u32, vbslq_s32, ts_mask_bytes32)
TS_MASK_PACK(u32, uint32_t, uint32x4_t, uint32x4_t, 4, vld1q_u32, vst1q_u32, vdupq_n_u32, vceqq_u32, vcltq_u32, vcleq_u32, vmvnq_u32, vbslq_u32, ts_mask_bytes32)
TS_MASK_PACK(s64, int64_t, int64x2_t, uint64x2_t, 2, vld1q_s64, vst1q_s64, vdupq_n_s64, vceqq_s64, vcltq_s64, vcleq_s64, ts_not_u64, vbslq_s64, ts_mask_bytes64)
TS_MASK_PACK(u64, uint64_t, uint64x2_t, uint64x2_t, 2, vld1q_u64, vst1q_u64, vdupq_n_u64, vceqq_u64, vcltq_u64, vcleq_u64, ts_not_u64, vbslq_u64, ts_mask_bytes64)
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
#include <emmintrin.h>
#include <xmmintrin.h>

static void ts_mask_bytes_x86(uint8_t *out, __m128i truth, unsigned bits) {
    if (bits == 64) truth = _mm_shuffle_epi32(truth, _MM_SHUFFLE(2, 0, 2, 0));
    if (bits >= 32) truth = _mm_packs_epi32(truth, truth);
    if (bits >= 16) truth = _mm_packs_epi16(truth, truth);
    truth = _mm_and_si128(truth, _mm_set1_epi8(1));
    uint8_t lanes[16]; _mm_storeu_si128((__m128i *)lanes, truth);
    memcpy(out, lanes, 128 / bits);
}
static __m128i ts_eq64(__m128i x, __m128i y) {
    __m128i equal = _mm_cmpeq_epi32(x, y);
    return _mm_and_si128(equal, _mm_shuffle_epi32(equal, _MM_SHUFFLE(2, 3, 0, 1)));
}
static __m128i ts_gt64(__m128i x, __m128i y, int sign) {
    __m128i high_x = _mm_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i high_y = _mm_shuffle_epi32(y, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i bias = _mm_set1_epi32(INT32_MIN);
    __m128i high = _mm_cmpgt_epi32(sign ? high_x : _mm_xor_si128(high_x, bias),
                                  sign ? high_y : _mm_xor_si128(high_y, bias));
    __m128i low = _mm_cmpgt_epi32(_mm_xor_si128(x, bias), _mm_xor_si128(y, bias));
    low = _mm_shuffle_epi32(low, _MM_SHUFFLE(2, 2, 0, 0));
    return _mm_or_si128(high, _mm_and_si128(_mm_cmpeq_epi32(high_x, high_y), low));
}
#define TS_MASK_FLOAT(suffix, type, vector, width, load, store, splat, eq, ne, lt, le, andv, orv, andnot, to_integer) \
static size_t ts_mask_compare_##suffix(unsigned op, void *output, const void *a, const void *b, \
                                     type sa, type sb, size_t n, int bytes) { \
    type *out = bytes ? NULL : output; const type *left = a, *right = b; size_t i = 0; \
    for (; n - i >= width; i += width) { \
        vector x = left ? load(left + i) : splat(sa), y = right ? load(right + i) : splat(sb); \
        vector truth = op == 0 ? eq(x, y) : op == 1 ? ne(x, y) : op == 2 ? lt(x, y) : \
                       op == 3 ? le(x, y) : op == 4 ? lt(y, x) : le(y, x); \
        if (bytes) ts_mask_bytes_x86((uint8_t *)output + i, to_integer(truth), sizeof(type) * 8); \
        else store(out + i, andv(truth, splat(1))); \
    } \
    return i; \
} \
static size_t ts_mask_select_##suffix(void *output, const void *bits, const void *a, const void *b, \
                                    type sa, type sb, size_t n, int bytes) { \
    type *out = output; const type *left = a, *right = b, *numeric = bits; \
    const uint8_t *mask_bytes = bits; size_t i = 0; \
    for (; n - i >= width; i += width) { \
        type lanes[width]; \
        if (bytes) for (size_t j = 0; j < width; ++j) lanes[j] = mask_bytes[i + j] != 0; \
        vector m = ne(load(bytes ? lanes : numeric + i), splat(0)); \
        vector x = left ? load(left + i) : splat(sa), y = right ? load(right + i) : splat(sb); \
        store(out + i, orv(andv(m, x), andnot(m, y))); \
    } \
    return i; \
}
TS_MASK_FLOAT(f32, float, __m128, 4, _mm_loadu_ps, _mm_storeu_ps, _mm_set1_ps,
              _mm_cmpeq_ps, _mm_cmpneq_ps, _mm_cmplt_ps, _mm_cmple_ps, _mm_and_ps, _mm_or_ps, _mm_andnot_ps, _mm_castps_si128)
TS_MASK_FLOAT(f64, double, __m128d, 2, _mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd,
              _mm_cmpeq_pd, _mm_cmpneq_pd, _mm_cmplt_pd, _mm_cmple_pd, _mm_and_pd, _mm_or_pd, _mm_andnot_pd, _mm_castpd_si128)

#define TS_MASK_INTEGER(suffix, type, width, splat, equal, greater) \
static size_t ts_mask_compare_##suffix(unsigned op, void *output, const void *a, const void *b, \
                                     type sa, type sb, size_t n, int bytes) { \
    type *out = bytes ? NULL : output; const type *left = a, *right = b; size_t i = 0; \
    __m128i ones = _mm_set1_epi32(-1); \
    for (; n - i >= width; i += width) { \
        __m128i x = left ? _mm_loadu_si128((const __m128i *)(left + i)) : splat(sa); \
        __m128i y = right ? _mm_loadu_si128((const __m128i *)(right + i)) : splat(sb); \
        __m128i e = equal(x, y), g = greater(x, y), l = greater(y, x); \
        __m128i truth = op == 0 ? e : op == 1 ? _mm_xor_si128(e, ones) : op == 2 ? l : \
                        op == 3 ? _mm_xor_si128(g, ones) : op == 4 ? g : _mm_xor_si128(l, ones); \
        if (bytes) ts_mask_bytes_x86((uint8_t *)output + i, truth, sizeof(type) * 8); \
        else _mm_storeu_si128((__m128i *)(out + i), _mm_and_si128(truth, splat(1))); \
    } \
    return i; \
} \
static size_t ts_mask_select_##suffix(void *output, const void *bits, const void *a, const void *b, \
                                    type sa, type sb, size_t n, int bytes) { \
    type *out = output; const type *left = a, *right = b, *numeric = bits; \
    const uint8_t *mask_bytes = bits; size_t i = 0; \
    for (; n - i >= width; i += width) { \
        type lanes[width]; \
        if (bytes) for (size_t j = 0; j < width; ++j) lanes[j] = mask_bytes[i + j] != 0; \
        __m128i m = _mm_loadu_si128((const __m128i *)(bytes ? lanes : numeric + i)); \
        m = equal(m, splat(0)); \
        __m128i x = left ? _mm_loadu_si128((const __m128i *)(left + i)) : splat(sa); \
        __m128i y = right ? _mm_loadu_si128((const __m128i *)(right + i)) : splat(sb); \
        _mm_storeu_si128((__m128i *)(out + i), _mm_or_si128(_mm_andnot_si128(m, x), _mm_and_si128(m, y))); \
    } \
    return i; \
}
TS_MASK_INTEGER(s8, int8_t, 16, _mm_set1_epi8, _mm_cmpeq_epi8, _mm_cmpgt_epi8)
#define ts_gt_u8(x, y) _mm_cmpgt_epi8(_mm_xor_si128(x, _mm_set1_epi8(INT8_MIN)), _mm_xor_si128(y, _mm_set1_epi8(INT8_MIN)))
TS_MASK_INTEGER(u8, uint8_t, 16, _mm_set1_epi8, _mm_cmpeq_epi8, ts_gt_u8)
TS_MASK_INTEGER(s16, int16_t, 8, _mm_set1_epi16, _mm_cmpeq_epi16, _mm_cmpgt_epi16)
#define ts_gt_u16(x, y) _mm_cmpgt_epi16(_mm_xor_si128(x, _mm_set1_epi16(INT16_MIN)), _mm_xor_si128(y, _mm_set1_epi16(INT16_MIN)))
TS_MASK_INTEGER(u16, uint16_t, 8, _mm_set1_epi16, _mm_cmpeq_epi16, ts_gt_u16)
TS_MASK_INTEGER(s32, int32_t, 4, _mm_set1_epi32, _mm_cmpeq_epi32, _mm_cmpgt_epi32)
#define ts_gt_u32(x, y) _mm_cmpgt_epi32(_mm_xor_si128(x, _mm_set1_epi32(INT32_MIN)), _mm_xor_si128(y, _mm_set1_epi32(INT32_MIN)))
TS_MASK_INTEGER(u32, uint32_t, 4, _mm_set1_epi32, _mm_cmpeq_epi32, ts_gt_u32)
#define ts_gt_s64(x, y) ts_gt64(x, y, 1)
TS_MASK_INTEGER(s64, int64_t, 2, _mm_set1_epi64x, ts_eq64, ts_gt_s64)
#define ts_gt_u64(x, y) ts_gt64(x, y, 0)
TS_MASK_INTEGER(u64, uint64_t, 2, _mm_set1_epi64x, ts_eq64, ts_gt_u64)
#else
#define TS_MASK_SCALAR(suffix, type) \
static size_t ts_mask_compare_##suffix(unsigned op, void *out, const void *a, const void *b, \
                                     type sa, type sb, size_t n, int bytes) { \
    (void)bytes; (void)op; (void)out; (void)a; (void)b; (void)sa; (void)sb; (void)n; return 0; \
} \
static size_t ts_mask_select_##suffix(void *out, const void *mask, const void *a, const void *b, \
                                    type sa, type sb, size_t n, int bytes) { \
    (void)out; (void)mask; (void)a; (void)b; (void)sa; (void)sb; (void)n; (void)bytes; return 0; \
}
TS_MASK_SCALAR(f32, float)
TS_MASK_SCALAR(f64, double)
TS_MASK_SCALAR(s8, int8_t)
TS_MASK_SCALAR(u8, uint8_t)
TS_MASK_SCALAR(s16, int16_t)
TS_MASK_SCALAR(u16, uint16_t)
TS_MASK_SCALAR(s32, int32_t)
TS_MASK_SCALAR(u32, uint32_t)
TS_MASK_SCALAR(s64, int64_t)
TS_MASK_SCALAR(u64, uint64_t)
#endif

static size_t ts_mask_count(const uint8_t *mask, size_t n) {
    size_t i = 0, total = 0;
#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
    for (; n - i >= 16; i += 16) {
        uint8x16_t truth = vandq_u8(vmvnq_u8(vceqq_u8(vld1q_u8(mask + i), vdupq_n_u8(0))), vdupq_n_u8(1));
        total += vaddvq_u8(truth);
    }
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
    for (; n - i >= 16; i += 16) {
        unsigned bits = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(
            _mm_loadu_si128((const __m128i *)(mask + i)), _mm_setzero_si128())) ^ 0xffffu;
        bits = bits - ((bits >> 1) & 0x5555u);
        bits = (bits & 0x3333u) + ((bits >> 2) & 0x3333u);
        bits = (bits + (bits >> 4)) & 0x0f0fu;
        total += (bits + (bits >> 8)) & 0x1fu;
    }
#endif
    for (; i < n; ++i) total += mask[i] != 0;
    return total;
}
#endif
