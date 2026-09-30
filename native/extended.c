#include <math.h>
#include <stddef.h>
#include <stdint.h>

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

TS_EXTENDED_FLOAT(f32, float, -, fabsf, sqrtf)
TS_EXTENDED_FLOAT(f64, double, -, fabs, sqrt)

int ts_extended_mask_reduce(unsigned op, const uint8_t *mask, size_t n, size_t *result) {
    size_t total = 0;
    for (size_t i = 0; i < n; ++i) total += mask[i] != 0;
    *result = op == 0 ? total : op == 1 ? total != 0 : total == n;
    return 0;
}
