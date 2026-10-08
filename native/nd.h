/* Layouts are validated by the Lisp entry points; input aliases are snapshotted. */
enum {
    TS_ND_ADD, TS_ND_SUBTRACT, TS_ND_MULTIPLY, TS_ND_DIVIDE,
    TS_ND_NEGATE, TS_ND_ABS, TS_ND_SQRT, TS_ND_RECIPROCAL,
    TS_ND_MIN, TS_ND_MAX, TS_ND_CLAMP, TS_ND_COMPARE, TS_ND_SELECT, TS_ND_CONVERT,
    TS_ND_LOG, TS_ND_TANH, TS_ND_SIGMOID,
    TS_ND_EXP, TS_ND_SIN, TS_ND_COS, TS_ND_SILU, TS_ND_GELU
};

int ts_nd_activation_math(void) { return 1; }
int ts_nd_inference_math(void) { return 1; }

static float ts_nd_gelu_f32(float x) {
    float cube = x * (x * x);
    float angle = 0.7978845608028654f * (x + 0.044715f * cube);
    return (0.5f * x) * (1.0f + tanhf(angle));
}

static double ts_nd_gelu_f64(double x) {
    double cube = x * (x * x);
    double angle = 0.7978845608028654 * (x + 0.044715 * cube);
    return (0.5 * x) * (1.0 + tanh(angle));
}

static const unsigned ts_nd_binary_ops[] = {TS_OP_ADD, TS_OP_SUBTRACT, TS_OP_MULTIPLY, TS_OP_DIVIDE};

#define TS_ND_COMPARE(op, a, b) ((op) == 0 ? (a) == (b) : (op) == 1 ? (a) != (b) : \
                                (op) == 2 ? (a) < (b) : (op) == 3 ? (a) <= (b) : \
                                (op) == 4 ? (a) > (b) : (a) >= (b))

void ts_nd_convert(void *out, const void *input, size_t n, int64_t ds, int64_t is,
                   unsigned destination_type, unsigned input_type, int rounding);

#define TS_ND_DECLARATIONS(suffix, type) \
int ts_extended_unary_##suffix(unsigned, type *, const type *, size_t); \
int ts_extended_minmax_##suffix(unsigned, type *, const type *, const type *, type, type, size_t); \
int ts_extended_clamp_##suffix(type *, const type *, const type *, const type *, type, type, size_t); \
int ts_extended_compare_##suffix(unsigned, uint8_t *, const type *, const type *, type, type, size_t); \
int ts_extended_select_##suffix(type *, const uint8_t *, const type *, const type *, type, type, size_t);

#define TS_ND_INTEGER_BINARY(suffix, type, bits_type) \
static int ts_nd_binary_##suffix(unsigned op, type *out, const type *a, const type *b, \
                                 type av, type bv, size_t n) { \
    if (!a || !b) return ts_bulk_binary_##suffix(ts_nd_binary_ops[op], (bits_type *)out, \
                                                (const bits_type *)a, (const bits_type *)b, av, bv, n); \
    switch (op) { \
    case TS_ND_ADD: return ts_add_##suffix((bits_type *)out, (const bits_type *)a, (const bits_type *)b, n); \
    case TS_ND_SUBTRACT: return ts_subtract_##suffix((bits_type *)out, (const bits_type *)a, (const bits_type *)b, n); \
    case TS_ND_MULTIPLY: return ts_multiply_##suffix((bits_type *)out, (const bits_type *)a, (const bits_type *)b, n); \
    default: return ts_divide_##suffix((bits_type *)out, (const bits_type *)a, (const bits_type *)b, n); \
    } \
}

#define TS_ND_FLOAT_BINARY(suffix, type) \
static int ts_nd_binary_##suffix(unsigned op, type *out, const type *a, const type *b, \
                                 type av, type bv, size_t n) { \
    if (!a || !b) ts_bulk_binary_##suffix(op, out, a, b, av, bv, n); \
    else if (op == TS_ND_ADD) ts_add_##suffix(out, a, b, n); \
    else if (op == TS_ND_SUBTRACT) ts_subtract_##suffix(out, a, b, n); \
    else if (op == TS_ND_MULTIPLY) ts_multiply_##suffix(out, a, b, n); \
    else ts_divide_##suffix(out, a, b, n); \
    return 0; \
} \
static type ts_nd_value_##suffix(unsigned op, type a, type b) { \
    switch (op) { \
    case TS_ND_ADD: return a + b; \
    case TS_ND_SUBTRACT: return a - b; \
    case TS_ND_MULTIPLY: return a * b; \
    default: return a / b; \
    } \
}

TS_ND_FLOAT_BINARY(f32, float)
TS_ND_FLOAT_BINARY(f64, double)
TS_ND_INTEGER_BINARY(s8, int8_t, uint8_t)
TS_ND_INTEGER_BINARY(u8, uint8_t, uint8_t)
TS_ND_INTEGER_BINARY(s16, int16_t, uint16_t)
TS_ND_INTEGER_BINARY(u16, uint16_t, uint16_t)
TS_ND_INTEGER_BINARY(s32, int32_t, uint32_t)
TS_ND_INTEGER_BINARY(u32, uint32_t, uint32_t)
TS_ND_INTEGER_BINARY(s64, int64_t, uint64_t)
TS_ND_INTEGER_BINARY(u64, uint64_t, uint64_t)

#define TS_ND_ROW(suffix, type, integer, value, square_root, absolute) \
TS_ND_DECLARATIONS(suffix, type) \
static int ts_nd_unary_##suffix(unsigned, type *, const type *, size_t); \
static int ts_nd_row_##suffix(unsigned op, void *const *p, const int64_t *s, size_t n, unsigned cmp) { \
    type *out = p[0]; \
    const type *a = p[1], *b = p[2], *c = p[3]; \
    if (s[0] == 1 && op < TS_ND_NEGATE && s[1] >= 0 && s[1] <= 1 && s[2] >= 0 && s[2] <= 1) \
        return ts_nd_binary_##suffix(op, out, s[1] ? a : NULL, s[2] ? b : NULL, a[0], b[0], n); \
    if (s[0] == 1 && op >= TS_ND_NEGATE && op <= TS_ND_RECIPROCAL && s[1] == 1) \
        return ts_nd_unary_##suffix(op, out, a, n); \
    if (s[0] == 1 && (op == TS_ND_MIN || op == TS_ND_MAX) && \
        s[1] >= 0 && s[1] <= 1 && s[2] >= 0 && s[2] <= 1) \
        return ts_extended_minmax_##suffix(op - TS_ND_MIN, out, s[1] ? a : NULL, s[2] ? b : NULL, a[0], b[0], n); \
    if (s[0] == 1 && op == TS_ND_CLAMP && s[1] == 1 && \
        s[2] >= 0 && s[2] <= 1 && s[3] >= 0 && s[3] <= 1) \
        return ts_extended_clamp_##suffix(out, a, s[2] ? b : NULL, s[3] ? c : NULL, b[0], c[0], n); \
    if (s[0] == 1 && op == TS_ND_COMPARE && s[1] >= 0 && s[1] <= 1 && s[2] >= 0 && s[2] <= 1) \
        return ts_extended_compare_##suffix(cmp, p[0], s[1] ? a : NULL, s[2] ? b : NULL, a[0], b[0], n); \
    if (s[0] == 1 && op == TS_ND_SELECT && s[1] == 1 && \
        s[2] >= 0 && s[2] <= 1 && s[3] >= 0 && s[3] <= 1) \
        return ts_extended_select_##suffix(out, p[1], s[2] ? b : NULL, s[3] ? c : NULL, b[0], c[0], n); \
    for (size_t i = 0; i < n; ++i) { \
        int64_t j = (int64_t)i; \
        if (op == TS_ND_SELECT) { \
            type result = ((const uint8_t *)p[1])[j * s[1]] ? b[j * s[2]] : c[j * s[3]]; \
            out[j * s[0]] = result; continue; \
        } \
        type x = a[j * s[1]], y = b ? b[j * s[2]] : 0; \
        if (op == TS_ND_COMPARE) { \
            ((uint8_t *)p[0])[j * s[0]] = TS_ND_COMPARE(cmp, x, y); continue; \
        } \
        type result; \
        if (op < TS_ND_NEGATE) { \
            if (integer && op == TS_ND_DIVIDE && !y) return -3; \
            result = value(integer ? ts_nd_binary_ops[op] : op, x, y); \
        } else if (op == TS_ND_NEGATE) result = integer ? value(TS_OP_NEGATE, x, 0) : -x; \
        else if (op == TS_ND_ABS) result = integer ? value(TS_OP_ABS, x, 0) : absolute(x); \
        else if (op == TS_ND_SQRT) { if (x < 0) return -2; result = x == 0 ? x : square_root(x); } \
        else if (op == TS_ND_RECIPROCAL) { if (x == 0) return -2; result = (type)1 / x; } \
        else if (op == TS_ND_LOG) result = sizeof(type) == sizeof(float) ? logf(x) : log(x); \
        else if (op == TS_ND_TANH) result = sizeof(type) == sizeof(float) ? tanhf(x) : tanh(x); \
        else if (op == TS_ND_SIGMOID) result = sizeof(type) == sizeof(float) ? ts_sigmoid_f32(x) : ts_sigmoid_f64(x); \
        else if (op == TS_ND_EXP) result = sizeof(type) == sizeof(float) ? expf(x) : exp(x); \
        else if (op == TS_ND_SIN) result = sizeof(type) == sizeof(float) ? sinf(x) : sin(x); \
        else if (op == TS_ND_COS) result = sizeof(type) == sizeof(float) ? cosf(x) : cos(x); \
        else if (op == TS_ND_SILU) result = x * (sizeof(type) == sizeof(float) ? ts_sigmoid_f32(x) : ts_sigmoid_f64(x)); \
        else if (op == TS_ND_GELU) result = sizeof(type) == sizeof(float) ? ts_nd_gelu_f32(x) : ts_nd_gelu_f64(x); \
        else if (op == TS_ND_MIN) result = y < x ? y : x; \
        else if (op == TS_ND_MAX) result = y > x ? y : x; \
        else if (op == TS_ND_CLAMP) { \
            type z = c[j * s[3]]; if (y > z) return -2; result = x < y ? y : x > z ? z : x; \
        } else return -1; \
        out[j * s[0]] = result; \
    } \
    return 0; \
}

TS_ND_ROW(f32, float, 0, ts_nd_value_f32, sqrtf, fabsf)
TS_ND_ROW(f64, double, 0, ts_nd_value_f64, sqrt, fabs)
#define TS_ND_IDENTITY(x) (x)
TS_ND_ROW(s8, int8_t, 1, ts_value_s8, sqrt, TS_ND_IDENTITY)
TS_ND_ROW(u8, uint8_t, 1, ts_value_u8, sqrt, TS_ND_IDENTITY)
TS_ND_ROW(s16, int16_t, 1, ts_value_s16, sqrt, TS_ND_IDENTITY)
TS_ND_ROW(u16, uint16_t, 1, ts_value_u16, sqrt, TS_ND_IDENTITY)
TS_ND_ROW(s32, int32_t, 1, ts_value_s32, sqrt, TS_ND_IDENTITY)
TS_ND_ROW(u32, uint32_t, 1, ts_value_u32, sqrt, TS_ND_IDENTITY)
TS_ND_ROW(s64, int64_t, 1, ts_value_s64, sqrt, TS_ND_IDENTITY)
TS_ND_ROW(u64, uint64_t, 1, ts_value_u64, sqrt, TS_ND_IDENTITY)

#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
#define TS_ND_INVALID_F32(op, x) vmaxvq_u32((op) == TS_ND_SQRT \
    ? vcltq_f32(x, vdupq_n_f32(0)) : vceqq_f32(x, vdupq_n_f32(0)))
static int ts_nd_invalid_f64(unsigned op, float64x2_t x) {
    uint64x2_t mask = op == TS_ND_SQRT ? vcltq_f64(x, vdupq_n_f64(0)) : vceqq_f64(x, vdupq_n_f64(0));
    return (vgetq_lane_u64(mask, 0) | vgetq_lane_u64(mask, 1)) != 0;
}
#define TS_ND_INVALID_F64(op, x) ts_nd_invalid_f64(op, x)
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
#define TS_ND_INVALID_F32(op, x) _mm_movemask_ps((op) == TS_ND_SQRT \
    ? _mm_cmplt_ps(x, _mm_setzero_ps()) : _mm_cmpeq_ps(x, _mm_setzero_ps()))
#define TS_ND_INVALID_F64(op, x) _mm_movemask_pd((op) == TS_ND_SQRT \
    ? _mm_cmplt_pd(x, _mm_setzero_pd()) : _mm_cmpeq_pd(x, _mm_setzero_pd()))
#else
#define TS_ND_INVALID_F32(op, x) ((op) == TS_ND_SQRT ? (x) < 0 : (x) == 0)
#define TS_ND_INVALID_F64(op, x) TS_ND_INVALID_F32(op, x)
#endif

#define TS_ND_FLOAT_UNARY(suffix, type, prefix, square_root) \
static int ts_nd_unary_##suffix(unsigned op, type *out, const type *input, size_t n) { \
    if (op < TS_ND_SQRT) return ts_extended_unary_##suffix(op - TS_ND_NEGATE, out, input, n); \
    size_t i = 0; \
    for (; i + prefix##_WIDTH <= n; i += prefix##_WIDTH) { \
        prefix##_VECTOR x = prefix##_LOAD(input + i); \
        if (TS_ND_INVALID_##prefix(op, x)) return -2; \
        prefix##_STORE(out + i, op == TS_ND_SQRT ? prefix##_SQRT(x) : prefix##_DIV(prefix##_SPLAT(1), x)); \
    } \
    for (; i < n; ++i) { \
        type x = input[i]; \
        if (op == TS_ND_SQRT ? x < 0 : x == 0) return -2; \
        out[i] = op == TS_ND_SQRT ? (x == 0 ? x : square_root(x)) : (type)1 / x; \
    } \
    return 0; \
}

#define TS_ND_INVALID_TS_F32(op, x) TS_ND_INVALID_F32(op, x)
#define TS_ND_INVALID_TS_F64(op, x) TS_ND_INVALID_F64(op, x)
TS_ND_FLOAT_UNARY(f32, float, TS_F32, sqrtf)
TS_ND_FLOAT_UNARY(f64, double, TS_F64, sqrt)

#define TS_ND_INTEGER_UNARY(suffix, type) \
static int ts_nd_unary_##suffix(unsigned op, type *out, const type *input, size_t n) { \
    if (op >= TS_ND_SQRT) return -1; \
    return ts_extended_unary_##suffix(op - TS_ND_NEGATE, out, input, n); \
}
TS_ND_INTEGER_UNARY(s8, int8_t)
TS_ND_INTEGER_UNARY(u8, uint8_t)
TS_ND_INTEGER_UNARY(s16, int16_t)
TS_ND_INTEGER_UNARY(u16, uint16_t)
TS_ND_INTEGER_UNARY(s32, int32_t)
TS_ND_INTEGER_UNARY(u32, uint32_t)
TS_ND_INTEGER_UNARY(s64, int64_t)
TS_ND_INTEGER_UNARY(u64, uint64_t)

typedef int (*ts_nd_row)(unsigned, void *const *, const int64_t *, size_t, unsigned);

int ts_nd_execute(unsigned op, unsigned type, unsigned destination_type, unsigned input_type,
                  unsigned comparison, int rounding, size_t rank, const int64_t *shape,
                  const int64_t *strides, void *const *pointers, int64_t *indices) {
    static const ts_nd_row rows[] = {ts_nd_row_f32, ts_nd_row_f64, ts_nd_row_s8, ts_nd_row_u8,
                                     ts_nd_row_s16, ts_nd_row_u16, ts_nd_row_s32, ts_nd_row_u32,
                                     ts_nd_row_s64, ts_nd_row_u64};
    static const size_t sizes[] = {4, 8, 1, 1, 2, 2, 4, 4, 8, 8};
    static const size_t conversion_sizes[] = {4, 8, 2, 2};
    if (op > TS_ND_GELU || (op >= TS_ND_LOG && type > 1) || type >= 10 ||
        comparison > 5 || rounding < 0 || rounding > 3 ||
        (op == TS_ND_CONVERT && (destination_type > 3 || input_type > 3))) return -1;
    for (size_t axis = 0; axis < rank; ++axis) {
        if (shape[axis] < 0) return -1;
        if (!shape[axis]) return 0;
        indices[axis] = 0;
    }
    size_t bytes[4] = {sizes[type], sizes[type], sizes[type], sizes[type]};
    if (op == TS_ND_COMPARE) bytes[0] = 1;
    if (op == TS_ND_SELECT) bytes[1] = 1;
    if (op == TS_ND_CONVERT) {
        bytes[0] = conversion_sizes[destination_type]; bytes[1] = conversion_sizes[input_type];
    }
    void *p[4]; int64_t steps[4];
    for (size_t operand = 0; operand < 4; ++operand) {
        p[operand] = pointers[operand];
        steps[operand] = rank ? strides[operand * rank + rank - 1] : 0;
    }
    if (!p[0] || !p[1] ||
        ((op < TS_ND_NEGATE || (op >= TS_ND_MIN && op <= TS_ND_SELECT)) && op != TS_ND_CONVERT && !p[2]) ||
        ((op == TS_ND_CLAMP || op == TS_ND_SELECT) && !p[3])) return -1;
    size_t n = rank ? (size_t)shape[rank - 1] : 1;
    for (;;) {
        int status = 0;
        if (op == TS_ND_CONVERT)
            ts_nd_convert(p[0], p[1], n, steps[0], steps[1], destination_type, input_type, rounding);
        else status = rows[type](op, p, steps, n, comparison);
        if (status) return status;
        size_t axis = rank > 1 ? rank - 1 : 0;
        while (axis) {
            --axis;
            if (++indices[axis] < shape[axis]) {
                for (size_t operand = 0; operand < 4; ++operand)
                    if (p[operand]) p[operand] = (char *)p[operand] + strides[operand * rank + axis] * (int64_t)bytes[operand];
                break;
            }
            indices[axis] = 0;
            for (size_t operand = 0; operand < 4; ++operand)
                if (p[operand]) p[operand] = (char *)p[operand] -
                    (shape[axis] - 1) * strides[operand * rank + axis] * (int64_t)bytes[operand];
        }
        if (!axis && (rank <= 1 || !indices[0])) return 0;
    }
}

#undef TS_ND_COMPARE
#undef TS_ND_DECLARATIONS
#undef TS_ND_INTEGER_BINARY
#undef TS_ND_FLOAT_BINARY
#undef TS_ND_ROW
#undef TS_ND_FLOAT_UNARY
#undef TS_ND_INTEGER_UNARY
#undef TS_ND_INVALID_F32
#undef TS_ND_INVALID_F64
#undef TS_ND_INVALID_TS_F32
#undef TS_ND_INVALID_TS_F64

#undef TS_ND_IDENTITY
