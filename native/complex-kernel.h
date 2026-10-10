#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
#define TS_COMPLEX_LOAD(suffix, type, vector, load, store) \
static ts_##suffix##_vector ts_##suffix##_load(const type *p) { \
    vector a = load(p); \
    ts_##suffix##_vector z = {a.val[0], a.val[1]}; return z; \
} \
static void ts_##suffix##_store(type *p, ts_##suffix##_vector z) { \
    vector a = {{z.real, z.imag}}; store(p, a); \
}
#define TS_C32_EQ(a, b) vreinterpretq_f32_u32(vceqq_f32(a, b))
#define TS_C64_EQ(a, b) vreinterpretq_f64_u64(vceqq_f64(a, b))
#define TS_C32_AND(a, b) vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)))
#define TS_C64_AND(a, b) vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)))
#define TS_C32_BLEND(mask, a, b) vbslq_f32(vreinterpretq_u32_f32(mask), a, b)
#define TS_C64_BLEND(mask, a, b) vbslq_f64(vreinterpretq_u64_f64(mask), a, b)
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
#define TS_COMPLEX_LOAD(suffix, type, load, store, real_shuffle, imag_shuffle, low, high) \
static ts_##suffix##_vector ts_##suffix##_load(const type *p) { \
    ts_##suffix##_vector z = {real_shuffle(load(p), load(p + 2 * TS_##suffix##_WIDTH)), \
                              imag_shuffle(load(p), load(p + 2 * TS_##suffix##_WIDTH))}; return z; \
} \
static void ts_##suffix##_store(type *p, ts_##suffix##_vector z) { \
    store(p, low(z.real, z.imag)); store(p + 2 * TS_##suffix##_WIDTH, high(z.real, z.imag)); \
}
#define TS_C32_EQ(a, b) _mm_cmpeq_ps(a, b)
#define TS_C64_EQ(a, b) _mm_cmpeq_pd(a, b)
#define TS_C32_AND(a, b) _mm_and_ps(a, b)
#define TS_C64_AND(a, b) _mm_and_pd(a, b)
#define TS_C32_BLEND(mask, a, b) _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b))
#define TS_C64_BLEND(mask, a, b) _mm_or_pd(_mm_and_pd(mask, a), _mm_andnot_pd(mask, b))
#else
#define TS_COMPLEX_LOAD(suffix, type) \
static ts_##suffix##_vector ts_##suffix##_load(const type *p) { \
    ts_##suffix##_vector z = {p[0], p[1]}; return z; \
} \
static void ts_##suffix##_store(type *p, ts_##suffix##_vector z) { p[0] = z.real; p[1] = z.imag; }
#define TS_C32_EQ(a, b) ((a) == (b))
#define TS_C64_EQ(a, b) ((a) == (b))
#define TS_C32_AND(a, b) ((a) && (b))
#define TS_C64_AND(a, b) ((a) && (b))
#define TS_C32_BLEND(mask, a, b) ((mask) ? (a) : (b))
#define TS_C64_BLEND(mask, a, b) ((mask) ? (a) : (b))
#endif

typedef struct { TS_F32_VECTOR real, imag; } ts_c32_vector;
typedef struct { TS_F64_VECTOR real, imag; } ts_c64_vector;

#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
TS_COMPLEX_LOAD(c32, float, float32x4x2_t, vld2q_f32, vst2q_f32)
TS_COMPLEX_LOAD(c64, double, float64x2x2_t, vld2q_f64, vst2q_f64)
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
#define TS_c32_WIDTH 2
#define TS_c64_WIDTH 1
#define TS_C32_REAL(a, b) _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0))
#define TS_C32_IMAG(a, b) _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1))
TS_COMPLEX_LOAD(c32, float, _mm_loadu_ps, _mm_storeu_ps, TS_C32_REAL, TS_C32_IMAG, _mm_unpacklo_ps, _mm_unpackhi_ps)
TS_COMPLEX_LOAD(c64, double, _mm_loadu_pd, _mm_storeu_pd, _mm_unpacklo_pd, _mm_unpackhi_pd, _mm_unpacklo_pd, _mm_unpackhi_pd)
#else
TS_COMPLEX_LOAD(c32, float)
TS_COMPLEX_LOAD(c64, double)
#endif

#define TS_COMPLEX_APPLY(suffix, type, width, zero, splat, add, sub, mul, eq, and, blend, absolute, root, scale) \
static int ts_complex_apply_##suffix(unsigned op, type *destination, \
                                     const type *left, const type *right, size_t m) { \
    size_t i = 0; \
    if (op == TS_OP_COPY) { \
        if (destination != left) memcpy(destination, left, 2 * m * sizeof(type)); \
        return 0; \
    } \
    if (op == TS_OP_ADD || op == TS_OP_SUBTRACT || op == TS_OP_MULTIPLY || \
        op == TS_OP_EQ || op == TS_OP_NE || op == TS_OP_SELECT) { \
        for (; i + width <= m; i += width) { \
            ts_##suffix##_vector a = ts_##suffix##_load(left + 2 * i); \
            ts_##suffix##_vector b = ts_##suffix##_load(right + 2 * i), z; \
            if (op == TS_OP_ADD) { z.real = add(a.real, b.real); z.imag = add(a.imag, b.imag); } \
            else if (op == TS_OP_SUBTRACT) { z.real = sub(a.real, b.real); z.imag = sub(a.imag, b.imag); } \
            else if (op == TS_OP_MULTIPLY) { \
                z.real = sub(mul(a.real, b.real), mul(a.imag, b.imag)); \
                z.imag = add(mul(a.real, b.imag), mul(a.imag, b.real)); \
            } else if (op == TS_OP_SELECT) { \
                ts_##suffix##_vector c = ts_##suffix##_load(destination + 2 * i); \
                z.real = blend(eq(a.real, zero()), c.real, b.real); \
                z.imag = blend(eq(a.real, zero()), c.imag, b.imag); \
            } else { \
                z.real = op == TS_OP_EQ \
                    ? blend(and(eq(a.real, b.real), eq(a.imag, b.imag)), splat(1), zero()) \
                    : blend(and(eq(a.real, b.real), eq(a.imag, b.imag)), zero(), splat(1)); \
                z.imag = zero(); \
            } \
            ts_##suffix##_store(destination + 2 * i, z); \
        } \
    } \
    for (; i < m; ++i) { \
        type ar = left[2 * i], ai = left[2 * i + 1]; \
        type br = right ? right[2 * i] : 0, bi = right ? right[2 * i + 1] : 0, re, im; \
        switch (op) { \
        case TS_OP_ADD: re = ar + br; im = ai + bi; break; \
        case TS_OP_SUBTRACT: re = ar - br; im = ai - bi; break; \
        case TS_OP_MULTIPLY: re = ar * br - ai * bi; im = ar * bi + ai * br; break; \
        case TS_OP_NEGATE: re = -ar; im = -ai; break; \
        case TS_OP_EQ: re = ar == br && ai == bi; im = 0; break; \
        case TS_OP_NE: re = ar != br || ai != bi; im = 0; break; \
        case TS_OP_SELECT: \
            if (ar == 0) continue; \
            re = br; im = bi; break; \
        case TS_OP_DIVIDE: { \
            if (br == 0 && bi == 0) return -3; \
            int exponent; frexp(fmax(absolute(br), absolute(bi)), &exponent); \
            double cr = scale(br, -exponent), ci = scale(bi, -exponent); \
            double xr = scale(ar, -exponent), xi = scale(ai, -exponent); \
            double denominator = cr * cr + ci * ci; \
            re = (type)((xr * cr + xi * ci) / denominator); \
            im = (type)((xi * cr - xr * ci) / denominator); break; \
        } \
        case TS_OP_SQRT: { \
            double x = absolute(ar), y = absolute(ai), largest = fmax(x, y); \
            if (largest == 0) { re = 0; im = ai; break; } \
            double r = x / largest, q = y / largest; \
            double t = root(largest) * root((hypot(r, q) + r) * 0.5); \
            if (ar >= 0) { re = (type)t; im = (type)(ai / (2 * t)); } \
            else { re = (type)(y / (2 * t)); im = (type)copysign(t, ai); } \
            break; \
        } \
        default: return -4; \
        } \
        destination[2 * i] = re; destination[2 * i + 1] = im; \
    } \
    return 0; \
} \
static int ts_complex_block_##suffix(const uint8_t *code, size_t code_length, \
                                     const type *constants, const type *const *inputs, \
                                     type *out, size_t n, size_t scratch_count, type *scratch) { \
    TS_KERNEL_EXECUTE(type, 2, 0, 0, ts_complex_apply_##suffix(op, destination, left, right, m)); \
}

TS_COMPLEX_APPLY(c32, float, TS_F32_WIDTH, TS_F32_ZERO, TS_F32_SPLAT, TS_F32_ADD, TS_F32_SUB, TS_F32_MUL,
                 TS_C32_EQ, TS_C32_AND, TS_C32_BLEND, fabs, sqrt, scalbn)
TS_COMPLEX_APPLY(c64, double, TS_F64_WIDTH, TS_F64_ZERO, TS_F64_SPLAT, TS_F64_ADD, TS_F64_SUB, TS_F64_MUL,
                 TS_C64_EQ, TS_C64_AND, TS_C64_BLEND, fabs, sqrt, scalbn)

enum { TS_COMPLEX_VALUES, TS_COMPLEX_MASK, TS_COMPLEX_COUNT, TS_COMPLEX_ANY,
       TS_COMPLEX_ALL, TS_COMPLEX_SUM, TS_COMPLEX_ASUM, TS_COMPLEX_NRM2 };

static int ts_complex_program_valid(const uint8_t *code, size_t length, const void *constants,
                                    size_t input_count, size_t scratch_count) {
    for (size_t pc = 0; pc < length; pc += 4) {
        unsigned op = code[pc], dst = code[pc + 1], a = code[pc + 2], b = code[pc + 3];
        if (dst >= TS_KERNEL_REGISTERS && dst != TS_KERNEL_OUTPUT) return 0;
        if (op == TS_OP_SPILL || op == TS_OP_RELOAD) {
            if (dst == TS_KERNEL_OUTPUT || ((size_t)a | ((size_t)b << 8)) >= scratch_count) return 0;
            continue;
        }
        if (op == TS_OP_CONSTANT) { if (!constants) return 0; continue; }
        if (op != TS_OP_COPY && op != TS_OP_ADD && op != TS_OP_SUBTRACT && op != TS_OP_MULTIPLY &&
            op != TS_OP_DIVIDE && op != TS_OP_NEGATE && op != TS_OP_SQRT && op != TS_OP_EQ &&
            op != TS_OP_NE && op != TS_OP_SELECT) return 0;
        if (a >= TS_KERNEL_REGISTERS && a - TS_KERNEL_REGISTERS >= input_count) return 0;
        if ((op >= TS_OP_ADD && op <= TS_OP_DIVIDE) || op == TS_OP_EQ || op == TS_OP_NE || op == TS_OP_SELECT)
            if (b >= TS_KERNEL_REGISTERS && b - TS_KERNEL_REGISTERS >= input_count) return 0;
    }
    return 1;
}

/* TODO: complex reductions materialize blocks; fuse the final opcode into packed reducers (#48). */
#define TS_COMPLEX_KERNEL(suffix, type, magnitude) \
static int ts_complex_execute_##suffix(const uint8_t *code, size_t code_length, \
                 const type *constants, const type *const *inputs, size_t input_count, \
                 type *out, uint8_t *mask, size_t *result, double *norm, \
                 size_t n, size_t scratch_count, unsigned mode) { \
    if (input_count > 248 || mode > TS_COMPLEX_NRM2 || code_length % 4 || \
        n > PTRDIFF_MAX / sizeof(type) / 2 || \
        (code_length && !code) || (input_count && !inputs)) return -4; \
    if (n && (!code_length || (mode == TS_COMPLEX_VALUES && !out) || (mode == TS_COMPLEX_MASK && !mask))) return -4; \
    if (!ts_complex_program_valid(code, code_length, constants, input_count, scratch_count)) return -4; \
    if (n) for (size_t j = 0; j < input_count; ++j) if (!inputs[j]) return -4; \
    if ((mode >= TS_COMPLEX_COUNT && mode <= TS_COMPLEX_ALL && !result) || \
        ((mode == TS_COMPLEX_SUM || mode == TS_COMPLEX_ASUM) && !out) || \
        (mode == TS_COMPLEX_NRM2 && !norm)) return -4; \
    if (!n) { \
        if (mode >= TS_COMPLEX_COUNT && mode <= TS_COMPLEX_ALL) *result = mode == TS_COMPLEX_ALL; \
        if (mode == TS_COMPLEX_SUM || mode == TS_COMPLEX_ASUM) { out[0] = 0; out[1] = 0; } \
        if (mode == TS_COMPLEX_NRM2) *norm = 0; \
        return 0; \
    } \
    if (scratch_count > SIZE_MAX / TS_KERNEL_BLOCK / 2 / sizeof(type)) return -1; \
    type *scratch = scratch_count ? malloc(scratch_count * TS_KERNEL_BLOCK * 2 * sizeof(type)) : NULL; \
    if (scratch_count && !scratch) return -1; \
    type total[2] = {0}; size_t truth = 0; double scale = 0, squares = 1; int status = 0; \
    for (size_t base = 0; base < n; base += TS_KERNEL_BLOCK) { \
        size_t m = n - base < TS_KERNEL_BLOCK ? n - base : TS_KERNEL_BLOCK; \
        const type *shifted[248]; type values[TS_KERNEL_BLOCK * 2]; \
        for (size_t j = 0; j < input_count; ++j) shifted[j] = inputs[j] + 2 * base; \
        status = ts_complex_block_##suffix(code, code_length, constants, shifted, values, m, scratch_count, scratch); \
        if (status) break; \
        for (size_t j = 0; j < m; ++j) { \
            type re = values[2 * j], im = values[2 * j + 1]; \
            switch (mode) { \
            case TS_COMPLEX_VALUES: out[2 * (base + j)] = re; out[2 * (base + j) + 1] = im; break; \
            case TS_COMPLEX_MASK: mask[base + j] = re != 0; break; \
            case TS_COMPLEX_COUNT: case TS_COMPLEX_ANY: case TS_COMPLEX_ALL: truth += re != 0; break; \
            case TS_COMPLEX_SUM: total[0] += re; total[1] += im; break; \
            case TS_COMPLEX_ASUM: total[0] += magnitude(re, im); break; \
            case TS_COMPLEX_NRM2: \
                for (size_t k = 0; k < 2; ++k) { \
                    double part = fabs((double)values[2 * j + k]); \
                    if (!part) continue; \
                    if (scale < part) { double r = scale / part; squares = 1 + squares * r * r; scale = part; } \
                    else { double r = part / scale; squares += r * r; } \
                } \
                break; \
            } \
        } \
    } \
    free(scratch); \
    if (status) return status; \
    if (mode >= TS_COMPLEX_COUNT && mode <= TS_COMPLEX_ALL) \
        *result = mode == TS_COMPLEX_COUNT ? truth : mode == TS_COMPLEX_ANY ? truth != 0 : truth == n; \
    if (mode == TS_COMPLEX_SUM || mode == TS_COMPLEX_ASUM) { out[0] = total[0]; out[1] = total[1]; } \
    if (mode == TS_COMPLEX_NRM2) *norm = scale * sqrt(squares); \
    return 0; \
} \
int ts_kernel_##suffix(const uint8_t *code, size_t code_length, const type *constants, \
                       const type *const *inputs, type *out, size_t n, size_t scratch_count) { \
    size_t input_count = 0; \
    if (code_length && !code) return -4; \
    for (size_t pc = 0; pc + 3 < code_length; pc += 4) { \
        unsigned op = code[pc]; \
        if (op == TS_OP_CONSTANT || op == TS_OP_SPILL || op == TS_OP_RELOAD) continue; \
        if (code[pc + 2] >= TS_KERNEL_REGISTERS) \
            if (input_count < code[pc + 2] - TS_KERNEL_REGISTERS + 1u) \
                input_count = code[pc + 2] - TS_KERNEL_REGISTERS + 1u; \
        if ((op >= TS_OP_ADD && op <= TS_OP_DIVIDE) || op == TS_OP_EQ || op == TS_OP_NE || op == TS_OP_SELECT) \
            if (code[pc + 3] >= TS_KERNEL_REGISTERS) \
                if (input_count < code[pc + 3] - TS_KERNEL_REGISTERS + 1u) \
                    input_count = code[pc + 3] - TS_KERNEL_REGISTERS + 1u; \
    } \
    return ts_complex_execute_##suffix(code, code_length, constants, inputs, input_count, out, NULL, NULL, NULL, \
                                       n, scratch_count, TS_COMPLEX_VALUES); \
} \
int ts_kernel_mask_##suffix(const uint8_t *code, size_t code_length, const type *constants, \
                  const type *const *inputs, size_t input_count, uint8_t *mask, \
                  size_t n, size_t scratch_count, unsigned reduction, size_t *result) { \
    if (reduction > 3) return -4; \
    return ts_complex_execute_##suffix(code, code_length, constants, inputs, input_count, NULL, mask, result, NULL, \
                                       n, scratch_count, TS_COMPLEX_MASK + reduction); \
} \
int ts_kernel_reduction_##suffix(const uint8_t *code, size_t code_length, const type *constants, \
                   const type *const *inputs, size_t input_count, size_t n, size_t scratch_count, \
                   unsigned reducer, double scale, type *value, double *wide, size_t *index) { \
    (void)scale; (void)index; \
    unsigned mode = reducer == 0 ? TS_COMPLEX_SUM : reducer == TS_REDUCE_ASUM ? TS_COMPLEX_ASUM \
                                 : reducer == TS_REDUCE_SUMSQ ? TS_COMPLEX_NRM2 : TS_COMPLEX_NRM2 + 1; \
    return ts_complex_execute_##suffix(code, code_length, constants, inputs, input_count, value, NULL, NULL, wide, \
                                       n, scratch_count, mode); \
}

TS_COMPLEX_KERNEL(c32, float, hypotf)
TS_COMPLEX_KERNEL(c64, double, hypot)
