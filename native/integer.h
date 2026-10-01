#if !defined(TS_SCALAR) && (defined(__aarch64__) || defined(_M_ARM64))
#define TS_INTEGER_VECTOR(suffix, type, bits, vector, load, store, add, sub, mul) \
static size_t ts_vector_##suffix(unsigned op, type *out, const type *a, const type *b, size_t n) { \
    size_t i = 0; \
    if (op != TS_OP_ADD && op != TS_OP_SUBTRACT && (op != TS_OP_MULTIPLY || bits == 64)) return 0; \
    for (; i + 128 / bits <= n; i += 128 / bits) { \
        vector x = load(a + i), y = load(b + i), z; \
        if (op == TS_OP_ADD) z = add(x, y); \
        else if (op == TS_OP_SUBTRACT) z = sub(x, y); \
        else z = mul(x, y); \
        store(out + i, z); \
    } \
    return i; \
}
TS_INTEGER_VECTOR(u8, uint8_t, 8, uint8x16_t, vld1q_u8, vst1q_u8, vaddq_u8, vsubq_u8, vmulq_u8)
TS_INTEGER_VECTOR(u16, uint16_t, 16, uint16x8_t, vld1q_u16, vst1q_u16, vaddq_u16, vsubq_u16, vmulq_u16)
TS_INTEGER_VECTOR(u32, uint32_t, 32, uint32x4_t, vld1q_u32, vst1q_u32, vaddq_u32, vsubq_u32, vmulq_u32)
static size_t ts_vector_u64(unsigned op, uint64_t *out, const uint64_t *a,
                            const uint64_t *b, size_t n) {
    if (op != TS_OP_ADD && op != TS_OP_SUBTRACT) return 0;
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64x2_t x = vld1q_u64(a + i), y = vld1q_u64(b + i);
        vst1q_u64(out + i, op == TS_OP_ADD ? vaddq_u64(x, y) : vsubq_u64(x, y));
    }
    return i;
}
#elif !defined(TS_SCALAR) && (defined(__x86_64__) || defined(_M_X64))
#define TS_INTEGER_VECTOR(suffix, type, bits, add, sub) \
static size_t ts_vector_##suffix(unsigned op, type *out, const type *a, const type *b, size_t n) { \
    size_t i = 0; \
    if (op != TS_OP_ADD && op != TS_OP_SUBTRACT && (op != TS_OP_MULTIPLY || bits != 16)) return 0; \
    for (; i + 128 / bits <= n; i += 128 / bits) { \
        __m128i x = _mm_loadu_si128((const __m128i *)(a + i)); \
        __m128i y = _mm_loadu_si128((const __m128i *)(b + i)); \
        __m128i z = op == TS_OP_ADD ? add(x, y) : op == TS_OP_SUBTRACT ? sub(x, y) : _mm_mullo_epi16(x, y); \
        _mm_storeu_si128((__m128i *)(out + i), z); \
    } \
    return i; \
}
TS_INTEGER_VECTOR(u8, uint8_t, 8, _mm_add_epi8, _mm_sub_epi8)
TS_INTEGER_VECTOR(u16, uint16_t, 16, _mm_add_epi16, _mm_sub_epi16)
TS_INTEGER_VECTOR(u32, uint32_t, 32, _mm_add_epi32, _mm_sub_epi32)
TS_INTEGER_VECTOR(u64, uint64_t, 64, _mm_add_epi64, _mm_sub_epi64)
#else
#define TS_INTEGER_VECTOR(suffix, type) \
static size_t ts_vector_##suffix(unsigned op, type *out, const type *a, const type *b, size_t n) { \
    (void)op; (void)out; (void)a; (void)b; (void)n; return 0; \
}
TS_INTEGER_VECTOR(u8, uint8_t)
TS_INTEGER_VECTOR(u16, uint16_t)
TS_INTEGER_VECTOR(u32, uint32_t)
TS_INTEGER_VECTOR(u64, uint64_t)
#endif
#undef TS_INTEGER_VECTOR

static int ts_mul_add_constant_pattern(const uint8_t *code, size_t length) {
    return length == 12 && code[0] == TS_OP_MULTIPLY &&
           code[1] < TS_KERNEL_REGISTERS &&
           code[2] >= TS_KERNEL_REGISTERS && code[3] >= TS_KERNEL_REGISTERS &&
           code[4] == TS_OP_CONSTANT && code[5] < TS_KERNEL_REGISTERS &&
           code[5] != code[1] && code[8] == TS_OP_ADD &&
           code[9] == TS_KERNEL_OUTPUT &&
           ((code[10] == code[1] && code[11] == code[5]) ||
            (code[10] == code[5] && code[11] == code[1]));
}

static int ts_partial_overlap(const void *out, const void *input, size_t bytes) {
    uintptr_t a = (uintptr_t)out, b = (uintptr_t)input;
    return a != b && (a > b ? a - b : b - a) < bytes;
}

#define TS_INTEGER(suffix, type, bits, signedp, vector) \
static type ts_value_##suffix(unsigned op, type a, type b) { \
    const uint64_t sign = UINT64_C(1) << (bits - 1); \
    uint64_t x = a, y = b; \
    int negative_a = signedp && (x & sign), negative_b = signedp && (y & sign); \
    switch (op) { \
    case TS_OP_ADD: return (type)(x + y); \
    case TS_OP_SUBTRACT: return (type)(x - y); \
    case TS_OP_MULTIPLY: return (type)(x * y); \
    case TS_OP_NEGATE: return (type)(UINT64_C(0) - x); \
    case TS_OP_ABS: return negative_a ? (type)(UINT64_C(0) - x) : a; \
    case TS_OP_MIN: return negative_a != negative_b ? (negative_a ? a : b) : (a < b ? a : b); \
    case TS_OP_MAX: return negative_a != negative_b ? (negative_a ? b : a) : (a > b ? a : b); \
    case TS_OP_DIVIDE: { \
        uint64_t magnitude_a = negative_a ? (type)(UINT64_C(0) - x) : x; \
        uint64_t magnitude_b = negative_b ? (type)(UINT64_C(0) - y) : y; \
        uint64_t quotient = magnitude_a / magnitude_b; \
        return (type)(negative_a != negative_b ? UINT64_C(0) - quotient : quotient); \
    } \
    default: return a; \
    } \
} \
static int ts_binary_##suffix(unsigned op, type *out, const type *a, const type *b, size_t n) { \
    size_t i = vector(op, out, a, b, n); \
    for (; i < n; ++i) { \
        if (op == TS_OP_DIVIDE && !b[i]) return -3; \
        out[i] = ts_value_##suffix(op, a[i], b[i]); \
    } \
    return 0; \
} \
int ts_bulk_binary_##suffix(unsigned op, type *out, const type *left, const type *right, \
                            type left_scalar, type right_scalar, size_t n) { \
    for (size_t i = 0; i < n; ++i) { \
        type a = left ? left[i] : left_scalar, b = right ? right[i] : right_scalar; \
        if (op == TS_OP_DIVIDE && !b) return -3; \
        out[i] = ts_value_##suffix(op, a, b); \
    } \
    return 0; \
} \
int ts_bulk_axpy_##suffix(type *y, type a, const type *x, size_t n) { \
    for (size_t i = 0; i < n; ++i) \
        y[i] = ts_value_##suffix(TS_OP_ADD, \
                ts_value_##suffix(TS_OP_MULTIPLY, a, x[i]), y[i]); \
    return 0; \
} \
int ts_add_##suffix(type *out, const type *a, const type *b, size_t n) { \
    return ts_binary_##suffix(TS_OP_ADD, out, a, b, n); \
} \
int ts_subtract_##suffix(type *out, const type *a, const type *b, size_t n) { \
    return ts_binary_##suffix(TS_OP_SUBTRACT, out, a, b, n); \
} \
int ts_multiply_##suffix(type *out, const type *a, const type *b, size_t n) { \
    return ts_binary_##suffix(TS_OP_MULTIPLY, out, a, b, n); \
} \
int ts_divide_##suffix(type *out, const type *a, const type *b, size_t n) { \
    return ts_binary_##suffix(TS_OP_DIVIDE, out, a, b, n); \
} \
int ts_sum_##suffix(const type *a, type *out, size_t n) { \
    enum { width = 128 / bits }; \
    type lanes[width] = {0}, result = 0; \
    size_t i = 0; \
    for (; i + width <= n; i += width) ts_binary_##suffix(TS_OP_ADD, lanes, lanes, a + i, width); \
    for (size_t j = 0; j < width; ++j) result = (type)((uint64_t)result + lanes[j]); \
    for (; i < n; ++i) result = (type)((uint64_t)result + a[i]); \
    *out = result; return 0; \
} \
int ts_dot_##suffix(const type *a, const type *b, type *out, size_t n) { \
    enum { width = 128 / bits }; \
    type products[width], lanes[width] = {0}, result = 0; \
    size_t i = 0; \
    for (; i + width <= n; i += width) { \
        ts_binary_##suffix(TS_OP_MULTIPLY, products, a + i, b + i, width); \
        ts_binary_##suffix(TS_OP_ADD, lanes, lanes, products, width); \
    } \
    for (size_t j = 0; j < width; ++j) result = (type)((uint64_t)result + lanes[j]); \
    for (; i < n; ++i) result = (type)((uint64_t)result + (uint64_t)a[i] * b[i]); \
    *out = result; return 0; \
} \
static int ts_integer_kernel_##suffix(const uint8_t *code, size_t code_length, \
                  const type *constants, const type *const *inputs, type *out, size_t n, \
                  size_t scratch_count, int sum, type *external_scratch) { \
    if (!n) { if (sum) *out = 0; return 0; } \
    if (bits == 64 && !sum && !scratch_count && n <= SIZE_MAX / sizeof(type) && \
        ts_mul_add_constant_pattern(code, code_length)) { \
        const type *a = inputs[code[2] - TS_KERNEL_REGISTERS]; \
        const type *b = inputs[code[3] - TS_KERNEL_REGISTERS]; \
        size_t bytes = n * sizeof(type); \
        if (!ts_partial_overlap(out, a, bytes) && !ts_partial_overlap(out, b, bytes)) { \
            type constant = constants[code[6]]; \
            for (size_t i = 0; i < n; ++i) \
                out[i] = (type)((uint64_t)a[i] * b[i] + constant); \
            return 0; \
        } \
    } \
    if (scratch_count > SIZE_MAX / TS_KERNEL_BLOCK / sizeof(type)) return -1; \
    type *scratch = external_scratch ? external_scratch \
        : scratch_count ? malloc(scratch_count * TS_KERNEL_BLOCK * sizeof(type)) : NULL; \
    if (scratch_count && !scratch) return -1; \
    type registers[TS_KERNEL_REGISTERS][TS_KERNEL_BLOCK], reduction[TS_KERNEL_BLOCK]; \
    type result = 0; \
    int status = 0; \
    for (size_t base = 0; base < n; base += TS_KERNEL_BLOCK) { \
        size_t m = n - base < TS_KERNEL_BLOCK ? n - base : TS_KERNEL_BLOCK; \
        for (size_t pc = 0; pc < code_length; pc += 4) { \
            unsigned op = code[pc], dst = code[pc + 1], a = code[pc + 2], b = code[pc + 3]; \
            if (op == TS_OP_SPILL || op == TS_OP_RELOAD) { \
                type *slot = scratch + ((size_t)a | ((size_t)b << 8)) * TS_KERNEL_BLOCK; \
                if (op == TS_OP_SPILL) memcpy(slot, registers[dst], m * sizeof(type)); \
                else memcpy(registers[dst], slot, m * sizeof(type)); \
                continue; \
            } \
            type *destination = dst == TS_KERNEL_OUTPUT ? (sum ? reduction : out + base) : registers[dst]; \
            const type *left = NULL, *right = NULL; \
            if (op != TS_OP_CONSTANT) left = a < TS_KERNEL_REGISTERS ? registers[a] : inputs[a - TS_KERNEL_REGISTERS] + base; \
            if ((op >= TS_OP_ADD && op <= TS_OP_DIVIDE) || op == TS_OP_MIN || op == TS_OP_MAX || \
                (op >= TS_OP_EQ && op <= TS_OP_SELECT)) \
                right = b < TS_KERNEL_REGISTERS ? registers[b] : inputs[b - TS_KERNEL_REGISTERS] + base; \
            if (op == TS_OP_CONSTANT) { \
                for (size_t j = 0; j < m; ++j) destination[j] = constants[a]; \
            } else if (op == TS_OP_COPY) { \
                if (destination != left) memcpy(destination, left, m * sizeof(type)); \
            } else if (op == TS_OP_NEGATE || op == TS_OP_ABS) { \
                for (size_t j = 0; j < m; ++j) destination[j] = ts_value_##suffix(op, left[j], 0); \
            } else if (op >= TS_OP_EQ && op <= TS_OP_GE) { \
                for (size_t j = 0; j < m; ++j) { \
                    type x = left[j], y = right[j]; \
                    int less = x != y && ts_value_##suffix(TS_OP_MIN, x, y) == x; \
                    int greater = x != y && !less; \
                    destination[j] = op == TS_OP_EQ ? x == y : op == TS_OP_NE ? x != y : \
                                     op == TS_OP_LT ? less : op == TS_OP_LE ? !greater : \
                                     op == TS_OP_GT ? greater : !less; \
                } \
            } else if (op == TS_OP_SELECT) { \
                for (size_t j = 0; j < m; ++j) \
                    destination[j] = left[j] ? right[j] : destination[j]; \
            } else { \
                status = ts_binary_##suffix(op, destination, left, right, m); \
                if (status) goto done; \
            } \
            if (sum && dst == TS_KERNEL_OUTPUT) { \
                type block; ts_sum_##suffix(destination, &block, m); \
                result = (type)((uint64_t)result + block); \
            } \
        } \
    } \
    if (sum) *out = result; \
    done: if (!external_scratch) free(scratch); return status; \
} \
int ts_kernel_##suffix(const uint8_t *code, size_t code_length, const type *constants, \
                      const type *const *inputs, type *out, size_t n, size_t scratch_count) { \
    return ts_integer_kernel_##suffix(code, code_length, constants, inputs, out, n, scratch_count, 0, NULL); \
} \
int ts_kernel_with_scratch_##suffix(const uint8_t *code, size_t code_length, const type *constants, \
                                   const type *const *inputs, type *out, size_t n, \
                                   size_t scratch_count, type *scratch, size_t capacity) { \
    if (n && scratch_count && (!scratch || capacity < scratch_count)) return -1; \
    return ts_integer_kernel_##suffix(code, code_length, constants, inputs, out, n, scratch_count, 0, scratch); \
} \
int ts_kernel_sum_##suffix(const uint8_t *code, size_t code_length, const type *constants, \
                          const type *const *inputs, type *out, size_t n, size_t scratch_count) { \
    return ts_integer_kernel_##suffix(code, code_length, constants, inputs, out, n, scratch_count, 1, NULL); \
}

TS_INTEGER(s8, uint8_t, 8, 1, ts_vector_u8)
TS_INTEGER(u8, uint8_t, 8, 0, ts_vector_u8)
TS_INTEGER(s16, uint16_t, 16, 1, ts_vector_u16)
TS_INTEGER(u16, uint16_t, 16, 0, ts_vector_u16)
TS_INTEGER(s32, uint32_t, 32, 1, ts_vector_u32)
TS_INTEGER(u32, uint32_t, 32, 0, ts_vector_u32)
TS_INTEGER(s64, uint64_t, 64, 1, ts_vector_u64)
TS_INTEGER(u64, uint64_t, 64, 0, ts_vector_u64)
#undef TS_INTEGER
