typedef struct {
    const void *data;
    size_t length, start, source_type, repeat, phase;
    int64_t stride;
} ts_kernel_input;

enum {
    TS_INPUT_OUTPUT, TS_INPUT_SUM, TS_INPUT_MINIMUM, TS_INPUT_MAXIMUM,
    TS_INPUT_ARGMIN, TS_INPUT_ARGMAX, TS_INPUT_ASUM, TS_INPUT_SUMSQ,
    TS_INPUT_MAXABS, TS_INPUT_SCALED_SUMSQ, TS_INPUT_MASK,
    TS_INPUT_COUNT, TS_INPUT_ANY, TS_INPUT_ALL
};

static size_t ts_input_size(size_t type) {
    static const size_t sizes[] = {4, 8, 1, 1, 2, 2, 4, 4, 8, 8};
    return type < 10 ? sizes[type] : 0;
}

static int ts_input_start(const ts_kernel_input *input, size_t row, size_t *start) {
    uint64_t magnitude = input->stride < 0
        ? (uint64_t)(-(input->stride + 1)) + 1 : (uint64_t)input->stride;
    if (magnitude && row > SIZE_MAX / magnitude) return -1;
    size_t shift = row * magnitude;
    if (input->stride < 0) {
        if (shift > input->start) return -1;
        *start = input->start - shift;
    } else {
        if (shift > SIZE_MAX - input->start) return -1;
        *start = input->start + shift;
    }
    return 0;
}

static int ts_validate_inputs(const ts_kernel_input *inputs, size_t input_count,
                              size_t rows, size_t n, size_t precision) {
    if (!inputs || !input_count || input_count > 247) return -1;
    for (size_t j = 0; j < input_count; ++j) {
        const ts_kernel_input *input = inputs + j;
        size_t bytes = ts_input_size(input->source_type), last = input->start;
        if (!bytes || (input->source_type < 2 && input->source_type != precision) ||
            !input->repeat || input->phase >= input->repeat ||
            input->length > PTRDIFF_MAX / bytes ||
            (input->data && (uintptr_t)input->data > UINTPTR_MAX - input->length * bytes) ||
            (rows && n && ts_input_start(input, rows - 1, &last))) return -1;
        size_t high = last > input->start ? last : input->start;
        if (high > input->length) return -1;
        if (rows && n) {
            if (!input->data || n - 1 > SIZE_MAX - input->phase) return -1;
            size_t span = (input->phase + n - 1) / input->repeat + 1;
            if (span > input->length - high) return -1;
        }
    }
    return 0;
}

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
#else
#define TS_LOAD_PACKED(suffix, name, out, input, n) 0
#endif

/* Volatile reads prevent Clang from vectorizing 64-bit-to-f32 through f64. */
#define TS_SOURCE_s64_f32 volatile int64_t
#define TS_SOURCE_u64_f32 volatile uint64_t
#define TS_SOURCE_s64_f64 int64_t
#define TS_SOURCE_u64_f64 uint64_t
#define TS_LOAD_WIDE_f32(name, out, input, n) 0
#define TS_LOAD_WIDE_f64(name, out, input, n) TS_LOAD_PACKED(f64, name, out, input, n)

#define TS_PREPARE_SOURCE(suffix, type, source, packed) \
    do { \
        const source *values = (const source *)input->data + start; \
        if (input->repeat == 1) { \
            values += base; \
            size_t i = packed; \
            for (; i < n; ++i) buffer[i] = (type)values[i]; \
        } else { \
            size_t position = input->phase + base; \
            size_t index = position / input->repeat; \
            size_t run = input->repeat - position % input->repeat; \
            for (size_t i = 0; i < n; ++index, run = input->repeat) { \
                size_t end = i + (run < n - i ? run : n - i); \
                type value = (type)values[index]; \
                TS_##suffix##_VECTOR pack = TS_##suffix##_SPLAT(value); \
                for (; end - i >= TS_##suffix##_WIDTH; i += TS_##suffix##_WIDTH) \
                    TS_##suffix##_STORE(buffer + i, pack); \
                for (; i < end; ++i) buffer[i] = value; \
            } \
        } \
    } while (0)

#define TS_PREPARE_INPUTS(suffix, type, precision, upper) \
static void ts_prepare_inputs_##suffix(const ts_kernel_input *inputs, size_t input_count, \
                                       size_t row, size_t base, size_t n, \
                                       type *buffers, const type **prepared) { \
    for (size_t j = 0; j < input_count; ++j) { \
        const ts_kernel_input *input = inputs + j; \
        size_t start; \
        ts_input_start(input, row, &start); \
        if (input->source_type == precision && input->repeat == 1) { \
            prepared[j] = (const type *)input->data + start + base; \
            continue; \
        } \
        type *buffer = buffers + j * TS_KERNEL_BLOCK; \
        prepared[j] = buffer; \
        switch (input->source_type) { \
        case 0: TS_PREPARE_SOURCE(upper, type, float, 0); break; \
        case 1: TS_PREPARE_SOURCE(upper, type, double, 0); break; \
        case 2: TS_PREPARE_SOURCE(upper, type, int8_t, TS_LOAD_PACKED(suffix, s8, buffer, values, n)); break; \
        case 3: TS_PREPARE_SOURCE(upper, type, uint8_t, TS_LOAD_PACKED(suffix, u8, buffer, values, n)); break; \
        case 4: TS_PREPARE_SOURCE(upper, type, int16_t, TS_LOAD_PACKED(suffix, s16, buffer, values, n)); break; \
        case 5: TS_PREPARE_SOURCE(upper, type, uint16_t, TS_LOAD_PACKED(suffix, u16, buffer, values, n)); break; \
        case 6: TS_PREPARE_SOURCE(upper, type, int32_t, TS_LOAD_PACKED(suffix, s32, buffer, values, n)); break; \
        case 7: TS_PREPARE_SOURCE(upper, type, uint32_t, TS_LOAD_PACKED(suffix, u32, buffer, values, n)); break; \
        case 8: TS_PREPARE_SOURCE(upper, type, TS_SOURCE_s64_##suffix, TS_LOAD_WIDE_##suffix(s64, buffer, values, n)); break; \
        case 9: TS_PREPARE_SOURCE(upper, type, TS_SOURCE_u64_##suffix, TS_LOAD_WIDE_##suffix(u64, buffer, values, n)); break; \
        } \
    } \
}

TS_PREPARE_INPUTS(f32, float, 0, F32)
TS_PREPARE_INPUTS(f64, double, 1, F64)

#define TS_KERNEL_INPUTS(suffix, type, precision) \
static int ts_run_inputs_##suffix(const uint8_t *code, size_t code_length, \
                                  const type *constants, const ts_kernel_input *inputs, \
                                  size_t input_count, size_t row, size_t n, size_t scratch_count, \
                                  unsigned mode, double scale, void *output, double *wide, \
                                  size_t *index, type *scratch, type *buffers) { \
    type total = 0, best = 0; \
    double accumulator = 0; \
    size_t best_index = 0, truths = 0; \
    int out_of_range = 0; \
    for (size_t base = 0; base < n;) { \
        size_t m = n - base < TS_KERNEL_BLOCK ? n - base : TS_KERNEL_BLOCK; \
        const type *prepared[247]; \
        type values[TS_KERNEL_BLOCK], block_sum = 0; \
        ts_prepare_inputs_##suffix(inputs, input_count, row, base, m, buffers, prepared); \
        int status = mode == TS_INPUT_SUM \
            ? ts_kernel_run_sum_##suffix(code, code_length, constants, prepared, \
                                          &block_sum, m, scratch_count, scratch) \
            : ts_kernel_run_elementwise_##suffix(code, code_length, constants, prepared, \
                                                  values, m, scratch_count, scratch); \
        if (status) return status; \
        if (mode == TS_INPUT_SUM) total += block_sum; \
        else if (mode == TS_INPUT_OUTPUT) memcpy((type *)output + base, values, m * sizeof(type)); \
        else if (mode >= TS_INPUT_MASK) { \
            for (size_t i = 0; i < m; ++i) { \
                int truth = values[i] != 0; \
                if (mode == TS_INPUT_MASK) ((uint8_t *)output)[base + i] = (uint8_t)truth; \
                else truths += truth; \
            } \
        } else if (mode <= TS_INPUT_ARGMAX) { \
            if (mode == TS_INPUT_MINIMUM || mode == TS_INPUT_ARGMIN) \
                TS_REDUCE_EXTREME(suffix, type, ts_reduce_less_##suffix) \
            else TS_REDUCE_EXTREME(suffix, type, ts_reduce_greater_##suffix) \
        } else if (mode == TS_INPUT_ASUM) { \
            type lane[TS_REDUCE_LANES] = {0}; \
            size_t i = 0; \
            for (; i + TS_REDUCE_LANES <= m; i += TS_REDUCE_LANES) \
                for (size_t k = 0; k < TS_REDUCE_LANES; ++k) \
                    lane[k] += ts_reduce_abs_##suffix(values[i + k]); \
            for (; i < m; ++i) lane[0] += ts_reduce_abs_##suffix(values[i]); \
            for (size_t k = 0; k < TS_REDUCE_LANES; ++k) total += lane[k]; \
        } else if (mode == TS_INPUT_MAXABS) { \
            for (size_t i = 0; i < m; ++i) { \
                double value = fabs((double)values[i]); \
                if (value > accumulator) accumulator = value; \
            } \
        } else { \
            double lane[TS_REDUCE_LANES] = {0}; \
            double divisor = mode == TS_INPUT_SUMSQ ? 1 : scale; \
            size_t i = 0; \
            for (; i < m; ++i) { \
                double value = (double)values[i] / divisor; \
                double clamped = fmin(fabs(value), TS_SQUARE_LIMIT); \
                if (!(fabs(value) <= TS_SQUARE_LIMIT)) out_of_range = 1; \
                lane[i < m - m % TS_REDUCE_LANES ? i % TS_REDUCE_LANES : 0] += clamped * clamped; \
            } \
            for (size_t k = 0; k < TS_REDUCE_LANES; ++k) accumulator += lane[k]; \
        } \
        base += m; \
    } \
    if (mode == TS_INPUT_SUM || mode == TS_INPUT_ASUM) *(type *)output = total; \
    else if (mode >= TS_INPUT_MINIMUM && mode <= TS_INPUT_ARGMAX) { \
        *(type *)output = best; *index = best_index; \
    } else if (mode >= TS_INPUT_SUMSQ && mode <= TS_INPUT_SCALED_SUMSQ) \
        *wide = out_of_range ? HUGE_VAL : accumulator; \
    else if (mode >= TS_INPUT_COUNT) \
        *index = mode == TS_INPUT_COUNT ? truths : mode == TS_INPUT_ANY ? truths != 0 : truths == n; \
    return 0; \
} \
int ts_kernel_inputs_##suffix(const uint8_t *code, size_t code_length, const type *constants, \
                              const ts_kernel_input *inputs, size_t input_count, void *output, \
                              size_t rows, size_t n, size_t scratch_count, unsigned mode, \
                              double scale, double *wide, size_t *index) { \
    if (mode > TS_INPUT_ALL || (rows != 1 && mode != TS_INPUT_SUM) || \
        (mode == TS_INPUT_SCALED_SUMSQ && !(scale > 0)) || \
        rows > PTRDIFF_MAX / sizeof(type) || n > PTRDIFF_MAX / sizeof(type) || \
        ts_validate_inputs(inputs, input_count, rows, n, precision)) return -1; \
    size_t output_count = mode == TS_INPUT_OUTPUT || mode == TS_INPUT_MASK ? n : rows; \
    size_t output_bytes = mode == TS_INPUT_MASK ? 1 : sizeof(type); \
    if (output && (uintptr_t)output > UINTPTR_MAX - output_count * output_bytes) return -1; \
    if ((rows && (n || (mode > TS_INPUT_OUTPUT && mode < TS_INPUT_SUMSQ)) && !output) || \
        (rows && mode >= TS_INPUT_SUMSQ && mode <= TS_INPUT_SCALED_SUMSQ && !wide) || \
        (rows && ((mode >= TS_INPUT_MINIMUM && mode <= TS_INPUT_ARGMAX) || mode >= TS_INPUT_COUNT) && !index)) return -1; \
    if (scratch_count > SIZE_MAX / TS_KERNEL_BLOCK / sizeof(type) || \
        input_count > SIZE_MAX / TS_KERNEL_BLOCK / sizeof(type)) return -1; \
    if (!rows || !n) { \
        for (size_t row = 0; row < rows; ++row) { \
            int status = ts_run_inputs_##suffix(code, code_length, constants, inputs, input_count, \
                row, 0, scratch_count, mode, scale, mode == TS_INPUT_SUM ? (type *)output + row : output, \
                wide, index, NULL, NULL); \
            if (status) return status; \
        } \
        return 0; \
    } \
    type *scratch = scratch_count ? malloc(scratch_count * TS_KERNEL_BLOCK * sizeof(type)) : NULL; \
    type *buffers = malloc(input_count * TS_KERNEL_BLOCK * sizeof(type)); \
    if ((scratch_count && !scratch) || !buffers) { free(scratch); free(buffers); return -1; } \
    int status = 0; \
    for (size_t row = 0; row < rows; ++row) { \
        status = ts_run_inputs_##suffix(code, code_length, constants, inputs, input_count, row, n, \
            scratch_count, mode, scale, mode == TS_INPUT_SUM ? (type *)output + row : output, \
            wide, index, scratch, buffers); \
        if (status) break; \
    } \
    free(scratch); free(buffers); \
    return status; \
}

TS_KERNEL_INPUTS(f32, float, 0)
TS_KERNEL_INPUTS(f64, double, 1)
