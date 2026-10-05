#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static float block_scale(const uint8_t *block) {
    uint16_t bits = (uint16_t)(block[0] | (uint16_t)block[1] << 8);
    unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    if (!exponent) return (bits & 32768 ? -1.0f : 1.0f) * ldexpf((float)fraction, -24);
    uint32_t wide = ((uint32_t)(bits & 32768) << 16) |
                    ((exponent == 31 ? 255u : exponent + 112u) << 23) | (fraction << 13);
    float scale;
    memcpy(&scale, &wide, sizeof(scale));
    return scale;
}

int q8_scalar(const uint8_t *blocks, const float *x, float *out, size_t rows, size_t width) {
    if (width % 32) return 1;
    for (size_t row = 0; row < rows; ++row) {
        float sum = 0;
        for (size_t col = 0; col < width; col += 32) {
            const uint8_t *block = blocks + (row * (width / 32) + col / 32) * 34;
            float scale = block_scale(block);
            for (size_t i = 0; i < 32; ++i) {
                int q = block[2 + i] < 128 ? block[2 + i] : (int)block[2 + i] - 256;
                sum += x[col + i] * ((float)q * scale);
            }
        }
        out[row] = sum;
    }
    return 0;
}

int q8_neon(const uint8_t *blocks, const float *x, float *out, size_t rows, size_t width) {
    if (width % 32) return 1;
    for (size_t row = 0; row < rows; ++row) {
        float32x4_t a = vdupq_n_f32(0), b = a, c = a, d = a;
        for (size_t col = 0; col < width; col += 32) {
            const uint8_t *block = blocks + (row * (width / 32) + col / 32) * 34;
            float32x4_t scale = vdupq_n_f32(block_scale(block));
            for (size_t i = 0; i < 32; i += 16) {
                int8x16_t q = vld1q_s8((const int8_t *)(block + 2 + i));
                int16x8_t lo = vmovl_s8(vget_low_s8(q)), hi = vmovl_s8(vget_high_s8(q));
                float32x4_t q0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
                float32x4_t q1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
                float32x4_t q2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
                float32x4_t q3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
                a = vfmaq_f32(a, vld1q_f32(x + col + i), vmulq_f32(q0, scale));
                b = vfmaq_f32(b, vld1q_f32(x + col + i + 4), vmulq_f32(q1, scale));
                c = vfmaq_f32(c, vld1q_f32(x + col + i + 8), vmulq_f32(q2, scale));
                d = vfmaq_f32(d, vld1q_f32(x + col + i + 12), vmulq_f32(q3, scale));
            }
        }
        out[row] = vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
    }
    return 0;
}
