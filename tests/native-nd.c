#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

int ts_nd_execute(unsigned, unsigned, unsigned, unsigned, unsigned, int, size_t,
                  const int64_t *, const int64_t *, void *const *, int64_t *);

int main(void) {
    float left[100], right[100], out[100];
    for (size_t i = 0; i < 100; ++i) { left[i] = (float)i; right[i] = 1; out[i] = -1; }
    void *p[] = {out + 30, left, right, NULL};
    int64_t shape[] = {3, 5}, strides[] = {-10, 2, 1, 0, 0, 1, 0, 0}, indices[4];
    assert(ts_nd_execute(0, 0, 0, 0, 0, 0, 2, shape, strides, p, indices) == 0);
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 5; ++col) assert(out[30 - 10 * row + 2 * col] == row + 1);
    assert(out[1] == -1);

    int64_t dimensions[] = {2, 2, 2, 3};
    int64_t steps[] = {24, 12, 6, 2, 24, 12, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0};
    p[0] = out; p[1] = left; p[2] = right;
    assert(ts_nd_execute(0, 0, 0, 0, 0, 0, 4, dimensions, steps, p, indices) == 0);
    for (int i = 0; i < 24; ++i) assert(out[2 * i] == left[2 * i] + 1);

    uint8_t mask[100] = {0};
    p[0] = mask; p[1] = left; p[2] = right;
    int64_t contiguous[] = {5, 1, 5, 1, 0, 0, 0, 0};
    assert(ts_nd_execute(11, 0, 0, 0, 4, 0, 2, shape, contiguous, p, indices) == 0);
    for (int i = 0; i < 15; ++i) assert(mask[i] == (left[i] > 1));
    p[0] = out; p[1] = mask; p[2] = left; p[3] = right;
    int64_t selection[] = {5, 1, 5, 1, 5, 1, 0, 0};
    assert(ts_nd_execute(12, 0, 0, 0, 0, 0, 2, shape, selection, p, indices) == 0);
    for (int i = 0; i < 15; ++i) assert(out[i] == (mask[i] ? left[i] : 1));

    int64_t a[] = {INT64_MIN, INT64_MAX, -3}, b[] = {-1, 1, 2}, result[3];
    p[0] = result; p[1] = a; p[2] = b; p[3] = NULL;
    int64_t three[] = {3}, unit[] = {1, 1, 1, 0};
    assert(ts_nd_execute(3, 8, 0, 0, 0, 0, 1, three, unit, p, indices) == 0);
    assert(result[0] == INT64_MIN && result[1] == INT64_MAX && result[2] == -1);
    b[2] = 0;
    assert(ts_nd_execute(3, 8, 0, 0, 0, 0, 1, three, unit, p, indices) == -3);

    uint16_t encoded[] = {0x8000, 0x3c00, 0x7e42, 0}, copied[8] = {0};
    p[0] = copied; p[1] = encoded + 2; p[2] = NULL;
    int64_t reverse[] = {2, -1, 0, 0};
    assert(ts_nd_execute(13, 5, 3, 3, 0, 0, 1, three, reverse, p, indices) == 0);
    assert(copied[0] == 0x7e42 && copied[2] == 0x3c00 && copied[4] == 0x8000);

    p[0] = out; p[1] = left;
    assert(ts_nd_execute(4, 0, 0, 0, 0, 0, 0, NULL, NULL, p, NULL) == 0);
    assert(out[0] == -left[0]);
    int64_t seven[] = {7};
    left[0] = -0.0f; left[5] = -1; out[0] = out[1] = -1;
    assert(ts_nd_execute(6, 0, 0, 0, 0, 0, 1, seven, unit, p, indices) == -2);
    assert(out[0] == 0 && out[1] == 1);
    uint32_t zero_bits; memcpy(&zero_bits, out, sizeof zero_bits);
    assert(zero_bits == 0x80000000u);
    left[5] = 5;
    int64_t empty[] = {0};
    void *nulls[] = {NULL, NULL, NULL, NULL};
    assert(ts_nd_execute(0, 0, 0, 0, 0, 0, 1, empty, unit, nulls, indices) == 0);
    assert(ts_nd_execute(99, 0, 0, 0, 0, 0, 1, three, unit, p, indices) == -1);
    return 0;
}
