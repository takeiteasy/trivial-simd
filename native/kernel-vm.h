#define TS_KERNEL_DECODE(type, components, output) \
    unsigned op = code[pc], dst = code[pc + 1], a = code[pc + 2], b = code[pc + 3]; \
    if (op == TS_OP_SPILL || op == TS_OP_RELOAD) { \
        type *slot = scratch + ((size_t)a | ((size_t)b << 8)) * TS_KERNEL_BLOCK * components; \
        if (op == TS_OP_SPILL) memcpy(slot, registers[dst], m * components * sizeof(type)); \
        else memcpy(registers[dst], slot, m * components * sizeof(type)); \
        continue; \
    } \
    type *destination = dst == TS_KERNEL_OUTPUT ? (output) : registers[dst]; \
    const type *left = NULL, *right = NULL; \
    if (op != TS_OP_CONSTANT) \
        left = a < TS_KERNEL_REGISTERS ? registers[a] : inputs[a - TS_KERNEL_REGISTERS] + base * components; \
    if ((op >= TS_OP_ADD && op <= TS_OP_DIVIDE) || op == TS_OP_MIN || op == TS_OP_MAX || \
        op == TS_OP_FMA || (op >= TS_OP_EQ && op <= TS_OP_SELECT)) \
        right = b < TS_KERNEL_REGISTERS ? registers[b] : inputs[b - TS_KERNEL_REGISTERS] + base * components

#define TS_KERNEL_EXECUTE(type, components, sum, reduce, apply) \
    if (!n) { if (sum) memset(out, 0, components * sizeof(type)); return 0; } \
    if (scratch_count > SIZE_MAX / TS_KERNEL_BLOCK / components / sizeof(type)) return -1; \
    type registers[TS_KERNEL_REGISTERS][TS_KERNEL_BLOCK * components]; \
    type result[components] = {0}; \
    for (size_t base = 0; base < n; base += TS_KERNEL_BLOCK) { \
        size_t m = n - base < TS_KERNEL_BLOCK ? n - base : TS_KERNEL_BLOCK; \
        type block_result[components] = {0}; \
        for (size_t pc = 0; pc < code_length; pc += 4) { \
            TS_KERNEL_DECODE(type, components, out + (sum ? 0 : base * components)); \
            const type *constant = op == TS_OP_CONSTANT ? constants + a * components : NULL; \
            if (sum && dst == TS_KERNEL_OUTPUT) { \
                int status = (reduce); \
                if (status) return status; \
            } else if (constant) { \
                for (size_t j = 0; j < m; ++j) \
                    memcpy(destination + j * components, constant, components * sizeof(type)); \
            } else { \
                int status = (apply); \
                if (status) return status; \
            } \
        } \
        if (sum) for (size_t j = 0; j < components; ++j) result[j] += block_result[j]; \
    } \
    if (sum) memcpy(out, result, components * sizeof(type)); \
    return 0
