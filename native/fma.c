#include <math.h>
#include "fma.h"

#if defined(TS_HAVE_HARDWARE_FMA) && (defined(__x86_64__) || defined(_M_X64))
#ifdef _MSC_VER
#include <intrin.h>
static volatile long fma_capability = -1;
#else
#include <cpuid.h>
#include <stdatomic.h>
static atomic_int fma_capability = ATOMIC_VAR_INIT(-1);
#endif

static int detect_fma(void) {
    unsigned int ecx;
#ifdef _MSC_VER
    int registers[4];
    __cpuid(registers, 0);
    if (registers[0] < 1) return 0;
    __cpuidex(registers, 1, 0);
    ecx = (unsigned int)registers[2];
#else
    unsigned int eax, ebx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return 0;
#endif
    const unsigned int required = (1u << 12) | (1u << 26) | (1u << 27) | (1u << 28);
    if ((ecx & required) != required) return 0;
#ifdef _MSC_VER
    if ((_xgetbv(0) & 6) != 6) return 0;
    __cpuidex(registers, 7, 0);
    return ((unsigned int)registers[1] & (1u << 5)) != 0;
#else
    unsigned int low, high;
    __asm__ volatile ("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    if ((low & 6) != 6 || __get_cpuid_max(0, 0) < 7) return 0;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1u << 5)) != 0;
#endif
}
#endif

int ts_fma_supported(void) {
#if defined(TS_HAVE_HARDWARE_FMA) && !defined(TS_FORCE_SOFTWARE_FMA) && (defined(__x86_64__) || defined(_M_X64))
#ifdef _MSC_VER
    long cached = _InterlockedCompareExchange(&fma_capability, -1, -1);
    if (cached < 0) {
        cached = detect_fma();
        _InterlockedCompareExchange(&fma_capability, cached, -1);
    }
#else
    int cached = atomic_load_explicit(&fma_capability, memory_order_relaxed);
    if (cached < 0) {
        cached = detect_fma();
        atomic_store_explicit(&fma_capability, cached, memory_order_relaxed);
    }
#endif
    return (int)cached;
#else
    return 0;
#endif
}

float ts_fma_scalar_f32(float a, float b, float c) { return fmaf(a, b, c); }
double ts_fma_scalar_f64(double a, double b, double c) { return fma(a, b, c); }
