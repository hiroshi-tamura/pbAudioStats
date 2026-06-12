/*
 * cpu_check.cpp - CPU capability guard
 *
 * This translation unit is compiled WITHOUT AVX2 codegen flags and contains
 * the real main(): it verifies the CPU supports the instruction set the rest
 * of the binary was built for BEFORE any AVX2 code can execute, then calls
 * app_main(). Without this, running the binary on a pre-AVX2 CPU dies with an
 * undiagnosable 0xC000001D illegal-instruction crash.
 */

#include <cstdio>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
static bool cpu_supports_avx2_fma() {
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuidex(info, 1, 0);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx = (info[2] & (1 << 28)) != 0;
    const bool fma = (info[2] & (1 << 12)) != 0;
    if (!osxsave || !avx || !fma) return false;
    // OS must save/restore XMM+YMM state
    if ((_xgetbv(0) & 0x6) != 0x6) return false;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;  // AVX2
}
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
static bool cpu_supports_avx2_fma() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#else
// Non-x86 targets (ARM64 NEON is baseline) need no runtime check.
static bool cpu_supports_avx2_fma() { return true; }
#endif

int app_main(int argc, char* argv[]);

int main(int argc, char* argv[]) {
#if defined(PB_REQUIRE_AVX2)
    if (!cpu_supports_avx2_fma()) {
        std::fputs("Error: pbAudioStats requires a CPU with AVX2/FMA support "
                   "(Intel Haswell 2013+ / AMD Excavator 2015+).\n", stderr);
        return 3;
    }
#endif
    return app_main(argc, argv);
}
