/*
 * pb_truepeak.cpp - True Peak Measurement
 *
 * This implementation uses Sample Peak measurement (same as SOX "Pk lev dB")
 * instead of ITU-R BS.1770-4 inter-sample peak detection.
 *
 * SOX compatible: Finds the maximum absolute sample value and converts to dB.
 *
 * SIMD optimizations: Uses pb_simd.h for accelerated peak detection.
 */

#include "pb_audio_stats.h"
#include "pb_simd.h"
#include <cmath>
#include <vector>
#include <algorithm>

// M_PI is not defined in MSVC by default
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pb_audio {

// ============================================================================
// True Peak Measurement (SOX Pk lev dB compatible)
// ============================================================================

double TruePeakMeter::measure(const AudioData& audio) {
    if (audio.samples.empty()) return -100.0;

    size_t total_samples = audio.samples.size();

    // Use SIMD-optimized peak detection
    // pb_audio::simd::find_peak_abs() uses AVX2/NEON/scalar depending on platform
    float max_peak = simd::find_peak_abs(audio.samples.data(), total_samples);

    // Convert to dB
    if (max_peak <= 0.0f) return -100.0;
    return 20.0 * std::log10(static_cast<double>(max_peak));
}

// ============================================================================
// Helper: Apply LPF (not used in sample peak measurement)
// ============================================================================

void TruePeakMeter::apply_lpf(const float* input, float* output, size_t count, int channels) {
    // Pass through - no filtering needed for sample peak
    for (size_t i = 0; i < count; i++) {
        output[i] = input[i];
    }
}

} // namespace pb_audio
