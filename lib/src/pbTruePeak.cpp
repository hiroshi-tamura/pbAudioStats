/*
 * pbTruePeak.cpp - True Peak Measurement (ITU-R BS.1770-4 / EBU Tech 3341)
 *
 * 4x oversampling with a polyphase Kaiser-windowed sinc low-pass filter
 * (48-tap prototype -> 4 phases x 12 taps, beta = 9.6, ~89 dB stopband),
 * reporting the maximum absolute interpolated sample value (dBTP).
 *
 * The heavy lifting lives in pbTruePeakScanner.h (internal): a streaming
 * block-FIR scanner with AVX2/NEON kernels, rigorous L1-bound chunk pruning
 * and end-of-stream flushing. This file only adapts it to the public API.
 */

#include "pbAudioStats.h"
#include "pbSimd.h"
#include "pbTruePeakScanner.h"

#include <cmath>
#include <limits>

namespace pb_audio {

double TruePeakMeter::measure(const AudioData& audio, double known_sample_peak_linear) {
    if (audio.samples.empty() || audio.channels == 0 || audio.total_frames == 0) {
        return -std::numeric_limits<double>::infinity();
    }

    tp_detail::TruePeakScanner scanner(audio.channels);

    // Sample-rate peak as a baseline; oversampled peaks must be at least this.
    // Seeding up front also makes chunk pruning effective from the start.
    double sample_peak = known_sample_peak_linear;
    if (sample_peak < 0.0) {
        sample_peak = static_cast<double>(
            simd::find_peak_abs(audio.samples.data(), audio.samples.size()));
    }
    scanner.seed_peak(sample_peak);

    scanner.process(audio.samples.data(), audio.total_frames);
    scanner.flush();

    double max_peak = scanner.peak_linear();
    if (max_peak <= 0.0) return -std::numeric_limits<double>::infinity();
    return 20.0 * std::log10(max_peak);
}

double TruePeakMeter::measure(const AudioData& audio) {
    return measure(audio, -1.0);
}

void TruePeakMeter::apply_lpf(const float* input, float* output, size_t count, int /*channels*/) {
    // Retained for ABI compatibility; not used by measure().
    for (size_t i = 0; i < count; ++i) output[i] = input[i];
}

}  // namespace pb_audio
