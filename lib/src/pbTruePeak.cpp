/*
 * pbTruePeak.cpp - True Peak Measurement (ITU-R BS.1770-4 / EBU Tech 3341)
 *
 * Performs 4x oversampling with a polyphase Kaiser-windowed sinc low-pass
 * filter and reports the maximum absolute interpolated sample value (dBTP).
 *
 * Configuration: 48-tap prototype filter -> 4 phases x 12 taps per phase.
 * The filter is designed once at startup with cutoff = pi / OVERSAMPLE in
 * the upsampled domain (i.e. half the original Nyquist) and a Kaiser window
 * with beta = 9.6 (~89 dB stopband).
 */

#include "pbAudioStats.h"
#include "pbSimd.h"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pb_audio {

namespace {

constexpr int OVERSAMPLE = 4;
constexpr int TAPS_PER_PHASE = 12;
constexpr int TOTAL_TAPS = OVERSAMPLE * TAPS_PER_PHASE;  // 48

using PhaseCoeffs = std::array<std::array<double, TAPS_PER_PHASE>, OVERSAMPLE>;

double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    double half_x = 0.5 * x;
    for (int k = 1; k < 50; ++k) {
        term *= (half_x * half_x) / static_cast<double>(k * k);
        sum += term;
        if (term < 1e-15 * sum) break;
    }
    return sum;
}

PhaseCoeffs build_phase_coeffs() {
    std::array<double, TOTAL_TAPS> proto{};
    const double cutoff = M_PI / static_cast<double>(OVERSAMPLE);
    const double center = (static_cast<double>(TOTAL_TAPS) - 1.0) * 0.5;
    const double beta = 9.6;
    const double i0_beta = bessel_i0(beta);

    for (int n = 0; n < TOTAL_TAPS; ++n) {
        double m = static_cast<double>(n) - center;
        double sinc = (std::fabs(m) < 1e-12) ? (cutoff / M_PI)
                                             : (std::sin(cutoff * m) / (M_PI * m));
        double r = (2.0 * n / static_cast<double>(TOTAL_TAPS - 1)) - 1.0;
        double window = bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0_beta;
        proto[n] = sinc * window * static_cast<double>(OVERSAMPLE);
    }

    PhaseCoeffs phases{};
    for (int p = 0; p < OVERSAMPLE; ++p) {
        for (int t = 0; t < TAPS_PER_PHASE; ++t) {
            phases[p][t] = proto[t * OVERSAMPLE + p];
        }
    }
    return phases;
}

const PhaseCoeffs& phase_coeffs() {
    static const PhaseCoeffs coeffs = build_phase_coeffs();
    return coeffs;
}

}  // namespace

double TruePeakMeter::measure(const AudioData& audio) {
    if (audio.samples.empty() || audio.channels == 0 || audio.total_frames == 0) {
        return -std::numeric_limits<double>::infinity();
    }

    const PhaseCoeffs& coeffs = phase_coeffs();
    const size_t total_frames = audio.total_frames;
    const int channels = audio.channels;

    // Sample-rate peak as a baseline; oversampled peaks must be at least this.
    float max_peak = simd::find_peak_abs(audio.samples.data(), audio.samples.size());

    // Per-channel sliding window of the latest TAPS_PER_PHASE input samples
    // (history[0] is the current frame; history[k] is x[n-k]).
    std::vector<double> history(static_cast<size_t>(channels) * TAPS_PER_PHASE, 0.0);

    for (size_t frame = 0; frame < total_frames; ++frame) {
        const float* fs = audio.samples.data() + frame * static_cast<size_t>(channels);
        for (int ch = 0; ch < channels; ++ch) {
            double* h = history.data() + static_cast<size_t>(ch) * TAPS_PER_PHASE;
            // Shift right
            for (int i = TAPS_PER_PHASE - 1; i > 0; --i) h[i] = h[i - 1];
            h[0] = static_cast<double>(fs[ch]);

            for (int p = 0; p < OVERSAMPLE; ++p) {
                double acc = 0.0;
                const auto& pc = coeffs[p];
                for (int t = 0; t < TAPS_PER_PHASE; ++t) {
                    acc += pc[t] * h[t];
                }
                float a = static_cast<float>(std::fabs(acc));
                if (a > max_peak) max_peak = a;
            }
        }
    }

    if (max_peak <= 0.0f) return -std::numeric_limits<double>::infinity();
    return 20.0 * std::log10(static_cast<double>(max_peak));
}

void TruePeakMeter::apply_lpf(const float* input, float* output, size_t count, int /*channels*/) {
    // Retained for ABI compatibility; not used by measure().
    for (size_t i = 0; i < count; ++i) output[i] = input[i];
}

}  // namespace pb_audio
