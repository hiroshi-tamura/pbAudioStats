/*
 * pb_rms.cpp - RMS Measurement (SOX compatible)
 * Uses exponential moving average with 50ms time constant (SOX default)
 *
 * SOX uses exponential smoothing for RMS Peak/Trough:
 *   mult = exp(-1 / (time_constant * sample_rate))
 *   avg_sigma_x2 = avg_sigma_x2 * mult + (1 - mult) * sample^2
 *
 * SOX Overall calculation:
 * - RMS lev dB = overall RMS across all channels (sum of squares / total samples)
 * - RMS Pk dB = max across all channels' max_sigma_x2
 * - RMS Tr dB = min across all channels' min_sigma_x2
 *
 * NOTE: RMS Min values are output as-is (no floor restriction).
 * Only -inf is floored to -96 dB for SOX compatibility.
 */

#include "pb_audio_stats.h"
#include "pb_simd.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

namespace pb_audio {

// Floor value for -inf only (SOX compatibility)
static constexpr double INF_FLOOR_DB = -96.0;

// ============================================================================
// RMS Meter Implementation (SOX compatible)
// ============================================================================

RMSMeter::Result RMSMeter::measure(const AudioData& audio, double window_ms) {
    Result result;

    // Default values for empty/invalid audio
    result.min_db = INF_FLOOR_DB;
    result.max_db = INF_FLOOR_DB;
    result.average_db = INF_FLOOR_DB;

    if (audio.samples.empty() || audio.channels == 0 || audio.sample_rate == 0) {
        return result;
    }

    const uint32_t channels = audio.channels;
    const uint64_t total_frames = audio.total_frames;
    const double sample_rate = static_cast<double>(audio.sample_rate);

    // Time constant in seconds (window_ms is the time constant, not window size)
    const double time_constant = window_ms / 1000.0;

    // Exponential smoothing multiplier (SOX formula)
    // mult = exp(-1 / (time_constant * sample_rate))
    const double mult = std::exp(-1.0 / (time_constant * sample_rate));
    const double one_minus_mult = 1.0 - mult;

    // Settling time: 5 * time_constant * sample_rate samples (frames)
    // SOX only starts tracking min/max after this settling period
    const uint64_t tc_samples = static_cast<uint64_t>(5.0 * time_constant * sample_rate);

    // Per-channel tracking (SOX processes each channel independently)
    std::vector<double> avg_sigma_x2(channels, 0.0);
    std::vector<double> max_sigma_x2(channels, 0.0);
    std::vector<double> min_sigma_x2(channels, std::numeric_limits<double>::max());
    std::vector<double> sum_sq(channels, 0.0);
    std::vector<uint64_t> sample_count(channels, 0);

    // Process each frame
    for (uint64_t frame = 0; frame < total_frames; frame++) {
        for (uint32_t ch = 0; ch < channels; ch++) {
            uint64_t idx = frame * channels + ch;
            double sample = static_cast<double>(audio.samples[idx]);
            double sample_sq = sample * sample;

            // Accumulate for overall RMS
            sum_sq[ch] += sample_sq;
            sample_count[ch]++;

            // Exponential moving average update
            avg_sigma_x2[ch] = avg_sigma_x2[ch] * mult + one_minus_mult * sample_sq;

            // Track min/max only after settling period
            if (frame >= tc_samples) {
                if (avg_sigma_x2[ch] > max_sigma_x2[ch]) {
                    max_sigma_x2[ch] = avg_sigma_x2[ch];
                }
                if (avg_sigma_x2[ch] < min_sigma_x2[ch]) {
                    min_sigma_x2[ch] = avg_sigma_x2[ch];
                }
            }
        }
    }

    // Handle files shorter than settling time (SOX behavior)
    // When num_samples < tc_samples, SOX uses sigma_x2 / num_samples instead of avg_sigma_x2
    for (uint32_t ch = 0; ch < channels; ch++) {
        if (sample_count[ch] < tc_samples) {
            double avg_power = sum_sq[ch] / static_cast<double>(sample_count[ch]);
            max_sigma_x2[ch] = avg_power;
            min_sigma_x2[ch] = avg_power;
        }
    }

    // Calculate Overall values (SOX method)
    // RMS lev dB: combine all channels
    // Use SIMD-optimized sum of squares for overall RMS calculation
    double total_sum_sq = 0.0;
    uint64_t total_sample_count = static_cast<uint64_t>(audio.samples.size());

    if (channels == 2) {
        // Use optimized stereo sum of squares
        total_sum_sq = simd::weighted_sum_of_squares_stereo(audio.samples.data(), total_frames);
    } else {
        // Use optimized sum of squares for all samples
        total_sum_sq = simd::sum_of_squares(audio.samples.data(), audio.samples.size());
    }

    if (total_sample_count > 0 && total_sum_sq > 0.0) {
        double avg_rms = std::sqrt(total_sum_sq / total_sample_count);
        result.average_db = 20.0 * std::log10(avg_rms);
    }

    // RMS Pk dB: max across all channels' max_sigma_x2
    double overall_max_sigma_x2 = 0.0;
    for (uint32_t ch = 0; ch < channels; ch++) {
        if (max_sigma_x2[ch] > overall_max_sigma_x2) {
            overall_max_sigma_x2 = max_sigma_x2[ch];
        }
    }

    if (overall_max_sigma_x2 > 0.0) {
        double max_rms = std::sqrt(overall_max_sigma_x2);
        result.max_db = 20.0 * std::log10(max_rms);
    }

    // RMS Tr dB: min across all channels' min_sigma_x2
    double overall_min_sigma_x2 = std::numeric_limits<double>::max();
    for (uint32_t ch = 0; ch < channels; ch++) {
        if (min_sigma_x2[ch] < overall_min_sigma_x2) {
            overall_min_sigma_x2 = min_sigma_x2[ch];
        }
    }

    if (overall_min_sigma_x2 > 0.0 && overall_min_sigma_x2 < std::numeric_limits<double>::max()) {
        double min_rms = std::sqrt(overall_min_sigma_x2);
        double min_db = 20.0 * std::log10(min_rms);
        // Only floor -inf to -96 dB (SOX compatibility)
        // All other values (including very low like -3202.64 dB) are output as-is
        if (std::isinf(min_db)) {
            result.min_db = INF_FLOOR_DB;
        } else {
            result.min_db = min_db;  // No floor restriction
        }
    } else {
        result.min_db = INF_FLOOR_DB;
    }

    return result;
}

} // namespace pb_audio
