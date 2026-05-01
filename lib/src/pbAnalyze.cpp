/*
 * pbAnalyze.cpp - Main Analysis Function
 */

#include "pbAudioStats.h"
#include "pbSimd.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace pb_audio {

// ============================================================================
// Main Analysis Function
// ============================================================================

AudioStats analyze(const std::string& filepath) {
    return analyze(filepath, false);
}

AudioStats analyze(const std::string& filepath, bool use_single_pass) {
    AudioStats stats;

    // File info
    fs::path p(filepath);
    stats.filepath = filepath;
    stats.filename_ext = p.filename().string();
    stats.filename = p.stem().string();

    uintmax_t file_size = 0;
    std::error_code file_size_ec;
    file_size = fs::file_size(p, file_size_ec);
    const uintmax_t stream_threshold_bytes = 32ull * 1024ull * 1024ull;
    bool use_stream = use_single_pass &&
        !file_size_ec &&
        file_size > stream_threshold_bytes;

    if (use_stream) {
        auto stream = AudioReader::open_stream(filepath);
        if (stream) {
            stats.sample_rate = stream->info.sample_rate;
            stats.bit_depth = stream->info.bit_depth;
            stats.channels = stream->info.channels;
            stats.duration_seconds = (stats.sample_rate > 0 && stream->info.total_frames > 0)
                ? static_cast<double>(stream->info.total_frames) / static_cast<double>(stats.sample_rate)
                : 0.0;
            stats.duration_formatted = format_duration(stats.duration_seconds);

            auto loudness = LoudnessMeter::measure_stream(*stream, 50.0);
            stats.integrated_loudness = loudness.loudness.integrated;
            stats.shortterm_max = loudness.loudness.shortterm_max;
            stats.momentary_max = loudness.loudness.momentary_max;
            stats.loudness_range = loudness.loudness.range;
            stats.sample_peak = loudness.loudness.sample_peak;

            stats.true_peak = stats.sample_peak;

            stats.rms_min = loudness.rms_min;
            stats.rms_max = loudness.rms_max;
            stats.rms_average = loudness.rms_average;
            return stats;
        }
    }

    // Load audio (fallback or SIMD-friendly path)
    auto audio = AudioReader::load(filepath);
    if (!audio) {
        // Return empty stats with file info only
        stats.sample_rate = 0;
        stats.bit_depth = 0;
        stats.channels = 0;
        stats.duration_seconds = 0;
        stats.duration_formatted = "00:00:00.000";
        stats.integrated_loudness = -70.0;
        stats.shortterm_max = -70.0;
        stats.momentary_max = -70.0;
        stats.loudness_range = 0.0;
        stats.sample_peak = -100.0;
        stats.true_peak = -100.0;
        stats.rms_min = -96.0;
        stats.rms_max = -96.0;
        stats.rms_average = -96.0;
        return stats;
    }

    // File info
    stats.sample_rate = audio->sample_rate;
    stats.bit_depth = audio->bit_depth;
    stats.channels = audio->channels;
    stats.duration_seconds = audio->duration_seconds();
    stats.duration_formatted = format_duration(stats.duration_seconds);

    if (!use_single_pass) {
        auto loudness = LoudnessMeter::measure(*audio);
        stats.integrated_loudness = loudness.integrated;
        stats.shortterm_max = loudness.shortterm_max;
        stats.momentary_max = loudness.momentary_max;
        stats.loudness_range = loudness.range;
        stats.sample_peak = loudness.sample_peak;

        stats.true_peak = stats.sample_peak;

        auto rms = RMSMeter::measure(*audio, 50.0);
        stats.rms_min = rms.min_db;
        stats.rms_max = rms.max_db;
        stats.rms_average = rms.average_db;
    } else {
        // Single pass is faster for short files
        auto loudness = LoudnessMeter::measure_with_rms(*audio, 50.0);
        stats.integrated_loudness = loudness.loudness.integrated;
        stats.shortterm_max = loudness.loudness.shortterm_max;
        stats.momentary_max = loudness.loudness.momentary_max;
        stats.loudness_range = loudness.loudness.range;
        stats.sample_peak = loudness.loudness.sample_peak;

        stats.true_peak = stats.sample_peak;

        stats.rms_min = loudness.rms_min;
        stats.rms_max = loudness.rms_max;
        stats.rms_average = loudness.rms_average;
    }

    return stats;
}

} // namespace pb_audio
