/*
 * pb_analyze.cpp - Main Analysis Function
 */

#include "pb_audio_stats.h"
#include "pb_simd.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace pb_audio {

// ============================================================================
// Main Analysis Function
// ============================================================================

AudioStats analyze(const std::string& filepath) {
    AudioStats stats;

    // File info
    fs::path p(filepath);
    stats.filepath = filepath;
    stats.filename_ext = p.filename().string();
    stats.filename = p.stem().string();

    // Load audio
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

    // Loudness measurement (BS.1770-4)
    auto loudness = LoudnessMeter::measure(*audio);
    stats.integrated_loudness = loudness.integrated;
    stats.shortterm_max = loudness.shortterm_max;
    stats.momentary_max = loudness.momentary_max;
    stats.loudness_range = loudness.range;
    stats.sample_peak = loudness.sample_peak;

    // True Peak measurement
    stats.true_peak = TruePeakMeter::measure(*audio);

    // RMS measurement (50ms window like SOX)
    auto rms = RMSMeter::measure(*audio, 50.0);
    stats.rms_min = rms.min_db;
    stats.rms_max = rms.max_db;
    stats.rms_average = rms.average_db;

    return stats;
}

} // namespace pb_audio
