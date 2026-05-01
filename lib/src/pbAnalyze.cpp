/*
 * pbAnalyze.cpp - Main Analysis Function
 *
 * Provides three analyze() overloads:
 *   - analyze(filepath): convenience, picks the best path automatically.
 *   - analyze(filepath, use_single_pass): legacy single/double pass switch.
 *   - analyze(AudioData&, source_path): runs on already-loaded data, used by
 *     pbNormalize to avoid loading the same file twice.
 *
 * The "double pass" path was historically slightly faster on long files for
 * RMS-only or Loudness-only output, but always doubled the work when both
 * were requested. We now always use measure_with_rms() (one walk through the
 * audio for both K-weighting and SOX-style RMS) which strictly dominates the
 * old double-pass implementation.
 */

#include "pbAudioStats.h"
#include "pbSimd.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace pb_audio {

namespace {

void fill_file_info(AudioStats& stats, const std::string& source_path) {
    fs::path p(source_path);
    stats.filepath = source_path;
    stats.filename_ext = p.filename().string();
    stats.filename = p.stem().string();
}

void fill_invalid(AudioStats& stats) {
    stats.valid = false;
    stats.sample_rate = 0;
    stats.bit_depth = 0;
    stats.channels = 0;
    stats.duration_seconds = 0.0;
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
}

void run_full_analysis(AudioStats& stats, const AudioData& audio) {
    stats.sample_rate = audio.sample_rate;
    stats.bit_depth = audio.bit_depth;
    stats.channels = audio.channels;
    stats.duration_seconds = audio.duration_seconds();
    stats.duration_formatted = format_duration(stats.duration_seconds);

    auto loudness = LoudnessMeter::measure_with_rms(audio, 50.0);
    stats.integrated_loudness = loudness.loudness.integrated;
    stats.shortterm_max = loudness.loudness.shortterm_max;
    stats.momentary_max = loudness.loudness.momentary_max;
    stats.loudness_range = loudness.loudness.range;
    stats.sample_peak = loudness.loudness.sample_peak;

    stats.rms_min = loudness.rms_min;
    stats.rms_max = loudness.rms_max;
    stats.rms_average = loudness.rms_average;

    stats.true_peak = TruePeakMeter::measure(audio);

    stats.valid = true;
}

void run_stream_analysis(AudioStats& stats, AudioStream& stream) {
    stats.sample_rate = stream.info.sample_rate;
    stats.bit_depth = stream.info.bit_depth;
    stats.channels = stream.info.channels;
    stats.duration_seconds = (stats.sample_rate > 0 && stream.info.total_frames > 0)
        ? static_cast<double>(stream.info.total_frames) /
          static_cast<double>(stats.sample_rate)
        : 0.0;
    stats.duration_formatted = format_duration(stats.duration_seconds);

    auto loudness = LoudnessMeter::measure_stream(stream, 50.0);
    stats.integrated_loudness = loudness.loudness.integrated;
    stats.shortterm_max = loudness.loudness.shortterm_max;
    stats.momentary_max = loudness.loudness.momentary_max;
    stats.loudness_range = loudness.loudness.range;
    stats.sample_peak = loudness.loudness.sample_peak;

    stats.rms_min = loudness.rms_min;
    stats.rms_max = loudness.rms_max;
    stats.rms_average = loudness.rms_average;

    // Streaming path cannot afford the polyphase oversampling pass without
    // a second read; report sample peak as a conservative true-peak estimate.
    stats.true_peak = stats.sample_peak;

    stats.valid = true;
}

}  // namespace

AudioStats analyze(const AudioData& audio, const std::string& source_path) {
    AudioStats stats;
    fill_file_info(stats, source_path);
    if (audio.samples.empty() || audio.channels == 0 || audio.sample_rate == 0) {
        fill_invalid(stats);
        return stats;
    }
    run_full_analysis(stats, audio);
    return stats;
}

AudioStats analyze(const std::string& filepath, bool use_single_pass) {
    AudioStats stats;
    fill_file_info(stats, filepath);

    uintmax_t file_size = 0;
    std::error_code file_size_ec;
    file_size = fs::file_size(fs::path(filepath), file_size_ec);
    const uintmax_t stream_threshold_bytes = 32ull * 1024ull * 1024ull;
    bool use_stream = use_single_pass && !file_size_ec &&
                      file_size > stream_threshold_bytes;

    if (use_stream) {
        auto stream = AudioReader::open_stream(filepath);
        if (stream) {
            run_stream_analysis(stats, *stream);
            return stats;
        }
    }

    auto audio = AudioReader::load(filepath);
    if (!audio) {
        fill_invalid(stats);
        return stats;
    }

    run_full_analysis(stats, *audio);
    return stats;
}

AudioStats analyze(const std::string& filepath) {
    return analyze(filepath, false);
}

}  // namespace pb_audio
