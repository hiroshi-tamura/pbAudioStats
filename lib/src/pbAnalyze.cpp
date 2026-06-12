/*
 * pbAnalyze.cpp - Main Analysis Function
 *
 * Provides three analyze() overloads:
 *   - analyze(filepath): convenience, picks the best path automatically.
 *   - analyze(filepath, use_single_pass): legacy switch (both values now
 *     behave identically; kept for API compatibility).
 *   - analyze(AudioData&, source_path): runs on already-loaded data, used by
 *     pbNormalize to avoid loading the same file twice.
 *
 * Data-flow notes:
 *   - The sample peak is scanned ONCE and shared by measure_with_rms and the
 *     true-peak meter (which also uses it to seed its pruning bound).
 *   - Files larger than the streaming threshold are analyzed in a single
 *     streaming pass (loudness + RMS + sample peak + exact oversampled true
 *     peak); results are identical to the in-memory path, so the choice is
 *     purely a memory/performance tradeoff.
 */

#include "pbAudioStats.h"
#include "pbSimd.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace pb_audio {

namespace {

fs::path utf8_path(const std::string& utf8) {
#if defined(_WIN32)
    return fs::u8path(utf8);
#else
    return fs::path(utf8);
#endif
}

void fill_file_info(AudioStats& stats, const std::string& source_path) {
    fs::path p = utf8_path(source_path);
    stats.filepath = source_path;
#if defined(_WIN32)
    stats.filename_ext = p.filename().u8string();
    stats.filename = p.stem().u8string();
#else
    stats.filename_ext = p.filename().string();
    stats.filename = p.stem().string();
#endif
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

    // One shared sample-peak scan (used by both meters below).
    const double sample_peak_linear = static_cast<double>(
        simd::find_peak_abs(audio.samples.data(), audio.samples.size()));

    auto loudness = LoudnessMeter::measure_with_rms(audio, 50.0, sample_peak_linear);
    stats.integrated_loudness = loudness.loudness.integrated;
    stats.shortterm_max = loudness.loudness.shortterm_max;
    stats.momentary_max = loudness.loudness.momentary_max;
    stats.loudness_range = loudness.loudness.range;
    stats.sample_peak = loudness.loudness.sample_peak;

    stats.rms_min = loudness.rms_min;
    stats.rms_max = loudness.rms_max;
    stats.rms_average = loudness.rms_average;

    stats.true_peak = TruePeakMeter::measure(audio, sample_peak_linear);

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

    // Exact BS.1770-4 oversampled true peak, computed in the same streaming
    // pass (previously this path silently reported the sample peak).
    stats.true_peak = loudness.true_peak;

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

AudioStats analyze(const std::string& filepath, bool /*use_single_pass*/) {
    AudioStats stats;
    fill_file_info(stats, filepath);

    // Stream large files: single pass over the data, bounded memory, and
    // (since the streaming path computes the exact oversampled true peak)
    // identical results to the in-memory path.
    uintmax_t file_size = 0;
    std::error_code file_size_ec;
    file_size = fs::file_size(utf8_path(filepath), file_size_ec);
    const uintmax_t stream_threshold_bytes = 32ull * 1024ull * 1024ull;
    const bool use_stream = !file_size_ec && file_size > stream_threshold_bytes;

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
