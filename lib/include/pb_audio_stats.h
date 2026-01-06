/*
 * pb_audio_stats.h - Audio Statistics Library
 * Supports: WAV, AIFF, MP3
 * Features: BS.1770-4 Loudness, True Peak, RMS, Normalization
 */

#ifndef PB_AUDIO_STATS_H
#define PB_AUDIO_STATS_H

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace pb_audio {

// Audio format enumeration
enum class AudioFormat {
    Unknown,
    WAV,
    AIFF,
    MP3
};

// Audio data container
struct AudioData {
    std::vector<float> samples;      // Interleaved samples
    uint32_t sample_rate;
    uint16_t channels;
    uint16_t bit_depth;
    uint64_t total_frames;

    double duration_seconds() const {
        return sample_rate > 0 ? (double)total_frames / sample_rate : 0.0;
    }
};

// Measurement results
struct AudioStats {
    // File information
    std::string filename;            // Without extension
    std::string filename_ext;        // With extension
    std::string filepath;            // Full path
    uint32_t sample_rate;
    uint16_t bit_depth;
    uint16_t channels;
    double duration_seconds;
    std::string duration_formatted;  // HH:MM:SS.mmm

    // Loudness (BS.1770-4)
    double integrated_loudness;      // LUFS
    double shortterm_max;            // LUFS
    double momentary_max;            // LUFS
    double loudness_range;           // LU

    // Peak
    double sample_peak;              // dBFS
    double true_peak;                // dBFS

    // RMS
    double rms_min;                  // dB
    double rms_max;                  // dB
    double rms_average;              // dB
};

// ============================================================================
// Audio File Reader
// ============================================================================

class AudioReader {
public:
    static AudioFormat detect_format(const std::string& filepath);
    static std::unique_ptr<AudioData> load(const std::string& filepath);

private:
    static std::unique_ptr<AudioData> load_wav(const std::string& filepath);
    static std::unique_ptr<AudioData> load_aiff(const std::string& filepath);
    static std::unique_ptr<AudioData> load_mp3(const std::string& filepath);
};

// ============================================================================
// BS.1770-4 Loudness Measurement
// ============================================================================

class LoudnessMeter {
public:
    struct Result {
        double integrated;       // LUFS
        double momentary_max;    // LUFS
        double shortterm_max;    // LUFS
        double range;            // LU (LRA)
        double sample_peak;      // dBFS
    };

    static Result measure(const AudioData& audio);

private:
    // K-weighting filter coefficients
    struct BiquadCoeffs {
        double b0, b1, b2;
        double a1, a2;
    };

    static BiquadCoeffs calc_high_shelf(double sample_rate);
    static BiquadCoeffs calc_high_pass(double sample_rate);
};

// ============================================================================
// True Peak Measurement (ITU-R BS.1770-4 compliant)
// ============================================================================

class TruePeakMeter {
public:
    // 4x oversampling as per BS.1770-4
    static double measure(const AudioData& audio);

private:
    static constexpr int OVERSAMPLE_FACTOR = 4;
    static void apply_lpf(const float* input, float* output, size_t count, int channels);
};

// ============================================================================
// RMS Measurement
// ============================================================================

class RMSMeter {
public:
    struct Result {
        double min_db;           // RMS trough (floor at -96 dB)
        double max_db;           // RMS peak
        double average_db;       // RMS average
    };

    // Window size in milliseconds (50ms as per sox default)
    static Result measure(const AudioData& audio, double window_ms = 50.0);
};

// ============================================================================
// Normalizer
// ============================================================================

class Normalizer {
public:
    enum class Target {
        Peak,
        TruePeak,
        Integrated,
        ShorttermMax,
        MomentaryMax,
        RMSMin,
        RMSMax,
        RMSAverage
    };

    // Returns gain in dB needed to normalize to target
    static double calculate_gain(const AudioStats& stats, Target target, double target_value);

    // Apply gain to audio data (modifies in place)
    static void apply_gain(AudioData& audio, double gain_db);

    // Normalize and save to file
    static bool normalize_and_save(const std::string& input_path,
                                   const std::string& output_path,
                                   Target target,
                                   double target_value);
};

// ============================================================================
// Main Analysis Function
// ============================================================================

// Analyze single file
AudioStats analyze(const std::string& filepath);

// Utility functions
std::string format_duration(double seconds);
double linear_to_db(double linear);
double db_to_linear(double db);

} // namespace pb_audio

#endif // PB_AUDIO_STATS_H
