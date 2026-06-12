/*
 * pbNormalize.cpp - Audio Normalization
 * Supports Peak, True Peak, Integrated Loudness, Short-term, Momentary, RMS
 *
 * WAV/AIFF: Native implementation (no external dependencies)
 *
 * Performance notes:
 * - normalize_and_save computes ONLY the metric the target needs (a peak-only
 *   normalization no longer runs K-weighting, LRA histograms and the
 *   oversampled true-peak pass).
 * - The gain is fused into the SIMD float->PCM conversion in the writers
 *   (one pass instead of gain pass + clip pass + scalar conversion pass).
 * - Output is converted and written in chunks (no file-sized staging buffer).
 *
 * Accuracy notes:
 * - float->PCM conversion rounds to nearest (ties-to-even), matching SoX;
 *   the previous implementation truncated (up to 1 LSB biased error).
 * - The 32-bit PCM path scales in double precision; the previous float-based
 *   scaling overflowed full-scale +1.0 samples to INT32_MIN.
 */

#include "pbAudioStats.h"
#include "pbSimd.h"
#include <cmath>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <algorithm>
#include <vector>

namespace fs = std::filesystem;

namespace pb_audio {

// ============================================================================
// Gain Calculation
// ============================================================================

double Normalizer::calculate_gain(const AudioStats& stats, Target target, double target_value) {
    double current_value = 0.0;

    switch (target) {
        case Target::Peak:
            current_value = stats.sample_peak;
            break;
        case Target::TruePeak:
            current_value = stats.true_peak;
            break;
        case Target::Integrated:
            current_value = stats.integrated_loudness;
            break;
        case Target::ShorttermMax:
            current_value = stats.shortterm_max;
            break;
        case Target::MomentaryMax:
            current_value = stats.momentary_max;
            break;
        case Target::RMSMin:
            current_value = stats.rms_min;
            break;
        case Target::RMSMax:
            current_value = stats.rms_max;
            break;
        case Target::RMSAverage:
            current_value = stats.rms_average;
            break;
    }

    // Gain needed = target - current
    return target_value - current_value;
}

// ============================================================================
// Apply Gain
// ============================================================================

void Normalizer::apply_gain(AudioData& audio, double gain_db) {
    if (gain_db == 0.0) return;

    float linear_gain = static_cast<float>(db_to_linear(gain_db));
    simd::apply_gain_and_clip(audio.samples.data(), audio.samples.size(),
                              linear_gain, -1.0f, 1.0f);
}

// ============================================================================
// Path / endian helpers
// ============================================================================

static fs::path utf8_path(const std::string& utf8) {
#if defined(_WIN32)
    return fs::u8path(utf8);
#else
    return fs::path(utf8);
#endif
}

static void write_u16_le(uint8_t* p, uint16_t v) {
    p[0] = v & 0xFF;
    p[1] = (v >> 8) & 0xFF;
}

static void write_u32_le(uint8_t* p, uint32_t v) {
    p[0] = v & 0xFF;
    p[1] = (v >> 8) & 0xFF;
    p[2] = (v >> 16) & 0xFF;
    p[3] = (v >> 24) & 0xFF;
}

static void write_u32_be(uint8_t* p, uint32_t v) {
    p[0] = (v >> 24) & 0xFF;
    p[1] = (v >> 16) & 0xFF;
    p[2] = (v >> 8) & 0xFF;
    p[3] = v & 0xFF;
}

static void write_u16_be(uint8_t* p, uint16_t v) {
    p[0] = (v >> 8) & 0xFF;
    p[1] = v & 0xFF;
}

// Write IEEE 754 extended precision (80-bit)
static void write_extended_be(uint8_t* p, double value) {
    if (value == 0.0) {
        memset(p, 0, 10);
        return;
    }

    int sign = (value < 0) ? 1 : 0;
    if (sign) value = -value;

    int exponent;
    double mantissa = std::frexp(value, &exponent);
    exponent += 16382;

    uint64_t mant = (uint64_t)(mantissa * (double)(1ULL << 63));

    p[0] = (sign << 7) | ((exponent >> 8) & 0x7F);
    p[1] = exponent & 0xFF;
    for (int i = 0; i < 8; i++) {
        p[2 + i] = (mant >> (56 - i * 8)) & 0xFF;
    }
}

// ============================================================================
// Chunked sample conversion + write (gain fused into the SIMD kernels)
// ============================================================================

static uint16_t sanitize_bit_depth(uint16_t bit_depth) {
    return (bit_depth == 8 || bit_depth == 16 || bit_depth == 24 || bit_depth == 32)
               ? bit_depth
               : 16;  // Default to 16-bit for unsupported depths (e.g. 64)
}

static bool write_samples_chunked(std::ofstream& file, const AudioData& audio,
                                  uint16_t bit_depth, float gain, bool big_endian,
                                  bool aiff_signed_8bit) {
    const size_t num_samples = audio.samples.size();
    const int bytes_per_sample = bit_depth / 8;

    constexpr size_t CHUNK_SAMPLES = 1u << 20;  // up to 4MB output per chunk
    std::vector<uint8_t> buf(std::min(num_samples, CHUNK_SAMPLES) * bytes_per_sample);

    size_t done = 0;
    while (done < num_samples) {
        size_t n = std::min(CHUNK_SAMPLES, num_samples - done);
        const float* src = audio.samples.data() + done;
        switch (bit_depth) {
            case 8:
                if (aiff_signed_8bit) simd::convert_float_to_s8(src, buf.data(), n, gain);
                else                  simd::convert_float_to_u8(src, buf.data(), n, gain);
                break;
            case 16:
                simd::convert_float_to_i16(src, buf.data(), n, gain, big_endian);
                break;
            case 24:
                simd::convert_float_to_i24(src, buf.data(), n, gain, big_endian);
                break;
            case 32:
                simd::convert_float_to_i32(src, buf.data(), n, gain, big_endian);
                break;
        }
        file.write(reinterpret_cast<const char*>(buf.data()),
                   static_cast<std::streamsize>(n * bytes_per_sample));
        if (!file.good()) return false;
        done += n;
    }
    return true;
}

// ============================================================================
// Write WAV File (native implementation)
// ============================================================================

static bool write_wav(const std::string& filepath, const AudioData& audio, float gain) {
    std::ofstream file(utf8_path(filepath), std::ios::binary);
    if (!file) return false;

    const uint16_t bit_depth = sanitize_bit_depth(audio.bit_depth);
    const size_t num_samples = audio.samples.size();
    const int bytes_per_sample = bit_depth / 8;
    const uint64_t data_size64 =
        static_cast<uint64_t>(num_samples) * static_cast<uint64_t>(bytes_per_sample);

    // RIFF sizes are 32-bit; refuse to write a silently-corrupt >4GB header.
    if (data_size64 > 0xFFFFFFFFull - 36ull) return false;
    const uint32_t data_size = static_cast<uint32_t>(data_size64);
    const uint32_t file_size = 36 + data_size;  // RIFF size = file size - 8

    uint8_t buf[4];

    // RIFF header
    file.write("RIFF", 4);
    write_u32_le(buf, file_size);
    file.write((char*)buf, 4);
    file.write("WAVE", 4);

    // fmt chunk
    file.write("fmt ", 4);
    write_u32_le(buf, 16);  // fmt chunk size
    file.write((char*)buf, 4);

    write_u16_le(buf, 1);  // audio format: PCM
    file.write((char*)buf, 2);
    write_u16_le(buf, audio.channels);
    file.write((char*)buf, 2);
    write_u32_le(buf, audio.sample_rate);
    file.write((char*)buf, 4);
    uint32_t byte_rate = audio.sample_rate * audio.channels * bytes_per_sample;
    write_u32_le(buf, byte_rate);
    file.write((char*)buf, 4);
    uint16_t block_align = audio.channels * bytes_per_sample;
    write_u16_le(buf, block_align);
    file.write((char*)buf, 2);
    write_u16_le(buf, bit_depth);
    file.write((char*)buf, 2);

    // data chunk
    file.write("data", 4);
    write_u32_le(buf, data_size);
    file.write((char*)buf, 4);

    return write_samples_chunked(file, audio, bit_depth, gain,
                                 /*big_endian=*/false, /*aiff_signed_8bit=*/false) &&
           file.good();
}

// ============================================================================
// Write AIFF File (native implementation)
// ============================================================================

static bool write_aiff(const std::string& filepath, const AudioData& audio, float gain) {
    std::ofstream file(utf8_path(filepath), std::ios::binary);
    if (!file) return false;

    const uint16_t bit_depth = sanitize_bit_depth(audio.bit_depth);
    const size_t num_samples = audio.samples.size();
    const int bytes_per_sample = bit_depth / 8;
    const uint64_t ssnd_data_size64 =
        static_cast<uint64_t>(num_samples) * static_cast<uint64_t>(bytes_per_sample);

    if (ssnd_data_size64 > 0xFFFFFFFFull - 64ull) return false;  // 32-bit chunk sizes
    const size_t ssnd_data_size = static_cast<size_t>(ssnd_data_size64);
    size_t ssnd_chunk_size = ssnd_data_size + 8;  // offset + blockSize + data
    size_t comm_chunk_size = 18;
    size_t form_size = 4 + 8 + comm_chunk_size + 8 + ssnd_chunk_size;

    // FORM header
    file.write("FORM", 4);
    uint8_t buf[10];
    write_u32_be(buf, (uint32_t)form_size);
    file.write((char*)buf, 4);
    file.write("AIFF", 4);

    // COMM chunk
    file.write("COMM", 4);
    write_u32_be(buf, (uint32_t)comm_chunk_size);
    file.write((char*)buf, 4);
    write_u16_be(buf, audio.channels);
    file.write((char*)buf, 2);
    write_u32_be(buf, (uint32_t)audio.total_frames);
    file.write((char*)buf, 4);
    write_u16_be(buf, bit_depth);
    file.write((char*)buf, 2);
    write_extended_be(buf, (double)audio.sample_rate);
    file.write((char*)buf, 10);

    // SSND chunk
    file.write("SSND", 4);
    write_u32_be(buf, (uint32_t)ssnd_chunk_size);
    file.write((char*)buf, 4);
    write_u32_be(buf, 0);  // offset
    file.write((char*)buf, 4);
    write_u32_be(buf, 0);  // blockSize
    file.write((char*)buf, 4);

    return write_samples_chunked(file, audio, bit_depth, gain,
                                 /*big_endian=*/true, /*aiff_signed_8bit=*/true) &&
           file.good();
}

// ============================================================================
// Normalize and Save
// ============================================================================

// Compute only the metric required by the normalization target. Loudness-,
// peak- and RMS-family targets each cost one specialized pass instead of the
// full analysis suite.
static bool measure_target_value(const AudioData& audio, Normalizer::Target target,
                                 double& out_value) {
    switch (target) {
        case Normalizer::Target::Peak: {
            float pk = simd::find_peak_abs(audio.samples.data(), audio.samples.size());
            out_value = linear_to_db(static_cast<double>(pk));
            return true;
        }
        case Normalizer::Target::TruePeak:
            out_value = TruePeakMeter::measure(audio);
            return true;
        case Normalizer::Target::Integrated:
        case Normalizer::Target::ShorttermMax:
        case Normalizer::Target::MomentaryMax: {
            auto r = LoudnessMeter::measure(audio);
            if (target == Normalizer::Target::Integrated) out_value = r.integrated;
            else if (target == Normalizer::Target::ShorttermMax) out_value = r.shortterm_max;
            else out_value = r.momentary_max;
            return true;
        }
        case Normalizer::Target::RMSMin:
        case Normalizer::Target::RMSMax:
        case Normalizer::Target::RMSAverage: {
            auto r = RMSMeter::measure(audio, 50.0);
            if (target == Normalizer::Target::RMSMin) out_value = r.min_db;
            else if (target == Normalizer::Target::RMSMax) out_value = r.max_db;
            else out_value = r.average_db;
            return true;
        }
    }
    return false;
}

bool Normalizer::normalize_and_save(const std::string& input_path,
                                    const std::string& output_path,
                                    Target target,
                                    double target_value) {
    AudioFormat out_format = AudioReader::detect_format(output_path);
    if (out_format == AudioFormat::MP3) {
        // Writing MP3 is not supported; silently emitting WAV bytes into a
        // .mp3 file (the previous behavior for in-place MP3 normalization)
        // destroys the input.
        return false;
    }

    // Load once, then measure only what the target needs.
    auto audio = AudioReader::load(input_path);
    if (!audio || audio->samples.empty() || audio->channels == 0 || audio->sample_rate == 0) {
        return false;
    }

    double current_value = 0.0;
    if (!measure_target_value(*audio, target, current_value)) return false;
    if (!std::isfinite(current_value)) {
        // Silent input (or measurement floor): a finite gain cannot reach the
        // target; the previous code amplified by +inf and wrote NaN samples.
        return false;
    }

    const double gain_db = target_value - current_value;
    const float gain = static_cast<float>(db_to_linear(gain_db));

    switch (out_format) {
        case AudioFormat::AIFF:
            return write_aiff(output_path, *audio, gain);
        case AudioFormat::WAV:
        default:
            return write_wav(output_path, *audio, gain);
    }
}

} // namespace pb_audio
