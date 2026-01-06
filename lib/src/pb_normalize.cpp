/*
 * pb_normalize.cpp - Audio Normalization
 * Supports Peak, True Peak, Integrated Loudness, Short-term, Momentary, RMS
 *
 * WAV/AIFF: Native implementation (no external dependencies)
 */

#include "pb_audio_stats.h"
#include <cmath>
#include <fstream>
#include <cstring>
#include <algorithm>

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

    double linear_gain = db_to_linear(gain_db);

    for (size_t i = 0; i < audio.samples.size(); i++) {
        double sample = audio.samples[i] * linear_gain;
        // Clipping prevention
        if (sample > 1.0) sample = 1.0;
        if (sample < -1.0) sample = -1.0;
        audio.samples[i] = (float)sample;
    }
}

// ============================================================================
// Little-endian writers (for WAV)
// ============================================================================

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

// ============================================================================
// Write WAV File (native implementation)
// ============================================================================

static bool write_wav(const std::string& filepath, const AudioData& audio) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;

    uint16_t bit_depth = (audio.bit_depth == 8 || audio.bit_depth == 16 ||
                          audio.bit_depth == 24 || audio.bit_depth == 32) ?
                          audio.bit_depth : 16;  // Default to 16-bit if invalid

    size_t num_samples = audio.total_frames * audio.channels;
    int bytes_per_sample = bit_depth / 8;
    uint32_t data_size = (uint32_t)(num_samples * bytes_per_sample);
    uint32_t file_size = 36 + data_size;  // RIFF size = file size - 8

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

    // Write samples (little-endian)
    for (size_t i = 0; i < num_samples; i++) {
        float s = audio.samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;

        switch (bit_depth) {
            case 8: {
                // 8-bit WAV is unsigned
                uint8_t v = (uint8_t)((s + 1.0f) * 127.5f);
                file.write((char*)&v, 1);
                break;
            }
            case 16: {
                int16_t v = (int16_t)(s * 32767.0f);
                write_u16_le(buf, (uint16_t)v);
                file.write((char*)buf, 2);
                break;
            }
            case 24: {
                int32_t v = (int32_t)(s * 8388607.0f);
                buf[0] = v & 0xFF;
                buf[1] = (v >> 8) & 0xFF;
                buf[2] = (v >> 16) & 0xFF;
                file.write((char*)buf, 3);
                break;
            }
            case 32: {
                int32_t v = (int32_t)(s * 2147483647.0f);
                write_u32_le(buf, (uint32_t)v);
                file.write((char*)buf, 4);
                break;
            }
        }
    }

    return true;
}

// ============================================================================
// Big-endian writers (for AIFF)
// ============================================================================

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

    uint64_t mant = (uint64_t)(mantissa * (1ULL << 63));

    p[0] = (sign << 7) | ((exponent >> 8) & 0x7F);
    p[1] = exponent & 0xFF;
    for (int i = 0; i < 8; i++) {
        p[2 + i] = (mant >> (56 - i * 8)) & 0xFF;
    }
}

// ============================================================================
// Write AIFF File (native implementation)
// ============================================================================

static bool write_aiff(const std::string& filepath, const AudioData& audio) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;

    size_t num_samples = audio.total_frames * audio.channels;
    int bytes_per_sample = audio.bit_depth / 8;
    size_t ssnd_data_size = num_samples * bytes_per_sample;
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
    write_u16_be(buf, audio.bit_depth);
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

    // Write samples (big-endian)
    for (size_t i = 0; i < num_samples; i++) {
        float s = audio.samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;

        switch (audio.bit_depth) {
            case 8: {
                int8_t v = (int8_t)(s * 127.0f);
                file.write((char*)&v, 1);
                break;
            }
            case 16: {
                int16_t v = (int16_t)(s * 32767.0f);
                write_u16_be(buf, (uint16_t)v);
                file.write((char*)buf, 2);
                break;
            }
            case 24: {
                int32_t v = (int32_t)(s * 8388607.0f);
                buf[0] = (v >> 16) & 0xFF;
                buf[1] = (v >> 8) & 0xFF;
                buf[2] = v & 0xFF;
                file.write((char*)buf, 3);
                break;
            }
            case 32: {
                int32_t v = (int32_t)(s * 2147483647.0f);
                write_u32_be(buf, (uint32_t)v);
                file.write((char*)buf, 4);
                break;
            }
        }
    }

    return true;
}

// ============================================================================
// Normalize and Save
// ============================================================================

bool Normalizer::normalize_and_save(const std::string& input_path,
                                    const std::string& output_path,
                                    Target target,
                                    double target_value) {
    // Load audio
    auto audio = AudioReader::load(input_path);
    if (!audio) return false;

    // Analyze current levels
    AudioStats stats = analyze(input_path);

    // Calculate gain
    double gain = calculate_gain(stats, target, target_value);

    // Apply gain
    apply_gain(*audio, gain);

    // Detect output format
    AudioFormat format = AudioReader::detect_format(output_path);

    switch (format) {
        case AudioFormat::WAV:
            return write_wav(output_path, *audio);
        case AudioFormat::AIFF:
            return write_aiff(output_path, *audio);
        default:
            // Default to WAV if format unknown
            return write_wav(output_path, *audio);
    }
}

} // namespace pb_audio
