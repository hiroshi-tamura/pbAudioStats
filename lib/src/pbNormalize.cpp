/*
 * pbNormalize.cpp - Audio Normalization
 * Supports Peak, True Peak, Integrated Loudness, Short-term, Momentary, RMS
 *
 * WAV/AIFF: Native implementation (no external dependencies)
 * SIMD optimizations: Uses pbSimd.h for accelerated gain application.
 */

#include "pbAudioStats.h"
#include "pbSimd.h"
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

    float linear_gain = static_cast<float>(db_to_linear(gain_db));
    simd::apply_gain_and_clip(audio.samples.data(), audio.samples.size(),
                              linear_gain, -1.0f, 1.0f);
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

    // Convert all samples into a contiguous byte buffer first, then write
    // in a single call. This avoids the per-sample stream overhead of
    // tens of millions of ofstream::write() calls on long files.
    std::vector<uint8_t> sample_buf(static_cast<size_t>(num_samples) * bytes_per_sample);
    uint8_t* dst = sample_buf.data();
    for (size_t i = 0; i < num_samples; i++) {
        float s = audio.samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;

        switch (bit_depth) {
            case 8: {
                *dst++ = static_cast<uint8_t>((s + 1.0f) * 127.5f);
                break;
            }
            case 16: {
                int16_t v = static_cast<int16_t>(s * 32767.0f);
                dst[0] = static_cast<uint8_t>(v & 0xFF);
                dst[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
                dst += 2;
                break;
            }
            case 24: {
                int32_t v = static_cast<int32_t>(s * 8388607.0f);
                dst[0] = static_cast<uint8_t>(v & 0xFF);
                dst[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
                dst[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
                dst += 3;
                break;
            }
            case 32: {
                int32_t v = static_cast<int32_t>(s * 2147483647.0f);
                dst[0] = static_cast<uint8_t>(v & 0xFF);
                dst[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
                dst[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
                dst[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
                dst += 4;
                break;
            }
        }
    }

    file.write(reinterpret_cast<const char*>(sample_buf.data()),
               static_cast<std::streamsize>(sample_buf.size()));
    return file.good();
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

    // Buffer all samples then write at once (see WAV writer above).
    std::vector<uint8_t> sample_buf(static_cast<size_t>(num_samples) * bytes_per_sample);
    uint8_t* dst = sample_buf.data();
    for (size_t i = 0; i < num_samples; i++) {
        float s = audio.samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;

        switch (audio.bit_depth) {
            case 8: {
                *dst++ = static_cast<uint8_t>(static_cast<int8_t>(s * 127.0f));
                break;
            }
            case 16: {
                int16_t v = static_cast<int16_t>(s * 32767.0f);
                dst[0] = static_cast<uint8_t>((v >> 8) & 0xFF);
                dst[1] = static_cast<uint8_t>(v & 0xFF);
                dst += 2;
                break;
            }
            case 24: {
                int32_t v = static_cast<int32_t>(s * 8388607.0f);
                dst[0] = static_cast<uint8_t>((v >> 16) & 0xFF);
                dst[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
                dst[2] = static_cast<uint8_t>(v & 0xFF);
                dst += 3;
                break;
            }
            case 32: {
                int32_t v = static_cast<int32_t>(s * 2147483647.0f);
                dst[0] = static_cast<uint8_t>((v >> 24) & 0xFF);
                dst[1] = static_cast<uint8_t>((v >> 16) & 0xFF);
                dst[2] = static_cast<uint8_t>((v >> 8) & 0xFF);
                dst[3] = static_cast<uint8_t>(v & 0xFF);
                dst += 4;
                break;
            }
        }
    }

    file.write(reinterpret_cast<const char*>(sample_buf.data()),
               static_cast<std::streamsize>(sample_buf.size()));
    return file.good();
}

// ============================================================================
// Normalize and Save
// ============================================================================

bool Normalizer::normalize_and_save(const std::string& input_path,
                                    const std::string& output_path,
                                    Target target,
                                    double target_value) {
    // Load once and analyze the loaded buffer to avoid double I/O.
    auto audio = AudioReader::load(input_path);
    if (!audio) return false;

    AudioStats stats = analyze(*audio, input_path);
    if (!stats.valid) return false;

    double gain = calculate_gain(stats, target, target_value);
    apply_gain(*audio, gain);

    AudioFormat format = AudioReader::detect_format(output_path);
    switch (format) {
        case AudioFormat::WAV:
            return write_wav(output_path, *audio);
        case AudioFormat::AIFF:
            return write_aiff(output_path, *audio);
        default:
            return write_wav(output_path, *audio);
    }
}

} // namespace pb_audio
