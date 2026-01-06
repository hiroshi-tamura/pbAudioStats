/*
 * pb_audio_reader.cpp - Audio File Reader
 *
 * WAV/AIFF: Native implementation (no external dependencies)
 * MP3: Uses dr_mp3 (header-only, public domain) - MP3 decoding is too complex for native implementation
 */

#include "pb_audio_stats.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>

// MP3のみ外部ヘッダー使用（MP3デコードは複雑すぎて自前実装不可）
// Note: DR_MP3_IMPLEMENTATION is defined via CMakeLists.txt
#include "dr_mp3.h"

namespace pb_audio {

// ============================================================================
// Utility functions for reading binary data
// ============================================================================

// Big-endian readers (for AIFF)
static uint16_t read_u16_be(const uint8_t* p) {
    return (p[0] << 8) | p[1];
}

static uint32_t read_u32_be(const uint8_t* p) {
    return (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
}

static int16_t read_i16_be(const uint8_t* p) {
    return (int16_t)read_u16_be(p);
}

static int32_t read_i24_be(const uint8_t* p) {
    int32_t v = (p[0] << 16) | (p[1] << 8) | p[2];
    if (v & 0x800000) v |= 0xFF000000;
    return v;
}

static int32_t read_i32_be(const uint8_t* p) {
    return (int32_t)read_u32_be(p);
}

// Little-endian readers (for WAV)
static uint16_t read_u16_le(const uint8_t* p) {
    return p[0] | (p[1] << 8);
}

static uint32_t read_u32_le(const uint8_t* p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
}

static int16_t read_i16_le(const uint8_t* p) {
    return (int16_t)read_u16_le(p);
}

static int32_t read_i24_le(const uint8_t* p) {
    int32_t v = p[0] | (p[1] << 8) | (p[2] << 16);
    if (v & 0x800000) v |= 0xFF000000;
    return v;
}

static int32_t read_i32_le(const uint8_t* p) {
    return (int32_t)read_u32_le(p);
}

// IEEE 754 extended precision (80-bit) to double (for AIFF sample rate)
static double read_extended_be(const uint8_t* p) {
    int sign = (p[0] >> 7) & 1;
    int exponent = ((p[0] & 0x7F) << 8) | p[1];
    uint64_t mantissa = 0;
    for (int i = 0; i < 8; i++) {
        mantissa = (mantissa << 8) | p[2 + i];
    }

    if (exponent == 0 && mantissa == 0) return 0.0;
    if (exponent == 0x7FFF) return sign ? -INFINITY : INFINITY;

    double value = (double)mantissa / (1ULL << 63);
    value = ldexp(value, exponent - 16383);
    return sign ? -value : value;
}

// ============================================================================
// Format Detection
// ============================================================================

AudioFormat AudioReader::detect_format(const std::string& filepath) {
    size_t dot = filepath.rfind('.');
    if (dot == std::string::npos) return AudioFormat::Unknown;

    std::string ext = filepath.substr(dot);
    for (char& c : ext) c = std::tolower(c);

    if (ext == ".wav") return AudioFormat::WAV;
    if (ext == ".aiff" || ext == ".aif") return AudioFormat::AIFF;
    if (ext == ".mp3") return AudioFormat::MP3;

    return AudioFormat::Unknown;
}

// ============================================================================
// Main Load Function
// ============================================================================

std::unique_ptr<AudioData> AudioReader::load(const std::string& filepath) {
    AudioFormat format = detect_format(filepath);

    switch (format) {
        case AudioFormat::WAV:
            return load_wav(filepath);
        case AudioFormat::AIFF:
            return load_aiff(filepath);
        case AudioFormat::MP3:
            return load_mp3(filepath);
        default:
            return nullptr;
    }
}

// ============================================================================
// WAV Loader (native implementation)
// ============================================================================

std::unique_ptr<AudioData> AudioReader::load_wav(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return nullptr;

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size < 44) return nullptr;  // Minimum WAV header size

    std::vector<uint8_t> buffer(file_size);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    // Check RIFF header
    if (memcmp(buffer.data(), "RIFF", 4) != 0) return nullptr;
    if (memcmp(buffer.data() + 8, "WAVE", 4) != 0) return nullptr;

    // Parse chunks
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    const uint8_t* data_ptr = nullptr;
    uint32_t data_size = 0;

    size_t pos = 12;  // Skip RIFF header
    while (pos + 8 <= file_size) {
        const char* chunk_id = reinterpret_cast<const char*>(buffer.data() + pos);
        uint32_t chunk_size = read_u32_le(buffer.data() + pos + 4);

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            // Format chunk
            if (chunk_size < 16) return nullptr;
            const uint8_t* fmt = buffer.data() + pos + 8;

            audio_format = read_u16_le(fmt);
            channels = read_u16_le(fmt + 2);
            sample_rate = read_u32_le(fmt + 4);
            // bytes_per_sec at fmt + 8 (skip)
            // block_align at fmt + 12 (skip)
            bits_per_sample = read_u16_le(fmt + 14);

            // Support PCM (1) and IEEE float (3)
            if (audio_format != 1 && audio_format != 3) {
                return nullptr;  // Compressed formats not supported
            }
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            // Data chunk
            data_ptr = buffer.data() + pos + 8;
            data_size = chunk_size;
        }

        // Move to next chunk (chunks are word-aligned)
        pos += 8 + chunk_size;
        if (chunk_size & 1) pos++;
    }

    if (!data_ptr || channels == 0 || sample_rate == 0 || bits_per_sample == 0) {
        return nullptr;
    }

    int bytes_per_sample = bits_per_sample / 8;
    uint64_t total_frames = data_size / (channels * bytes_per_sample);

    auto audio = std::make_unique<AudioData>();
    audio->sample_rate = sample_rate;
    audio->channels = channels;
    audio->bit_depth = bits_per_sample;
    audio->total_frames = total_frames;
    audio->samples.resize(total_frames * channels);

    // Convert samples to float
    for (uint64_t i = 0; i < total_frames * channels; i++) {
        const uint8_t* p = data_ptr + i * bytes_per_sample;
        float sample = 0.0f;

        if (audio_format == 3) {
            // IEEE float
            if (bits_per_sample == 32) {
                uint32_t v = read_u32_le(p);
                memcpy(&sample, &v, sizeof(float));
            } else if (bits_per_sample == 64) {
                // 64-bit float - read and convert to 32-bit float
                uint64_t v = read_u32_le(p) | ((uint64_t)read_u32_le(p + 4) << 32);
                double d;
                memcpy(&d, &v, sizeof(double));
                sample = (float)d;
            }
        } else {
            // PCM integer
            switch (bits_per_sample) {
                case 8:
                    // 8-bit WAV is unsigned
                    sample = (p[0] - 128) / 128.0f;
                    break;
                case 16:
                    sample = read_i16_le(p) / 32768.0f;
                    break;
                case 24:
                    sample = read_i24_le(p) / 8388608.0f;
                    break;
                case 32:
                    sample = read_i32_le(p) / 2147483648.0f;
                    break;
            }
        }
        audio->samples[i] = sample;
    }

    return audio;
}

// ============================================================================
// AIFF Loader (native implementation)
// ============================================================================

std::unique_ptr<AudioData> AudioReader::load_aiff(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return nullptr;

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(file_size);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    if (file_size < 12) return nullptr;

    // Check FORM header
    if (memcmp(buffer.data(), "FORM", 4) != 0) return nullptr;

    bool is_aifc = memcmp(buffer.data() + 8, "AIFC", 4) == 0;
    bool is_aiff = memcmp(buffer.data() + 8, "AIFF", 4) == 0;
    if (!is_aiff && !is_aifc) return nullptr;

    uint16_t channels = 0;
    uint32_t num_frames = 0;
    uint16_t bits_per_sample = 0;
    double sample_rate = 0;
    const uint8_t* data_ptr = nullptr;
    uint32_t data_size = 0;
    bool is_little_endian = false;

    size_t pos = 12;
    while (pos + 8 <= file_size) {
        const char* chunk_id = reinterpret_cast<const char*>(buffer.data() + pos);
        uint32_t chunk_size = read_u32_be(buffer.data() + pos + 4);

        if (memcmp(chunk_id, "COMM", 4) == 0) {
            const uint8_t* comm = buffer.data() + pos + 8;
            channels = read_u16_be(comm);
            num_frames = read_u32_be(comm + 2);
            bits_per_sample = read_u16_be(comm + 6);
            sample_rate = read_extended_be(comm + 8);

            if (is_aifc && chunk_size >= 22) {
                char comp[5] = {0};
                memcpy(comp, comm + 18, 4);
                if (strcmp(comp, "sowt") == 0) {
                    is_little_endian = true;  // Little endian PCM
                } else if (strcmp(comp, "NONE") != 0 && strcmp(comp, "none") != 0) {
                    return nullptr;  // Compressed, not supported
                }
            }
        } else if (memcmp(chunk_id, "SSND", 4) == 0) {
            uint32_t offset = read_u32_be(buffer.data() + pos + 8);
            data_ptr = buffer.data() + pos + 16 + offset;
            data_size = chunk_size - 8 - offset;
        }

        pos += 8 + chunk_size;
        if (chunk_size & 1) pos++;
    }

    if (!data_ptr || channels == 0 || sample_rate == 0) return nullptr;

    auto audio = std::make_unique<AudioData>();
    audio->sample_rate = (uint32_t)sample_rate;
    audio->channels = channels;
    audio->bit_depth = bits_per_sample;
    audio->total_frames = num_frames;
    audio->samples.resize(num_frames * channels);

    int bytes_per_sample = bits_per_sample / 8;

    for (uint64_t i = 0; i < num_frames * channels; i++) {
        float sample = 0.0f;
        const uint8_t* p = data_ptr + i * bytes_per_sample;

        if (is_little_endian) {
            // Little endian (sowt)
            switch (bits_per_sample) {
                case 8:
                    sample = ((int8_t)p[0]) / 128.0f;
                    break;
                case 16:
                    sample = (int16_t)(p[0] | (p[1] << 8)) / 32768.0f;
                    break;
                case 24: {
                    int32_t v = p[0] | (p[1] << 8) | (p[2] << 16);
                    if (v & 0x800000) v |= 0xFF000000;
                    sample = v / 8388608.0f;
                    break;
                }
                case 32:
                    sample = (int32_t)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24)) / 2147483648.0f;
                    break;
            }
        } else {
            // Big endian (standard AIFF)
            switch (bits_per_sample) {
                case 8:
                    sample = ((int8_t)p[0]) / 128.0f;
                    break;
                case 16:
                    sample = read_i16_be(p) / 32768.0f;
                    break;
                case 24:
                    sample = read_i24_be(p) / 8388608.0f;
                    break;
                case 32:
                    sample = read_i32_be(p) / 2147483648.0f;
                    break;
            }
        }
        audio->samples[i] = sample;
    }

    return audio;
}

// ============================================================================
// MP3 Loader (using dr_mp3)
// MP3のみ外部ヘッダー使用 - MP3デコードはハフマン符号化、MDCT、サブバンド合成等の
// 複雑な処理が必要なため、自前実装は現実的ではない
// ============================================================================

std::unique_ptr<AudioData> AudioReader::load_mp3(const std::string& filepath) {
    drmp3_config config;
    drmp3_uint64 total_frames;

    float* samples = drmp3_open_file_and_read_pcm_frames_f32(
        filepath.c_str(), &config, &total_frames, NULL);

    if (!samples) return nullptr;

    auto audio = std::make_unique<AudioData>();
    audio->sample_rate = config.sampleRate;
    audio->channels = config.channels;
    audio->total_frames = total_frames;
    audio->bit_depth = 16;  // MP3 is typically treated as 16-bit equivalent

    audio->samples.resize(total_frames * config.channels);
    memcpy(audio->samples.data(), samples, total_frames * config.channels * sizeof(float));
    drmp3_free(samples, NULL);

    return audio;
}

} // namespace pb_audio
