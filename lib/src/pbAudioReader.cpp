/*
 * pbAudioReader.cpp - Audio File Reader
 *
 * WAV/AIFF: Native implementation (no external dependencies)
 * MP3: Uses dr_mp3 (header-only, public domain) - MP3 decoding is too complex for native implementation
 */

#include "pbAudioStats.h"
#include "pbSimd.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

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

namespace {

class WavStream final : public AudioStream {
public:
    explicit WavStream(const std::string& filepath) : file(filepath, std::ios::binary) {
        if (!file) {
            return;
        }

        uint8_t header[12] = {};
        file.read(reinterpret_cast<char*>(header), sizeof(header));
        if (file.gcount() != sizeof(header)) return;
        if (memcmp(header, "RIFF", 4) != 0) return;
        if (memcmp(header + 8, "WAVE", 4) != 0) return;

        uint16_t audio_format = 0;
        uint16_t channels = 0;
        uint32_t sample_rate = 0;
        uint16_t bits_per_sample = 0;
        uint64_t data_offset = 0;
        uint32_t data_size = 0;

        while (file) {
            char chunk_id[4] = {};
            uint8_t size_buf[4] = {};
            file.read(chunk_id, 4);
            if (file.gcount() != 4) break;
            file.read(reinterpret_cast<char*>(size_buf), 4);
            if (file.gcount() != 4) break;

            uint32_t chunk_size = read_u32_le(size_buf);
            std::streampos chunk_data_pos = file.tellg();

            if (memcmp(chunk_id, "fmt ", 4) == 0) {
                if (chunk_size < 16) return;
                uint8_t fmt[16] = {};
                file.read(reinterpret_cast<char*>(fmt), sizeof(fmt));
                if (file.gcount() != sizeof(fmt)) return;

                audio_format = read_u16_le(fmt);
                channels = read_u16_le(fmt + 2);
                sample_rate = read_u32_le(fmt + 4);
                bits_per_sample = read_u16_le(fmt + 14);

                if (audio_format != 1 && audio_format != 3) {
                    return;
                }

                if (chunk_size > sizeof(fmt)) {
                    file.seekg(chunk_data_pos + static_cast<std::streamoff>(chunk_size), std::ios::beg);
                }
            } else if (memcmp(chunk_id, "data", 4) == 0) {
                data_offset = static_cast<uint64_t>(chunk_data_pos);
                data_size = chunk_size;
                file.seekg(chunk_data_pos + static_cast<std::streamoff>(chunk_size), std::ios::beg);
            } else {
                file.seekg(chunk_data_pos + static_cast<std::streamoff>(chunk_size), std::ios::beg);
            }

            if (chunk_size & 1) {
                file.seekg(1, std::ios::cur);
            }

            if (data_offset && channels && sample_rate && bits_per_sample) {
                break;
            }
        }

        if (!data_offset || channels == 0 || sample_rate == 0 || bits_per_sample == 0) {
            return;
        }

        int bytes_per_sample = bits_per_sample / 8;
        uint64_t total_frames = data_size / (channels * bytes_per_sample);

        info.sample_rate = sample_rate;
        info.channels = channels;
        info.bit_depth = bits_per_sample;
        info.total_frames = total_frames;

        audio_format_ = audio_format;
        bytes_per_sample_ = bytes_per_sample;
        bytes_per_frame_ = static_cast<size_t>(channels) * bytes_per_sample;
        frames_remaining_ = total_frames;

        file.clear();
        file.seekg(static_cast<std::streamoff>(data_offset), std::ios::beg);
        valid_ = true;
    }

    bool valid() const { return valid_; }

    size_t read_frames(float* buffer, size_t max_frames) override {
        if (!valid_ || frames_remaining_ == 0) return 0;

        size_t frames_to_read = static_cast<size_t>(
            std::min<uint64_t>(frames_remaining_, static_cast<uint64_t>(max_frames)));
        size_t bytes_to_read = frames_to_read * bytes_per_frame_;
        if (bytes_to_read == 0) return 0;

        if (io_buffer_.size() < bytes_to_read) io_buffer_.resize(bytes_to_read);
        file.read(reinterpret_cast<char*>(io_buffer_.data()), bytes_to_read);
        size_t bytes_read = static_cast<size_t>(file.gcount());
        size_t frames_read = bytes_read / bytes_per_frame_;
        if (frames_read == 0) return 0;

        const size_t total_samples = frames_read * info.channels;
        const uint8_t* src = io_buffer_.data();

        if (audio_format_ == 3) {
            if (info.bit_depth == 32) {
                std::memcpy(buffer, src, total_samples * sizeof(float));
            } else if (info.bit_depth == 64) {
                const double* dsrc = reinterpret_cast<const double*>(src);
                for (size_t i = 0; i < total_samples; ++i) {
                    buffer[i] = static_cast<float>(dsrc[i]);
                }
            }
        } else {
            switch (info.bit_depth) {
                case 8:
                    simd::convert_u8_to_float(src, buffer, total_samples);
                    break;
                case 16:
                    simd::convert_i16_le_to_float(src, buffer, total_samples);
                    break;
                case 24:
                    simd::convert_i24_le_to_float(src, buffer, total_samples);
                    break;
                case 32:
                    simd::convert_i32_le_to_float(src, buffer, total_samples);
                    break;
            }
        }

        frames_remaining_ -= frames_read;
        return frames_read;
    }

private:
    std::ifstream file;
    bool valid_ = false;
    uint16_t audio_format_ = 0;
    int bytes_per_sample_ = 0;
    size_t bytes_per_frame_ = 0;
    uint64_t frames_remaining_ = 0;
    std::vector<uint8_t> io_buffer_;
};

class AiffStream final : public AudioStream {
public:
    explicit AiffStream(const std::string& filepath) : file(filepath, std::ios::binary) {
        if (!file) {
            return;
        }

        uint8_t header[12] = {};
        file.read(reinterpret_cast<char*>(header), sizeof(header));
        if (file.gcount() != sizeof(header)) return;
        if (memcmp(header, "FORM", 4) != 0) return;

        bool is_aifc = memcmp(header + 8, "AIFC", 4) == 0;
        bool is_aiff = memcmp(header + 8, "AIFF", 4) == 0;
        if (!is_aiff && !is_aifc) return;

        uint16_t channels = 0;
        uint32_t num_frames = 0;
        uint16_t bits_per_sample = 0;
        double sample_rate = 0.0;
        uint64_t data_offset = 0;
        uint32_t data_size = 0;
        bool is_little_endian = false;

        while (file) {
            char chunk_id[4] = {};
            uint8_t size_buf[4] = {};
            file.read(chunk_id, 4);
            if (file.gcount() != 4) break;
            file.read(reinterpret_cast<char*>(size_buf), 4);
            if (file.gcount() != 4) break;

            uint32_t chunk_size = read_u32_be(size_buf);
            std::streampos chunk_data_pos = file.tellg();

            if (memcmp(chunk_id, "COMM", 4) == 0) {
                if (chunk_size < 18) return;
                uint8_t comm[26] = {};
                size_t to_read = std::min<size_t>(chunk_size, sizeof(comm));
                file.read(reinterpret_cast<char*>(comm), to_read);
                if (file.gcount() != static_cast<std::streamsize>(to_read)) return;

                channels = read_u16_be(comm);
                num_frames = read_u32_be(comm + 2);
                bits_per_sample = read_u16_be(comm + 6);
                sample_rate = read_extended_be(comm + 8);

                if (is_aifc && chunk_size >= 22) {
                    char comp[5] = {0};
                    memcpy(comp, comm + 18, 4);
                    if (strcmp(comp, "sowt") == 0) {
                        is_little_endian = true;
                    } else if (strcmp(comp, "NONE") != 0 && strcmp(comp, "none") != 0) {
                        return;
                    }
                }

                if (chunk_size > to_read) {
                    file.seekg(chunk_data_pos + static_cast<std::streamoff>(chunk_size), std::ios::beg);
                }
            } else if (memcmp(chunk_id, "SSND", 4) == 0) {
                if (chunk_size < 8) return;
                uint8_t ssnd[8] = {};
                file.read(reinterpret_cast<char*>(ssnd), sizeof(ssnd));
                if (file.gcount() != sizeof(ssnd)) return;
                uint32_t offset = read_u32_be(ssnd);
                data_offset = static_cast<uint64_t>(chunk_data_pos) + 8 + offset;
                data_size = chunk_size - 8 - offset;
                file.seekg(chunk_data_pos + static_cast<std::streamoff>(chunk_size), std::ios::beg);
            } else {
                file.seekg(chunk_data_pos + static_cast<std::streamoff>(chunk_size), std::ios::beg);
            }

            if (chunk_size & 1) {
                file.seekg(1, std::ios::cur);
            }

            if (data_offset && channels && sample_rate) {
                break;
            }
        }

        if (!data_offset || channels == 0 || sample_rate == 0) {
            return;
        }

        info.sample_rate = static_cast<uint32_t>(sample_rate);
        info.channels = channels;
        info.bit_depth = bits_per_sample;
        info.total_frames = num_frames ? num_frames : (data_size / (channels * (bits_per_sample / 8)));

        bytes_per_sample_ = bits_per_sample / 8;
        bytes_per_frame_ = static_cast<size_t>(channels) * bytes_per_sample_;
        frames_remaining_ = info.total_frames;
        is_little_endian_ = is_little_endian;

        file.clear();
        file.seekg(static_cast<std::streamoff>(data_offset), std::ios::beg);
        valid_ = true;
    }

    bool valid() const { return valid_; }

    size_t read_frames(float* buffer, size_t max_frames) override {
        if (!valid_ || frames_remaining_ == 0) return 0;

        size_t frames_to_read = static_cast<size_t>(
            std::min<uint64_t>(frames_remaining_, static_cast<uint64_t>(max_frames)));
        size_t bytes_to_read = frames_to_read * bytes_per_frame_;
        if (bytes_to_read == 0) return 0;

        if (io_buffer_.size() < bytes_to_read) io_buffer_.resize(bytes_to_read);
        file.read(reinterpret_cast<char*>(io_buffer_.data()), bytes_to_read);
        size_t bytes_read = static_cast<size_t>(file.gcount());
        size_t frames_read = bytes_read / bytes_per_frame_;
        if (frames_read == 0) return 0;

        const size_t total_samples = frames_read * info.channels;
        const uint8_t* src = io_buffer_.data();

        switch (info.bit_depth) {
            case 8:
                simd::convert_s8_to_float(src, buffer, total_samples);
                break;
            case 16:
                if (is_little_endian_) simd::convert_i16_le_to_float(src, buffer, total_samples);
                else                   simd::convert_i16_be_to_float(src, buffer, total_samples);
                break;
            case 24:
                if (is_little_endian_) simd::convert_i24_le_to_float(src, buffer, total_samples);
                else                   simd::convert_i24_be_to_float(src, buffer, total_samples);
                break;
            case 32:
                if (is_little_endian_) simd::convert_i32_le_to_float(src, buffer, total_samples);
                else                   simd::convert_i32_be_to_float(src, buffer, total_samples);
                break;
        }

        frames_remaining_ -= frames_read;
        return frames_read;
    }

private:
    std::ifstream file;
    bool valid_ = false;
    bool is_little_endian_ = false;
    int bytes_per_sample_ = 0;
    size_t bytes_per_frame_ = 0;
    uint64_t frames_remaining_ = 0;
    std::vector<uint8_t> io_buffer_;
};

class Mp3Stream final : public AudioStream {
public:
    explicit Mp3Stream(const std::string& filepath) {
        if (!drmp3_init_file(&mp3_, filepath.c_str(), nullptr)) {
            return;
        }
        valid_ = true;
        info.sample_rate = mp3_.sampleRate;
        info.channels = static_cast<uint16_t>(mp3_.channels);
        info.bit_depth = 16;
        info.total_frames = drmp3_get_pcm_frame_count(&mp3_);
    }

    ~Mp3Stream() override {
        if (valid_) {
            drmp3_uninit(&mp3_);
        }
    }

    bool valid() const { return valid_; }

    size_t read_frames(float* buffer, size_t max_frames) override {
        if (!valid_) return 0;
        drmp3_uint64 frames = drmp3_read_pcm_frames_f32(&mp3_, static_cast<drmp3_uint64>(max_frames), buffer);
        return static_cast<size_t>(frames);
    }

private:
    drmp3 mp3_ = {};
    bool valid_ = false;
};

} // namespace

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

std::unique_ptr<AudioStream> AudioReader::open_stream(const std::string& filepath) {
    AudioFormat format = detect_format(filepath);

    switch (format) {
        case AudioFormat::WAV: {
            auto stream = std::make_unique<WavStream>(filepath);
            if (!static_cast<WavStream*>(stream.get())->valid()) return nullptr;
            return stream;
        }
        case AudioFormat::AIFF: {
            auto stream = std::make_unique<AiffStream>(filepath);
            if (!static_cast<AiffStream*>(stream.get())->valid()) return nullptr;
            return stream;
        }
        case AudioFormat::MP3: {
            auto stream = std::make_unique<Mp3Stream>(filepath);
            if (!static_cast<Mp3Stream*>(stream.get())->valid()) return nullptr;
            return stream;
        }
        default:
            return nullptr;
    }
}

// ============================================================================
// WAV Loader (streaming-based to avoid a duplicate file-sized buffer)
// ============================================================================

std::unique_ptr<AudioData> AudioReader::load_wav(const std::string& filepath) {
    auto stream = std::make_unique<WavStream>(filepath);
    if (!stream->valid()) return nullptr;

    auto audio = std::make_unique<AudioData>();
    audio->sample_rate = stream->info.sample_rate;
    audio->channels = stream->info.channels;
    audio->bit_depth = stream->info.bit_depth;
    audio->total_frames = stream->info.total_frames;
    if (audio->channels == 0 || audio->total_frames == 0) {
        return audio;
    }
    audio->samples.resize(audio->total_frames * audio->channels);

    constexpr size_t CHUNK_FRAMES = 65536;
    uint64_t total_read = 0;
    while (total_read < audio->total_frames) {
        size_t to_read = static_cast<size_t>(
            std::min<uint64_t>(CHUNK_FRAMES, audio->total_frames - total_read));
        size_t got = stream->read_frames(
            audio->samples.data() + total_read * audio->channels, to_read);
        if (got == 0) break;
        total_read += got;
    }
    if (total_read < audio->total_frames) {
        audio->total_frames = total_read;
        audio->samples.resize(total_read * audio->channels);
    }
    return audio;
}

// ============================================================================
// AIFF Loader (streaming-based to avoid a duplicate file-sized buffer)
// ============================================================================

std::unique_ptr<AudioData> AudioReader::load_aiff(const std::string& filepath) {
    auto stream = std::make_unique<AiffStream>(filepath);
    if (!stream->valid()) return nullptr;

    auto audio = std::make_unique<AudioData>();
    audio->sample_rate = stream->info.sample_rate;
    audio->channels = stream->info.channels;
    audio->bit_depth = stream->info.bit_depth;
    audio->total_frames = stream->info.total_frames;
    if (audio->channels == 0 || audio->total_frames == 0) {
        return audio;
    }
    audio->samples.resize(audio->total_frames * audio->channels);

    constexpr size_t CHUNK_FRAMES = 65536;
    uint64_t total_read = 0;
    while (total_read < audio->total_frames) {
        size_t to_read = static_cast<size_t>(
            std::min<uint64_t>(CHUNK_FRAMES, audio->total_frames - total_read));
        size_t got = stream->read_frames(
            audio->samples.data() + total_read * audio->channels, to_read);
        if (got == 0) break;
        total_read += got;
    }
    if (total_read < audio->total_frames) {
        audio->total_frames = total_read;
        audio->samples.resize(total_read * audio->channels);
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
