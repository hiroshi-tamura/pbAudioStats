/*
 * pbAudioReader.cpp - Audio File Reader
 *
 * WAV/AIFF: Native implementation (no external dependencies)
 *   - PCM 8/16/24/32-bit, IEEE float 32/64-bit, WAVE_FORMAT_EXTENSIBLE
 *   - AIFF/AIFC (big-endian PCM and 'sowt' little-endian)
 * MP3: Uses dr_mp3 (header-only, public domain) - MP3 decoding is too complex
 *      for a native implementation.
 *
 * Robustness rules (all enforced in the constructors):
 *   - bits_per_sample must be a supported value (division-by-zero guard)
 *   - declared data sizes are clamped to the actual file size
 *   - unsupported format combinations fail cleanly (no uninitialized output)
 *
 * All file paths are interpreted as UTF-8 and opened via std::filesystem
 * (wide-char on Windows), so Japanese / non-ANSI filenames work correctly.
 */

#include "pbAudioStats.h"
#include "pbSimd.h"
#include <fstream>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <cctype>
#include <algorithm>
#include <limits>

// MP3のみ外部ヘッダー使用（MP3デコードは複雑すぎて自前実装不可）
// Note: DR_MP3_IMPLEMENTATION is defined via CMakeLists.txt
#include "dr_mp3.h"

namespace fs = std::filesystem;

namespace pb_audio {

// ============================================================================
// UTF-8 path helpers (Windows: wide path; others: as-is)
// ============================================================================

static fs::path utf8_path(const std::string& utf8) {
#if defined(_WIN32)
    // Interpret the incoming std::string as UTF-8 (the CLI converts argv
    // accordingly); fs::u8path performs the UTF-8 -> native conversion.
    return fs::u8path(utf8);
#else
    return fs::path(utf8);
#endif
}

static std::ifstream open_binary(const std::string& utf8) {
    return std::ifstream(utf8_path(utf8), std::ios::binary);
}

static uint64_t file_size_of(std::ifstream& file) {
    auto pos = file.tellg();
    file.seekg(0, std::ios::end);
    auto end = file.tellg();
    file.clear();
    file.seekg(pos, std::ios::beg);
    return end > 0 ? static_cast<uint64_t>(end) : 0;
}

// ============================================================================
// Utility functions for reading binary data
// ============================================================================

// Big-endian readers (for AIFF)
static uint16_t read_u16_be(const uint8_t* p) {
    return (p[0] << 8) | p[1];
}

static uint32_t read_u32_be(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
}

// Little-endian readers (for WAV)
static uint16_t read_u16_le(const uint8_t* p) {
    return p[0] | (p[1] << 8);
}

static uint32_t read_u32_le(const uint8_t* p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24);
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

    double value = (double)mantissa / (double)(1ULL << 63);
    value = ldexp(value, exponent - 16383);
    return sign ? -value : value;
}

namespace {

constexpr uint16_t WAVE_FORMAT_PCM = 0x0001;
constexpr uint16_t WAVE_FORMAT_IEEE_FLOAT = 0x0003;
constexpr uint16_t WAVE_FORMAT_EXTENSIBLE = 0xFFFE;

class WavStream final : public AudioStream {
public:
    explicit WavStream(const std::string& filepath) : file(open_binary(filepath)) {
        if (!file) {
            return;
        }
        const uint64_t file_size = file_size_of(file);

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
        uint64_t data_size = 0;

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
                uint8_t fmt[40] = {};
                size_t to_read = std::min<size_t>(chunk_size, sizeof(fmt));
                file.read(reinterpret_cast<char*>(fmt), to_read);
                if (file.gcount() != static_cast<std::streamsize>(to_read)) return;

                audio_format = read_u16_le(fmt);
                channels = read_u16_le(fmt + 2);
                sample_rate = read_u32_le(fmt + 4);
                bits_per_sample = read_u16_le(fmt + 14);

                if (audio_format == WAVE_FORMAT_EXTENSIBLE) {
                    // Resolve the real format from the SubFormat GUID
                    // (first two bytes hold the equivalent format tag).
                    if (chunk_size < 40) return;
                    audio_format = read_u16_le(fmt + 24);
                }

                if (audio_format != WAVE_FORMAT_PCM && audio_format != WAVE_FORMAT_IEEE_FLOAT) {
                    return;
                }

                file.clear();
                file.seekg(chunk_data_pos + static_cast<std::streamoff>(chunk_size), std::ios::beg);
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

        // Supported bit depths only (also guards the division below).
        if (audio_format == WAVE_FORMAT_PCM) {
            if (bits_per_sample != 8 && bits_per_sample != 16 &&
                bits_per_sample != 24 && bits_per_sample != 32) {
                return;
            }
        } else {  // IEEE float
            if (bits_per_sample != 32 && bits_per_sample != 64) {
                return;
            }
        }

        // Clamp the declared data size to what the file actually contains
        // (handles truncated files and 0xFFFFFFFF streaming placeholders).
        if (data_offset > file_size) return;
        uint64_t available = file_size - data_offset;
        if (data_size > available) data_size = available;

        int bytes_per_sample = bits_per_sample / 8;
        uint64_t total_frames = data_size / (static_cast<uint64_t>(channels) * bytes_per_sample);

        info.sample_rate = sample_rate;
        info.channels = channels;
        info.bit_depth = bits_per_sample;
        info.total_frames = total_frames;

        audio_format_ = audio_format;
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

        // 32-bit float: identical layout — read straight into the destination
        // (avoids a full staging-buffer copy).
        if (audio_format_ == WAVE_FORMAT_IEEE_FLOAT && info.bit_depth == 32) {
            file.read(reinterpret_cast<char*>(buffer), bytes_to_read);
            size_t frames_read = static_cast<size_t>(file.gcount()) / bytes_per_frame_;
            frames_remaining_ -= frames_read;
            return frames_read;
        }

        if (io_buffer_.size() < bytes_to_read) io_buffer_.resize(bytes_to_read);
        file.read(reinterpret_cast<char*>(io_buffer_.data()), bytes_to_read);
        size_t bytes_read = static_cast<size_t>(file.gcount());
        size_t frames_read = bytes_read / bytes_per_frame_;
        if (frames_read == 0) return 0;

        const size_t total_samples = frames_read * info.channels;
        const uint8_t* src = io_buffer_.data();

        if (audio_format_ == WAVE_FORMAT_IEEE_FLOAT) {
            // 64-bit float
            simd::convert_f64_to_f32(reinterpret_cast<const double*>(src), buffer, total_samples);
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
    size_t bytes_per_frame_ = 0;
    uint64_t frames_remaining_ = 0;
    std::vector<uint8_t> io_buffer_;
};

class AiffStream final : public AudioStream {
public:
    explicit AiffStream(const std::string& filepath) : file(open_binary(filepath)) {
        if (!file) {
            return;
        }
        const uint64_t file_size = file_size_of(file);

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
        uint64_t data_size = 0;
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

                if (is_aifc) {
                    // AIFC requires a compression type; only uncompressed PCM
                    // ('NONE'/'none') and byte-swapped PCM ('sowt') are
                    // supported. Anything else (fl32, ima4, ...) fails cleanly.
                    if (chunk_size < 22) return;
                    char comp[5] = {0};
                    memcpy(comp, comm + 18, 4);
                    if (strcmp(comp, "sowt") == 0) {
                        is_little_endian = true;
                    } else if (strcmp(comp, "NONE") != 0 && strcmp(comp, "none") != 0) {
                        return;
                    }
                }

                file.clear();
                file.seekg(chunk_data_pos + static_cast<std::streamoff>(chunk_size), std::ios::beg);
            } else if (memcmp(chunk_id, "SSND", 4) == 0) {
                if (chunk_size < 8) return;
                uint8_t ssnd[8] = {};
                file.read(reinterpret_cast<char*>(ssnd), sizeof(ssnd));
                if (file.gcount() != sizeof(ssnd)) return;
                uint32_t offset = read_u32_be(ssnd);
                if (offset > chunk_size - 8) return;  // would underflow data_size
                data_offset = static_cast<uint64_t>(chunk_data_pos) + 8 + offset;
                data_size = static_cast<uint64_t>(chunk_size) - 8 - offset;
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

        if (!data_offset || channels == 0) {
            return;
        }

        // Sample rate sanity (the 80-bit float field is attacker-controlled;
        // an out-of-range double -> uint32 cast is undefined behavior).
        if (!(sample_rate >= 1.0 && sample_rate <= 10000000.0)) {
            return;
        }

        if (bits_per_sample != 8 && bits_per_sample != 16 &&
            bits_per_sample != 24 && bits_per_sample != 32) {
            return;
        }

        if (data_offset > file_size) return;
        uint64_t available = file_size - data_offset;
        if (data_size > available) data_size = available;

        const uint64_t bytes_per_frame = static_cast<uint64_t>(channels) * (bits_per_sample / 8);
        uint64_t frames_in_data = data_size / bytes_per_frame;
        uint64_t total_frames = num_frames ? std::min<uint64_t>(num_frames, frames_in_data)
                                           : frames_in_data;

        info.sample_rate = static_cast<uint32_t>(sample_rate);
        info.channels = channels;
        info.bit_depth = bits_per_sample;
        info.total_frames = total_frames;

        bytes_per_frame_ = static_cast<size_t>(bytes_per_frame);
        frames_remaining_ = total_frames;
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
    size_t bytes_per_frame_ = 0;
    uint64_t frames_remaining_ = 0;
    std::vector<uint8_t> io_buffer_;
};

// ----------------------------------------------------------------------------
// Fast MP3 frame-count estimation (Xing/Info/VBRI header or bitrate estimate).
// Avoids drmp3_get_pcm_frame_count(), which decodes/scans the entire file
// before any real work (2x file I/O for streamed MP3 analysis). The result is
// used only for duration display; the actual analysis is exact regardless.
// ----------------------------------------------------------------------------

static uint64_t estimate_mp3_total_frames(const std::string& filepath,
                                          uint32_t decoded_sample_rate) {
    std::ifstream f = open_binary(filepath);
    if (!f) return 0;
    const uint64_t file_size = file_size_of(f);

    uint8_t head[16] = {};
    f.read(reinterpret_cast<char*>(head), 10);
    if (f.gcount() != 10) return 0;

    uint64_t offset = 0;
    if (memcmp(head, "ID3", 3) == 0) {
        // ID3v2: syncsafe 28-bit size + 10-byte header
        offset = 10 + ((static_cast<uint64_t>(head[6] & 0x7F) << 21) |
                       ((head[7] & 0x7F) << 14) |
                       ((head[8] & 0x7F) << 7) |
                       (head[9] & 0x7F));
    }

    // Scan up to 64KB for the first MPEG frame sync.
    constexpr size_t SCAN_MAX = 65536;
    std::vector<uint8_t> buf(SCAN_MAX);
    f.clear();
    f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    f.read(reinterpret_cast<char*>(buf.data()), SCAN_MAX);
    size_t got = static_cast<size_t>(f.gcount());
    if (got < 4) return 0;

    static const int BITRATE_V1L3[16] = {0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0};
    static const int BITRATE_V2L3[16] = {0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0};
    static const int SAMPLE_RATES[4][4] = {
        {11025, 12000, 8000, 0},   // MPEG2.5
        {0, 0, 0, 0},
        {22050, 24000, 16000, 0},  // MPEG2
        {44100, 48000, 32000, 0},  // MPEG1
    };

    for (size_t i = 0; i + 4 <= got; ++i) {
        const uint8_t* h = buf.data() + i;
        if (h[0] != 0xFF || (h[1] & 0xE0) != 0xE0) continue;
        int version = (h[1] >> 3) & 0x3;   // 0=2.5, 2=2, 3=1
        int layer = (h[1] >> 1) & 0x3;     // 1=Layer3
        if (version == 1 || layer != 1) continue;
        int bitrate_idx = (h[2] >> 4) & 0xF;
        int sr_idx = (h[2] >> 2) & 0x3;
        int sr = SAMPLE_RATES[version][sr_idx];
        if (sr == 0 || bitrate_idx == 0 || bitrate_idx == 15) continue;

        bool is_v1 = (version == 3);
        int channel_mode = (h[3] >> 6) & 0x3;
        bool mono = (channel_mode == 3);
        uint32_t spf = is_v1 ? 1152 : 576;  // samples per frame, Layer III

        // Xing/Info header offset (after the 4-byte frame header)
        size_t side_info = is_v1 ? (mono ? 17 : 32) : (mono ? 9 : 17);
        size_t xing_off = i + 4 + side_info;
        if (xing_off + 16 <= got) {
            const uint8_t* x = buf.data() + xing_off;
            if (memcmp(x, "Xing", 4) == 0 || memcmp(x, "Info", 4) == 0) {
                uint32_t flags = read_u32_be(x + 4);
                if (flags & 0x1) {  // FRAMES field present
                    uint32_t mpeg_frames = read_u32_be(x + 8);
                    return static_cast<uint64_t>(mpeg_frames) * spf;
                }
            }
        }
        // VBRI header (fixed 32-byte offset after the frame header)
        size_t vbri_off = i + 4 + 32;
        if (vbri_off + 18 <= got) {
            const uint8_t* v = buf.data() + vbri_off;
            if (memcmp(v, "VBRI", 4) == 0) {
                uint32_t mpeg_frames = read_u32_be(v + 14);
                return static_cast<uint64_t>(mpeg_frames) * spf;
            }
        }

        // CBR fallback: estimate from the first frame's bitrate.
        int kbps = is_v1 ? BITRATE_V1L3[bitrate_idx] : BITRATE_V2L3[bitrate_idx];
        if (kbps > 0 && file_size > offset) {
            double seconds = static_cast<double>(file_size - offset) * 8.0 / (kbps * 1000.0);
            uint32_t rate = decoded_sample_rate ? decoded_sample_rate : static_cast<uint32_t>(sr);
            return static_cast<uint64_t>(seconds * rate + 0.5);
        }
        break;
    }
    return 0;
}

#if defined(_WIN32)
static std::wstring utf8_to_wide_path(const std::string& utf8) {
    return utf8_path(utf8).wstring();
}
#endif

static bool drmp3_init_path(drmp3* mp3, const std::string& filepath) {
#if defined(_WIN32)
    std::wstring wide = utf8_to_wide_path(filepath);
    return drmp3_init_file_w(mp3, wide.c_str(), nullptr) != 0;
#else
    return drmp3_init_file(mp3, filepath.c_str(), nullptr) != 0;
#endif
}

class Mp3Stream final : public AudioStream {
public:
    explicit Mp3Stream(const std::string& filepath) {
        if (!drmp3_init_path(&mp3_, filepath)) {
            return;
        }
        valid_ = true;
        info.sample_rate = mp3_.sampleRate;
        info.channels = static_cast<uint16_t>(mp3_.channels);
        info.bit_depth = 16;
        // Frame count from Xing/VBRI header (or bitrate estimate): avoids a
        // full pre-scan of the file. Used for duration display only.
        info.total_frames = estimate_mp3_total_frames(filepath, mp3_.sampleRate);
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
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

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
// WAV / AIFF Loaders (streaming-based to avoid a duplicate file-sized buffer)
// ============================================================================

static std::unique_ptr<AudioData> load_via_stream(AudioStream& stream) {
    auto audio = std::make_unique<AudioData>();
    audio->sample_rate = stream.info.sample_rate;
    audio->channels = stream.info.channels;
    audio->bit_depth = stream.info.bit_depth;
    audio->total_frames = stream.info.total_frames;
    if (audio->channels == 0 || audio->total_frames == 0) {
        return audio;
    }
    audio->samples.resize(audio->total_frames * audio->channels);

    constexpr size_t CHUNK_FRAMES = 262144;  // ~1-6MB reads depending on format
    uint64_t total_read = 0;
    while (total_read < audio->total_frames) {
        size_t to_read = static_cast<size_t>(
            std::min<uint64_t>(CHUNK_FRAMES, audio->total_frames - total_read));
        size_t got = stream.read_frames(
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

std::unique_ptr<AudioData> AudioReader::load_wav(const std::string& filepath) {
    auto stream = std::make_unique<WavStream>(filepath);
    if (!stream->valid()) return nullptr;
    return load_via_stream(*stream);
}

std::unique_ptr<AudioData> AudioReader::load_aiff(const std::string& filepath) {
    auto stream = std::make_unique<AiffStream>(filepath);
    if (!stream->valid()) return nullptr;
    return load_via_stream(*stream);
}

// ============================================================================
// MP3 Loader (using dr_mp3)
// MP3のみ外部ヘッダー使用 - MP3デコードはハフマン符号化、MDCT、サブバンド合成等の
// 複雑な処理が必要なため、自前実装は現実的ではない
//
// Decodes in chunks directly into the destination vector: the previous
// drmp3_open_file_and_read_pcm_frames_f32 + memcpy approach allocated and
// wrote the full PCM twice.
// ============================================================================

std::unique_ptr<AudioData> AudioReader::load_mp3(const std::string& filepath) {
    drmp3 mp3;
    if (!drmp3_init_path(&mp3, filepath)) return nullptr;

    auto audio = std::make_unique<AudioData>();
    audio->sample_rate = mp3.sampleRate;
    audio->channels = static_cast<uint16_t>(mp3.channels);
    audio->bit_depth = 16;  // MP3 is typically treated as 16-bit equivalent
    if (audio->channels == 0) {
        drmp3_uninit(&mp3);
        return nullptr;
    }

    const size_t channels = audio->channels;
    // Pre-reserve from the header estimate to avoid reallocation churn.
    uint64_t est_frames = estimate_mp3_total_frames(filepath, mp3.sampleRate);
    if (est_frames > 0) {
        audio->samples.reserve(static_cast<size_t>((est_frames + 4096) * channels));
    }

    constexpr size_t CHUNK_FRAMES = 65536;
    uint64_t total_frames = 0;
    for (;;) {
        audio->samples.resize((total_frames + CHUNK_FRAMES) * channels);
        drmp3_uint64 got = drmp3_read_pcm_frames_f32(
            &mp3, CHUNK_FRAMES, audio->samples.data() + total_frames * channels);
        total_frames += got;
        if (got < CHUNK_FRAMES) break;
    }
    drmp3_uninit(&mp3);

    audio->samples.resize(static_cast<size_t>(total_frames * channels));
    audio->total_frames = total_frames;
    if (total_frames == 0) return nullptr;

    return audio;
}

} // namespace pb_audio
