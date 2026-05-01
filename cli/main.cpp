/*
 * pbAudioStats CLI Tool
 * Usage: pbAudioStats [options] <file_or_directory> [output_file]
 *
 * Supports: WAV, AIFF, MP3
 * Features: BS.1770-4 Loudness, True Peak, RMS, Normalization
 */

#include "pbAudioStats.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace pb_audio;

// ============================================================================
// Cancellation (Ctrl+C)
// ============================================================================

static std::atomic<bool> g_cancelled{false};
extern "C" void cli_signal_handler(int) {
    g_cancelled.store(true, std::memory_order_relaxed);
}

// ============================================================================
// Configuration
// ============================================================================

struct Config {
    bool show_filename = false;
    bool show_filename_ext = false;
    bool show_filepath = false;
    bool show_sample_rate = false;
    bool show_bit_depth = false;
    bool show_channels = false;
    bool show_time = false;
    bool show_duration = false;

    bool show_integrated = false;
    bool show_shortterm = false;
    bool show_momentary = false;
    bool show_lra = false;
    bool show_sample_peak = false;
    bool show_true_peak = false;
    bool show_rms_min = false;
    bool show_rms_max = false;
    bool show_rms_avg = false;

    bool normalize = false;
    Normalizer::Target norm_target = Normalizer::Target::Peak;
    double norm_value = 0.0;

    std::string input_path;
    std::string output_path;
    bool csv_output = false;
    int num_threads = 0;

    bool has_any_analysis_output() const {
        return show_filename || show_filename_ext || show_filepath ||
               show_sample_rate || show_bit_depth || show_channels ||
               show_time || show_duration ||
               show_integrated || show_shortterm || show_momentary ||
               show_lra || show_sample_peak || show_true_peak ||
               show_rms_min || show_rms_max || show_rms_avg;
    }

    void set_all_outputs() {
        show_filename = true;
        show_filename_ext = true;
        show_filepath = true;
        show_sample_rate = true;
        show_bit_depth = true;
        show_channels = true;
        show_time = true;
        show_duration = true;
        show_integrated = true;
        show_shortterm = true;
        show_momentary = true;
        show_lra = true;
        show_sample_peak = true;
        show_true_peak = true;
        show_rms_min = true;
        show_rms_max = true;
        show_rms_avg = true;
    }
};

// ============================================================================
// Argument Parsing
// ============================================================================

static void print_usage() {
    std::cout << "pbAudioStats - Audio Statistics and Normalization Tool\n\n";
    std::cout << "Usage: pbAudioStats [options] <file_or_directory> [output_file]\n\n";

    std::cout << "File Information Options:\n";
    std::cout << "  -f     File name (without extension)\n";
    std::cout << "  -fe    File name (with extension)\n";
    std::cout << "  -fea   Full file path\n";
    std::cout << "  -sr    Sample rate (Hz)\n";
    std::cout << "  -bt    Bit depth (bits)\n";
    std::cout << "  -ch    Number of channels\n";
    std::cout << "  -tm    Total time (HH:MM:SS.mmm)\n";
    std::cout << "  -du    Duration in seconds\n\n";

    std::cout << "Loudness Analysis Options:\n";
    std::cout << "  -i     Integrated Loudness (LUFS)\n";
    std::cout << "  -s     Short-term Loudness Maximum (LUFS)\n";
    std::cout << "  -m     Momentary Loudness Maximum (LUFS)\n";
    std::cout << "  -l     Loudness Range LRA (LU)\n";
    std::cout << "  -pk    Sample Peak (dBFS)\n";
    std::cout << "  -tp    True Peak (dBTP, ITU-R BS.1770-4 4x oversampling)\n";
    std::cout << "  -rn    RMS Minimum (dB)\n";
    std::cout << "  -rm    RMS Maximum (dB)\n";
    std::cout << "  -ra    RMS Average (dB)\n\n";

    std::cout << "Normalization Options (mutually exclusive with analysis options):\n";
    std::cout << "  -norm-pk:<value>   Normalize to Peak value (dBFS)\n";
    std::cout << "  -norm-tp:<value>   Normalize to True Peak value (dBTP)\n";
    std::cout << "  -norm-i:<value>    Normalize to Integrated Loudness (LUFS)\n";
    std::cout << "  -norm-s:<value>    Normalize to Short-term Max (LUFS)\n";
    std::cout << "  -norm-m:<value>    Normalize to Momentary Max (LUFS)\n";
    std::cout << "  -norm-rn:<value>   Normalize to RMS Min (dB)\n";
    std::cout << "  -norm-rm:<value>   Normalize to RMS Max (dB)\n";
    std::cout << "  -norm-ra:<value>   Normalize to RMS Average (dB)\n\n";

    std::cout << "Other Options:\n";
    std::cout << "  -j<N>  Number of threads for parallel processing (default: auto)\n";
    std::cout << "  -h     Show this help message\n\n";

    std::cout << "If [output_file] ends in .csv the analysis result is written\n";
    std::cout << "as UTF-8 CSV (with BOM) instead of stdout.\n\n";

    std::cout << "Examples:\n";
    std::cout << "  pbAudioStats -i -tp input.wav\n";
    std::cout << "  pbAudioStats -f -fe -i -tp ./audio_folder/ results.csv\n";
    std::cout << "  pbAudioStats -norm-i:-23.0 input.wav output.wav\n";
}

static bool parse_norm_option(const std::string& arg, Config& config) {
    size_t colon = arg.find(':');
    if (colon == std::string::npos) return false;

    std::string type = arg.substr(0, colon);
    std::string value_str = arg.substr(colon + 1);
    if (value_str.empty()) return false;

    size_t pos = 0;
    try {
        config.norm_value = std::stod(value_str, &pos);
    } catch (...) {
        return false;
    }
    if (pos != value_str.size()) return false;  // Reject "-23xx"

    config.normalize = true;

    if      (type == "-norm-pk") config.norm_target = Normalizer::Target::Peak;
    else if (type == "-norm-tp") config.norm_target = Normalizer::Target::TruePeak;
    else if (type == "-norm-i")  config.norm_target = Normalizer::Target::Integrated;
    else if (type == "-norm-s")  config.norm_target = Normalizer::Target::ShorttermMax;
    else if (type == "-norm-m")  config.norm_target = Normalizer::Target::MomentaryMax;
    else if (type == "-norm-rn") config.norm_target = Normalizer::Target::RMSMin;
    else if (type == "-norm-rm") config.norm_target = Normalizer::Target::RMSMax;
    else if (type == "-norm-ra") config.norm_target = Normalizer::Target::RMSAverage;
    else return false;

    return true;
}

static bool parse_args(int argc, char* argv[], Config& config) {
    std::vector<std::string> positional;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage();
            return false;
        } else if (arg == "-f")    config.show_filename = true;
        else if (arg == "-fe")     config.show_filename_ext = true;
        else if (arg == "-fea")    config.show_filepath = true;
        else if (arg == "-sr")     config.show_sample_rate = true;
        else if (arg == "-bt")     config.show_bit_depth = true;
        else if (arg == "-ch")     config.show_channels = true;
        else if (arg == "-tm")     config.show_time = true;
        else if (arg == "-du")     config.show_duration = true;
        else if (arg == "-i")      config.show_integrated = true;
        else if (arg == "-s")      config.show_shortterm = true;
        else if (arg == "-m")      config.show_momentary = true;
        else if (arg == "-l")      config.show_lra = true;
        else if (arg == "-pk")     config.show_sample_peak = true;
        else if (arg == "-tp")     config.show_true_peak = true;
        else if (arg == "-rn")     config.show_rms_min = true;
        else if (arg == "-rm")     config.show_rms_max = true;
        else if (arg == "-ra")     config.show_rms_avg = true;
        else if (arg.rfind("-norm-", 0) == 0) {
            if (!parse_norm_option(arg, config)) {
                std::cerr << "Error: invalid normalization option: " << arg << "\n";
                return false;
            }
        } else if (arg.rfind("-j", 0) == 0) {
            if (arg.length() > 2) {
                try {
                    size_t pos = 0;
                    int v = std::stoi(arg.substr(2), &pos);
                    if (pos != arg.size() - 2 || v < 1) {
                        std::cerr << "Error: -j requires a positive integer (got "
                                  << arg << ").\n";
                        return false;
                    }
                    config.num_threads = v;
                } catch (...) {
                    std::cerr << "Error: invalid -j value: " << arg << "\n";
                    return false;
                }
            }
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            return false;
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.empty()) {
        std::cerr << "Error: No input file or directory specified.\n";
        print_usage();
        return false;
    }

    config.input_path = positional[0];

    if (positional.size() >= 2) {
        config.output_path = positional[1];
        if (config.output_path.size() >= 4) {
            std::string ext = config.output_path.substr(config.output_path.size() - 4);
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".csv") config.csv_output = true;
        }
    }

    if (config.normalize && config.has_any_analysis_output()) {
        std::cerr << "Error: analysis options (-f, -i, -tp, ...) cannot be combined "
                     "with -norm-* options.\n";
        return false;
    }

    if (!config.normalize && !config.has_any_analysis_output()) {
        config.set_all_outputs();
    }

    return true;
}

// ============================================================================
// File Collection (with size for largest-first scheduling)
// ============================================================================

struct FileEntry {
    std::string path;
    uintmax_t size = 0;
};

static std::vector<FileEntry> collect_audio_entries(const std::string& path) {
    std::vector<FileEntry> entries;

    auto try_add = [&entries](const fs::path& p) {
        std::string s = fs::absolute(p).string();
        if (AudioReader::detect_format(s) == AudioFormat::Unknown) return;
        FileEntry e;
        e.path = std::move(s);
        std::error_code ec;
        e.size = fs::file_size(p, ec);
        if (ec) e.size = 0;
        entries.push_back(std::move(e));
    };

    if (fs::is_regular_file(path)) {
        try_add(path);
    } else if (fs::is_directory(path)) {
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file()) try_add(entry.path());
        }
    }

    return entries;
}

// Schedule largest first to keep the tail short on heterogeneous workloads.
static void sort_largest_first(std::vector<FileEntry>& entries) {
    std::sort(entries.begin(), entries.end(),
              [](const FileEntry& a, const FileEntry& b) {
                  if (a.size != b.size) return a.size > b.size;
                  return a.path < b.path;
              });
}

// ============================================================================
// Output Formatting
// ============================================================================

static void format_double(char* buf, size_t cap, double val, int precision) {
    std::snprintf(buf, cap, "%.*f", precision, val);
}

static void append_escaped_csv(std::string& out, const std::string& s) {
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out.push_back(c);
    }
    out.push_back('"');
}

static void output_header(const Config& config, std::ostream& out) {
    bool first = true;
    auto col = [&](const char* name) {
        if (!first) out << ',';
        out << name;
        first = false;
    };

    if (config.show_filename) col("File name (without extension)");
    if (config.show_filename_ext) col("File name (with extension)");
    if (config.show_filepath) col("Full file path");
    if (config.show_sample_rate) col("Sample rate (Hz)");
    if (config.show_bit_depth) col("Bit depth (bits)");
    if (config.show_channels) col("Number of channels");
    if (config.show_time) col("Total time (HH:MM:SS.mmm)");
    if (config.show_duration) col("Duration (seconds)");
    if (config.show_integrated) col("Integrated Loudness (LUFS)");
    if (config.show_shortterm) col("Short-term Loudness Maximum (LUFS)");
    if (config.show_momentary) col("Momentary Loudness Maximum (LUFS)");
    if (config.show_lra) col("Loudness Range LRA (LU)");
    if (config.show_sample_peak) col("Peak (dBFS)");
    if (config.show_true_peak) col("True Peak (dBTP)");
    if (config.show_rms_min) col("RMS Minimum (dB)");
    if (config.show_rms_max) col("RMS Maximum (dB)");
    if (config.show_rms_avg) col("RMS Average (dB)");
    out << '\n';
}

static void output_stats(const Config& config, const AudioStats& stats, std::ostream& out) {
    bool first = true;
    auto comma = [&]() { if (!first) out << ','; first = false; };

    char buf[64];

    if (config.show_filename)     { comma(); std::string s; append_escaped_csv(s, stats.filename);     out << s; }
    if (config.show_filename_ext) { comma(); std::string s; append_escaped_csv(s, stats.filename_ext); out << s; }
    if (config.show_filepath)     { comma(); std::string s; append_escaped_csv(s, stats.filepath);     out << s; }
    if (config.show_sample_rate)  { comma(); out << stats.sample_rate; }
    if (config.show_bit_depth)    { comma(); out << stats.bit_depth; }
    if (config.show_channels)     { comma(); out << stats.channels; }
    if (config.show_time)         { comma(); out << stats.duration_formatted; }
    if (config.show_duration)     { comma(); format_double(buf, sizeof(buf), stats.duration_seconds, 3); out << buf; }
    if (config.show_integrated)   { comma(); format_double(buf, sizeof(buf), stats.integrated_loudness, 1); out << buf; }
    if (config.show_shortterm)    { comma(); format_double(buf, sizeof(buf), stats.shortterm_max, 1); out << buf; }
    if (config.show_momentary)    { comma(); format_double(buf, sizeof(buf), stats.momentary_max, 1); out << buf; }
    if (config.show_lra)          { comma(); format_double(buf, sizeof(buf), stats.loudness_range, 1); out << buf; }
    if (config.show_sample_peak)  { comma(); format_double(buf, sizeof(buf), stats.sample_peak, 1); out << buf; }
    if (config.show_true_peak)    { comma(); format_double(buf, sizeof(buf), stats.true_peak, 2); out << buf; }
    if (config.show_rms_min)      { comma(); format_double(buf, sizeof(buf), stats.rms_min, 2); out << buf; }
    if (config.show_rms_max)      { comma(); format_double(buf, sizeof(buf), stats.rms_max, 2); out << buf; }
    if (config.show_rms_avg)      { comma(); format_double(buf, sizeof(buf), stats.rms_average, 2); out << buf; }
    out << '\n';
}

// ============================================================================
// Parallel Analysis
// ============================================================================

static void process_files_parallel(const std::vector<FileEntry>& entries,
                                   const Config& config,
                                   std::vector<AudioStats>& results) {
    int num_threads = config.num_threads;
    if (num_threads <= 0) {
        num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }
    // Cap analysis threads to avoid I/O thrashing on systems with many cores.
    num_threads = std::min(num_threads, 8);

    const size_t total = entries.size();
    results.resize(total);
    if (total == 0) return;

    const bool use_single_pass = total < 64;
    const size_t print_every = (total < 64) ? std::max<size_t>(1, total)
                                            : std::max<size_t>(1, total / 50);

    std::atomic<size_t> next_index{0};
    std::atomic<size_t> completed{0};
    std::mutex progress_mutex;

    auto worker = [&]() {
        while (!g_cancelled.load(std::memory_order_relaxed)) {
            size_t idx = next_index.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total) break;

            results[idx] = analyze(entries[idx].path, use_single_pass);

            size_t done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
            std::lock_guard<std::mutex> lock(progress_mutex);
            if (done == total || done % print_every == 0) {
                std::cerr << "\rProcessing: " << done << "/" << total
                          << " files..." << std::flush;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(num_threads));
    for (int i = 0; i < num_threads; ++i) threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    std::lock_guard<std::mutex> lock(progress_mutex);
    std::cerr << "\rProcessing: " << total << "/" << total
              << " files... Done!\n";
}

// ============================================================================
// Parallel Normalization
// ============================================================================

static int process_normalize(const Config& config) {
    auto entries = collect_audio_entries(config.input_path);
    if (entries.empty()) {
        std::cerr << "Error: No audio files found.\n";
        return 1;
    }
    sort_largest_first(entries);

    if (entries.size() == 1) {
        const std::string& input = entries[0].path;
        std::string output = config.output_path.empty() ? input : config.output_path;

        std::cout << "Normalizing: " << input << "\n";
        std::cout << "  Target: " << config.norm_value << " dB\n";

        if (!Normalizer::normalize_and_save(input, output,
                                            config.norm_target, config.norm_value)) {
            std::cerr << "Error: Failed to normalize file.\n";
            return 1;
        }
        std::cout << "  Output: " << output << "\n";
        std::cout << "Done!\n";
        return 0;
    }

    std::string output_dir = config.output_path.empty()
        ? fs::path(config.input_path).string()
        : config.output_path;

    if (!fs::exists(output_dir)) {
        std::error_code ec;
        fs::create_directories(output_dir, ec);
        if (ec) {
            std::cerr << "Error: cannot create output directory: " << output_dir << "\n";
            return 1;
        }
    }

    int num_threads = config.num_threads;
    if (num_threads <= 0) {
        num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }
    // Normalization loads full files into memory; cap aggressively to avoid OOM.
    num_threads = std::min(num_threads, 4);

    std::atomic<size_t> next_index{0};
    std::atomic<size_t> completed{0};
    std::atomic<size_t> success_count{0};
    std::atomic<size_t> fail_count{0};
    std::mutex progress_mutex;

    const size_t total = entries.size();

    auto worker = [&]() {
        while (!g_cancelled.load(std::memory_order_relaxed)) {
            size_t idx = next_index.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total) break;

            fs::path p(entries[idx].path);
            std::string output_file = output_dir + "/" + p.filename().string();

            bool ok = Normalizer::normalize_and_save(entries[idx].path, output_file,
                                                     config.norm_target, config.norm_value);
            (ok ? success_count : fail_count).fetch_add(1, std::memory_order_relaxed);

            size_t done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
            std::lock_guard<std::mutex> lock(progress_mutex);
            if (done == total || done % std::max<size_t>(1, total / 50) == 0) {
                std::cerr << "\rNormalizing: " << done << "/" << total
                          << " files..." << std::flush;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(num_threads));
    for (int i = 0; i < num_threads; ++i) threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    std::cerr << "\rNormalizing: " << total << "/" << total
              << " files... Done!\n";
    std::cout << "Success: " << success_count.load()
              << ", Failed: " << fail_count.load() << "\n";
    std::cout << "Output directory: " << output_dir << "\n";

    return (fail_count.load() > 0) ? 1 : 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    Config config;
    if (!parse_args(argc, argv, config)) return 1;

    if (!fs::exists(config.input_path)) {
        std::cerr << "Error: Input path does not exist: " << config.input_path << "\n";
        return 1;
    }

    std::signal(SIGINT, cli_signal_handler);

    if (config.normalize) {
        return process_normalize(config);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    auto entries = collect_audio_entries(config.input_path);
    if (entries.empty()) {
        std::cerr << "Error: No audio files found.\n";
        return 1;
    }
    sort_largest_first(entries);

    std::cerr << "Found " << entries.size() << " audio file(s).\n";

    std::vector<AudioStats> results;
    process_files_parallel(entries, config, results);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cerr << "Analysis completed in " << duration.count() << " ms.\n";

    size_t error_count = 0;
    for (const auto& s : results) if (!s.valid) ++error_count;
    if (error_count > 0) {
        std::cerr << "Warning: " << error_count
                  << " file(s) failed to load and contain placeholder values.\n";
    }

    if (config.csv_output && !config.output_path.empty()) {
        std::ofstream out(config.output_path, std::ios::binary);
        if (!out) {
            std::cerr << "Error: Cannot open output file: " << config.output_path << "\n";
            return 1;
        }
        // UTF-8 BOM for Excel compatibility
        const unsigned char bom[3] = {0xEF, 0xBB, 0xBF};
        out.write(reinterpret_cast<const char*>(bom), 3);

        output_header(config, out);
        for (const auto& stats : results) output_stats(config, stats, out);
        std::cout << "Results written to: " << config.output_path << "\n";
    } else {
        output_header(config, std::cout);
        for (const auto& stats : results) output_stats(config, stats, std::cout);
    }

    return (error_count > 0) ? 2 : 0;
}
