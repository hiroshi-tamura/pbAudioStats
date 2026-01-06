/*
 * pb_audio_stats CLI Tool
 * Usage: pb_audio_stats [options] <file_or_directory> [output_file]
 *
 * Supports: WAV, AIFF, MP3
 * Features: BS.1770-4 Loudness, True Peak, RMS, Normalization
 */

#include "pb_audio_stats.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <chrono>

namespace fs = std::filesystem;
using namespace pb_audio;

// ============================================================================
// Configuration
// ============================================================================

struct Config {
    // File info options
    bool show_filename = false;       // -f
    bool show_filename_ext = false;   // -fe
    bool show_filepath = false;       // -fea
    bool show_sample_rate = false;    // -sr
    bool show_bit_depth = false;      // -bt
    bool show_channels = false;       // -ch
    bool show_time = false;           // -tm
    bool show_duration = false;       // -du

    // Loudness options
    bool show_integrated = false;     // -i
    bool show_shortterm = false;      // -s
    bool show_momentary = false;      // -m
    bool show_lra = false;            // -l
    bool show_sample_peak = false;    // -pk (sample peak)
    bool show_true_peak = false;      // -tp
    bool show_rms_min = false;        // -rn
    bool show_rms_max = false;        // -rm
    bool show_rms_avg = false;        // -ra

    // Normalization options
    bool normalize = false;
    Normalizer::Target norm_target;
    double norm_value = 0.0;

    // Other
    std::string input_path;
    std::string output_path;
    bool csv_output = false;
    int num_threads = 0;  // 0 = auto

    bool has_any_output() const {
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
    std::cout << "pb_audio_stats - Audio Statistics and Normalization Tool\n\n";
    std::cout << "Usage: pb_audio_stats [options] <file_or_directory> [output_file]\n\n";

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
    std::cout << "  -tp    True Peak (dBFS)\n";
    std::cout << "  -rn    RMS Minimum (dB)\n";
    std::cout << "  -rm    RMS Maximum (dB)\n";
    std::cout << "  -ra    RMS Average (dB)\n\n";

    std::cout << "Normalization Options (cannot be used with analysis options):\n";
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

    std::cout << "Examples:\n";
    std::cout << "  pb_audio_stats -i -tp input.wav\n";
    std::cout << "  pb_audio_stats -f -fe -i -tp ./audio_folder/ results.csv\n";
    std::cout << "  pb_audio_stats -norm-i:-23.0 input.wav output.wav\n";
}

static bool parse_norm_option(const std::string& arg, Config& config) {
    size_t colon = arg.find(':');
    if (colon == std::string::npos) return false;

    std::string type = arg.substr(0, colon);
    std::string value_str = arg.substr(colon + 1);

    try {
        config.norm_value = std::stod(value_str);
    } catch (...) {
        return false;
    }

    config.normalize = true;

    if (type == "-norm-pk") {
        config.norm_target = Normalizer::Target::Peak;
    } else if (type == "-norm-tp") {
        config.norm_target = Normalizer::Target::TruePeak;
    } else if (type == "-norm-i") {
        config.norm_target = Normalizer::Target::Integrated;
    } else if (type == "-norm-s") {
        config.norm_target = Normalizer::Target::ShorttermMax;
    } else if (type == "-norm-m") {
        config.norm_target = Normalizer::Target::MomentaryMax;
    } else if (type == "-norm-rn") {
        config.norm_target = Normalizer::Target::RMSMin;
    } else if (type == "-norm-rm") {
        config.norm_target = Normalizer::Target::RMSMax;
    } else if (type == "-norm-ra") {
        config.norm_target = Normalizer::Target::RMSAverage;
    } else {
        return false;
    }

    return true;
}

static bool parse_args(int argc, char* argv[], Config& config) {
    std::vector<std::string> positional;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage();
            return false;
        } else if (arg == "-f") {
            config.show_filename = true;
        } else if (arg == "-fe") {
            config.show_filename_ext = true;
        } else if (arg == "-fea") {
            config.show_filepath = true;
        } else if (arg == "-sr") {
            config.show_sample_rate = true;
        } else if (arg == "-bt") {
            config.show_bit_depth = true;
        } else if (arg == "-ch") {
            config.show_channels = true;
        } else if (arg == "-tm") {
            config.show_time = true;
        } else if (arg == "-du") {
            config.show_duration = true;
        } else if (arg == "-i") {
            config.show_integrated = true;
        } else if (arg == "-s") {
            config.show_shortterm = true;
        } else if (arg == "-m") {
            config.show_momentary = true;
        } else if (arg == "-l") {
            config.show_lra = true;
        } else if (arg == "-pk") {
            config.show_sample_peak = true;
        } else if (arg == "-tp") {
            config.show_true_peak = true;
        } else if (arg == "-rn") {
            config.show_rms_min = true;
        } else if (arg == "-rm") {
            config.show_rms_max = true;
        } else if (arg == "-ra") {
            config.show_rms_avg = true;
        } else if (arg.rfind("-norm-", 0) == 0) {
            if (!parse_norm_option(arg, config)) {
                std::cerr << "Error: Invalid normalization option: " << arg << "\n";
                return false;
            }
        } else if (arg.rfind("-j", 0) == 0) {
            if (arg.length() > 2) {
                config.num_threads = std::stoi(arg.substr(2));
            }
        } else if (arg[0] == '-') {
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
        // Check if output is CSV
        if (config.output_path.size() >= 4) {
            std::string ext = config.output_path.substr(config.output_path.size() - 4);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".csv") {
                config.csv_output = true;
            }
        }
    }

    // If no output options specified, show all
    if (!config.normalize && !config.has_any_output()) {
        config.set_all_outputs();
    }

    return true;
}

// ============================================================================
// File Collection
// ============================================================================

static std::vector<std::string> collect_audio_files(const std::string& path) {
    std::vector<std::string> files;

    if (fs::is_regular_file(path)) {
        AudioFormat fmt = AudioReader::detect_format(path);
        if (fmt != AudioFormat::Unknown) {
            files.push_back(fs::absolute(path).string());
        }
    } else if (fs::is_directory(path)) {
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file()) {
                std::string filepath = entry.path().string();
                AudioFormat fmt = AudioReader::detect_format(filepath);
                if (fmt != AudioFormat::Unknown) {
                    files.push_back(fs::absolute(filepath).string());
                }
            }
        }
        // Sort files for consistent output
        std::sort(files.begin(), files.end());
    }

    return files;
}

// ============================================================================
// Output Formatting
// ============================================================================

static std::string format_value(double val, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << val;
    return oss.str();
}

static std::string escape_csv(const std::string& s) {
    if (s.find(',') != std::string::npos || s.find('"') != std::string::npos) {
        std::string escaped = "\"";
        for (char c : s) {
            if (c == '"') escaped += "\"\"";
            else escaped += c;
        }
        escaped += "\"";
        return escaped;
    }
    return "\"" + s + "\"";
}

static void output_header(const Config& config, std::ostream& out) {
    std::vector<std::string> headers;

    if (config.show_filename) headers.push_back("File name (without extension)");
    if (config.show_filename_ext) headers.push_back("File name (with extension)");
    if (config.show_filepath) headers.push_back("Full file path");
    if (config.show_sample_rate) headers.push_back("Sample rate (Hz)");
    if (config.show_bit_depth) headers.push_back("Bit depth (bits)");
    if (config.show_channels) headers.push_back("Number of channels");
    if (config.show_time) headers.push_back("Total time (HH:MM:SS.mmm)");
    if (config.show_duration) headers.push_back("Duration (seconds)");
    if (config.show_integrated) headers.push_back("Integrated Loudness (LUFS)");
    if (config.show_shortterm) headers.push_back("Short-term Loudness Maximum (LUFS)");
    if (config.show_momentary) headers.push_back("Momentary Loudness Maximum (LUFS)");
    if (config.show_lra) headers.push_back("Loudness Range LRA (LU)");
    if (config.show_sample_peak) headers.push_back("Peak (dBFS)");
    if (config.show_true_peak) headers.push_back("True Peak (dBFS)");
    if (config.show_rms_min) headers.push_back("RMS Minimum (dB)");
    if (config.show_rms_max) headers.push_back("RMS Maximum (dB)");
    if (config.show_rms_avg) headers.push_back("RMS Average (dB)");

    for (size_t i = 0; i < headers.size(); i++) {
        if (i > 0) out << ",";
        out << headers[i];
    }
    out << "\n";
}

static void output_stats(const Config& config, const AudioStats& stats, std::ostream& out) {
    std::vector<std::string> values;

    if (config.show_filename) values.push_back(escape_csv(stats.filename));
    if (config.show_filename_ext) values.push_back(escape_csv(stats.filename_ext));
    if (config.show_filepath) values.push_back(escape_csv(stats.filepath));
    if (config.show_sample_rate) values.push_back(std::to_string(stats.sample_rate));
    if (config.show_bit_depth) values.push_back(std::to_string(stats.bit_depth));
    if (config.show_channels) values.push_back(std::to_string(stats.channels));
    if (config.show_time) values.push_back(stats.duration_formatted);
    if (config.show_duration) values.push_back(format_value(stats.duration_seconds, 3));
    if (config.show_integrated) values.push_back(format_value(stats.integrated_loudness, 1));
    if (config.show_shortterm) values.push_back(format_value(stats.shortterm_max, 1));
    if (config.show_momentary) values.push_back(format_value(stats.momentary_max, 1));
    if (config.show_lra) values.push_back(format_value(stats.loudness_range, 1));
    if (config.show_sample_peak) values.push_back(format_value(stats.sample_peak, 1));
    if (config.show_true_peak) values.push_back(format_value(stats.true_peak, 2));
    if (config.show_rms_min) values.push_back(format_value(stats.rms_min, 2));
    if (config.show_rms_max) values.push_back(format_value(stats.rms_max, 2));
    if (config.show_rms_avg) values.push_back(format_value(stats.rms_average, 2));

    for (size_t i = 0; i < values.size(); i++) {
        if (i > 0) out << ",";
        out << values[i];
    }
    out << "\n";
}

// ============================================================================
// Parallel Processing
// ============================================================================

static void process_files_parallel(const std::vector<std::string>& files,
                                   const Config& config,
                                   std::vector<AudioStats>& results) {
    int num_threads = config.num_threads;
    if (num_threads <= 0) {
        num_threads = std::max(1, (int)std::thread::hardware_concurrency());
    }

    results.resize(files.size());
    std::atomic<size_t> next_index(0);
    std::atomic<size_t> completed(0);
    std::mutex progress_mutex;

    auto worker = [&]() {
        while (true) {
            size_t idx = next_index.fetch_add(1);
            if (idx >= files.size()) break;

            results[idx] = analyze(files[idx]);

            size_t done = ++completed;
            if (done % 10 == 0 || done == files.size()) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                std::cerr << "\rProcessing: " << done << "/" << files.size() << " files..." << std::flush;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cerr << "\rProcessing: " << files.size() << "/" << files.size() << " files... Done!\n";
}

// ============================================================================
// Normalization Processing
// ============================================================================

static int process_normalize(const Config& config) {
    std::vector<std::string> files = collect_audio_files(config.input_path);

    if (files.empty()) {
        std::cerr << "Error: No audio files found.\n";
        return 1;
    }

    if (files.size() == 1) {
        // Single file normalization
        std::string output = config.output_path;
        if (output.empty()) {
            // Generate output filename
            fs::path p(files[0]);
            std::string stem = p.stem().string();
            std::string ext = p.extension().string();
            output = p.parent_path().string() + "/" + stem + "_normalized" + ext;
        }

        std::cout << "Normalizing: " << files[0] << "\n";
        std::cout << "  Target: " << config.norm_value << " dB\n";

        if (Normalizer::normalize_and_save(files[0], output, config.norm_target, config.norm_value)) {
            std::cout << "  Output: " << output << "\n";
            std::cout << "Done!\n";
            return 0;
        } else {
            std::cerr << "Error: Failed to normalize file.\n";
            return 1;
        }
    } else {
        // Multiple file normalization
        std::string output_dir = config.output_path;
        if (output_dir.empty()) {
            output_dir = fs::path(config.input_path).string() + "_normalized";
        }

        if (!fs::exists(output_dir)) {
            fs::create_directories(output_dir);
        }

        std::atomic<size_t> success_count(0);
        std::atomic<size_t> fail_count(0);
        std::atomic<size_t> current(0);

        int num_threads = config.num_threads;
        if (num_threads <= 0) {
            num_threads = std::max(1, (int)std::thread::hardware_concurrency());
        }

        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; i++) {
                fs::path p(files[i]);
                std::string output_file = output_dir + "/" + p.filename().string();

                if (Normalizer::normalize_and_save(files[i], output_file,
                                                    config.norm_target, config.norm_value)) {
                    success_count++;
                } else {
                    fail_count++;
                }

                size_t done = ++current;
                if (done % 10 == 0 || done == files.size()) {
                    std::cerr << "\rNormalizing: " << done << "/" << files.size() << " files..." << std::flush;
                }
            }
        };

        std::vector<std::thread> threads;
        size_t chunk = (files.size() + num_threads - 1) / num_threads;
        for (int i = 0; i < num_threads; i++) {
            size_t start = i * chunk;
            size_t end = std::min(start + chunk, files.size());
            if (start < end) {
                threads.emplace_back(worker, start, end);
            }
        }

        for (auto& t : threads) {
            t.join();
        }

        std::cerr << "\rNormalizing: " << files.size() << "/" << files.size() << " files... Done!\n";
        std::cout << "Success: " << success_count << ", Failed: " << fail_count << "\n";
        std::cout << "Output directory: " << output_dir << "\n";

        return (fail_count > 0) ? 1 : 0;
    }
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
    if (!parse_args(argc, argv, config)) {
        return 1;
    }

    // Check input exists
    if (!fs::exists(config.input_path)) {
        std::cerr << "Error: Input path does not exist: " << config.input_path << "\n";
        return 1;
    }

    // Handle normalization mode
    if (config.normalize) {
        return process_normalize(config);
    }

    // Analysis mode
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::string> files = collect_audio_files(config.input_path);

    if (files.empty()) {
        std::cerr << "Error: No audio files found.\n";
        return 1;
    }

    std::cerr << "Found " << files.size() << " audio file(s).\n";

    // Process files
    std::vector<AudioStats> results;
    process_files_parallel(files, config, results);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cerr << "Analysis completed in " << duration.count() << " ms.\n";

    // Output results
    if (config.csv_output && !config.output_path.empty()) {
        std::ofstream out(config.output_path);
        if (!out) {
            std::cerr << "Error: Cannot open output file: " << config.output_path << "\n";
            return 1;
        }
        output_header(config, out);
        for (const auto& stats : results) {
            output_stats(config, stats, out);
        }
        std::cout << "Results written to: " << config.output_path << "\n";
    } else {
        // Output to console
        output_header(config, std::cout);
        for (const auto& stats : results) {
            output_stats(config, stats, std::cout);
        }
    }

    return 0;
}
