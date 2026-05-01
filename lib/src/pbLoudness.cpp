/*
 * pbLoudness.cpp - BS.1770-4 Loudness Measurement
 *
 * Implements:
 * - K-weighting filter (high shelf + high pass biquads)
 * - Momentary loudness (400ms block, 75% overlap)
 * - Short-term loudness (3000ms block, 67% overlap)
 * - Integrated loudness (with -10 LU relative gate)
 * - Loudness Range (LRA) - 10-95 percentile
 * - Sample Peak
 *
 * Reference: ITU-R BS.1770-4, EBU R 128, lib1770-2
 */

#include "pbAudioStats.h"
#include "pbSimd.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>

namespace pb_audio {

// ============================================================================
// Constants
// ============================================================================

static constexpr double SILENCE_THRESHOLD = -70.0;  // Absolute gate threshold (LUFS)
static constexpr double RELATIVE_GATE = -10.0;      // Relative gate (LU below ungated mean)

static constexpr double MOMENTARY_BLOCK_MS = 400.0;
static constexpr int MOMENTARY_PARTITION = 4;       // 75% overlap (4 partitions)

static constexpr double SHORTTERM_BLOCK_MS = 3000.0;
static constexpr int SHORTTERM_PARTITION = 3;       // 67% overlap (3 partitions)

// Histogram settings
static constexpr double HIST_MIN_DB = -70.0;
static constexpr double HIST_MAX_DB = 5.0;
static constexpr int HIST_GRAIN = 100;              // 0.01 dB resolution
static constexpr int HIST_NBINS = static_cast<int>(HIST_GRAIN * (HIST_MAX_DB - HIST_MIN_DB) + 1);

// LRA percentiles
static constexpr double LRA_LOWER_PERCENTILE = 0.10;
static constexpr double LRA_UPPER_PERCENTILE = 0.95;

// ============================================================================
// Utility Functions
// ============================================================================

// Convert linear power to LUFS
static inline double power_to_lufs(double power) {
    if (power <= 0.0) return -std::numeric_limits<double>::infinity();
    return -0.691 + 10.0 * std::log10(power);
}

// Convert LUFS to linear power
static inline double lufs_to_power(double lufs) {
    return std::pow(10.0, 0.1 * (0.691 + lufs));
}

// Denormalize small values to prevent denormal performance issues
static inline double denormalize(double x) {
    return (std::fabs(x) < 1.0e-15) ? 0.0 : x;
}

// ============================================================================
// Biquad Filter State
// ============================================================================

class BiquadFilter {
public:
    double b0, b1, b2;
    double a1, a2;

    // Filter state (per channel)
    struct State {
        double x1 = 0.0, x2 = 0.0;  // Input history
        double y1 = 0.0, y2 = 0.0;  // Output history
    };

    BiquadFilter() : b0(1.0), b1(0.0), b2(0.0), a1(0.0), a2(0.0) {}

    // Process single sample
    double process(double x, State& s) const {
        double y = denormalize(b0 * x + b1 * s.x1 + b2 * s.x2 - a1 * s.y1 - a2 * s.y2);
        s.x2 = s.x1;
        s.x1 = x;
        s.y2 = s.y1;
        s.y1 = y;
        return y;
    }

    void reset(State& s) const {
        s.x1 = s.x2 = s.y1 = s.y2 = 0.0;
    }
};

// ============================================================================
// K-weighting Filter Calculation
//
// Stage 1: High Shelf (shelving filter for head-related effects)
// Stage 2: High Pass (RLB weighting - rumble filter)
//
// Reference coefficients are for 48kHz, requantized for other sample rates.
// ============================================================================

// Reference filter at 48kHz
static void get_reference_high_shelf(BiquadFilter& f) {
    f.b0 = 1.53512485958697;
    f.b1 = -2.69169618940638;
    f.b2 = 1.19839281085285;
    f.a1 = -1.69065929318241;
    f.a2 = 0.73248077421585;
}

static void get_reference_high_pass(BiquadFilter& f) {
    f.b0 = 1.0;
    f.b1 = -2.0;
    f.b2 = 1.0;
    f.a1 = -1.99004745483398;
    f.a2 = 0.99007225036621;
}

// Parametric/analog domain parameters for filter requantization
struct BiquadParams {
    double k;   // Frequency warping factor
    double q;   // Q factor
    double vb;  // Band gain
    double vl;  // Low frequency gain
    double vh;  // High frequency gain
};

// Extract analog parameters from digital biquad coefficients
static void biquad_to_params(const BiquadFilter& f, BiquadParams& p) {
    double x11 = f.a1 - 2.0;
    double x12 = f.a1;
    double x1 = -f.a1 - 2.0;

    double x21 = f.a2 - 1.0;
    double x22 = f.a2 + 1.0;
    double x2 = -f.a2 + 1.0;

    double dx = x22 * x11 - x12 * x21;
    double k_sq = (x22 * x1 - x12 * x2) / dx;
    double k_by_q = (x11 * x2 - x21 * x1) / dx;
    double a0 = 1.0 + k_by_q + k_sq;

    p.k = std::sqrt(k_sq);
    p.q = p.k / k_by_q;
    p.vb = 0.5 * a0 * (f.b0 - f.b2) / k_by_q;
    p.vl = 0.25 * a0 * (f.b0 + f.b1 + f.b2) / k_sq;
    p.vh = 0.25 * a0 * (f.b0 - f.b1 + f.b2);
}

// Requantize filter coefficients for target sample rate
static void requantize_biquad(const BiquadFilter& ref, double ref_rate,
                              BiquadFilter& out, double target_rate) {
    if (ref_rate == target_rate) {
        out = ref;
        return;
    }

    BiquadParams p;
    biquad_to_params(ref, p);

    // Frequency warping
    double k = std::tan((ref_rate / target_rate) * std::atan(p.k));
    double k_sq = k * k;
    double k_by_q = k / p.q;
    double a0 = 1.0 + k_by_q + k_sq;

    out.a1 = denormalize((2.0 * (k_sq - 1.0)) / a0);
    out.a2 = denormalize((1.0 - k_by_q + k_sq) / a0);
    out.b0 = denormalize((p.vh + p.vb * k_by_q + p.vl * k_sq) / a0);
    out.b1 = denormalize((2.0 * (p.vl * k_sq - p.vh)) / a0);
    out.b2 = denormalize((p.vh - p.vb * k_by_q + p.vl * k_sq) / a0);
}

// ============================================================================
// Channel Weights (per BS.1770-4)
// ============================================================================

// G values for channel weighting
// L, R, C: 1.0
// Ls, Rs (surround): 1.41 (~+1.5 dB)
// LFE: excluded (0.0)
static double get_channel_weight(int channel_index, int total_channels) {
    // For stereo: L=0, R=1 -> weight 1.0
    // For 5.1: L=0, R=1, C=2, LFE=3, Ls=4, Rs=5

    if (total_channels <= 2) {
        return 1.0;  // Stereo or mono
    }

    // 5.1 layout
    switch (channel_index) {
        case 0: return 1.0;   // Left
        case 1: return 1.0;   // Right
        case 2: return 1.0;   // Center
        case 3: return 0.0;   // LFE (excluded)
        case 4: return 1.41;  // Left Surround
        case 5: return 1.41;  // Right Surround
        default: return 1.0;
    }
}

// ============================================================================
// Histogram for Integrated Loudness & LRA
// ============================================================================

class LoudnessHistogram {
public:
    struct Bin {
        double db;          // dB value of this bin
        double power_lo;    // Lower power bound
        double power_hi;    // Upper power bound
        uint64_t count;     // Number of blocks in this bin
    };

    std::vector<Bin> bins;

    // First pass statistics (for relative gate calculation)
    double cumulative_power = 0.0;
    uint64_t total_count = 0;

    // Maximum block power
    double max_power = 0.0;

    LoudnessHistogram() {
        bins.resize(HIST_NBINS);
        double step = 1.0 / HIST_GRAIN;

        for (int i = 0; i < HIST_NBINS; ++i) {
            double db = step * i + HIST_MIN_DB;
            double power = lufs_to_power(db);

            bins[i].db = db;
            bins[i].power_lo = power;
            bins[i].power_hi = (i < HIST_NBINS - 1) ? lufs_to_power(db + step) :
                               std::numeric_limits<double>::infinity();
            bins[i].count = 0;
        }
    }

    // Add a block's mean square power to histogram
    void add_block(double power) {
        // Update maximum
        if (power > max_power) {
            max_power = power;
        }

        // Binary search for correct bin
        int lo = 0, hi = HIST_NBINS - 1;
        int found = -1;

        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (power < bins[mid].power_lo) {
                hi = mid - 1;
            } else if (power >= bins[mid].power_hi) {
                lo = mid + 1;
            } else {
                found = mid;
                break;
            }
        }

        if (found >= 0) {
            bins[found].count++;

            // Cumulative moving average for first pass
            ++total_count;
            cumulative_power += (power - cumulative_power) / static_cast<double>(total_count);
        }
    }

    // Get mean loudness above absolute threshold (first pass)
    double get_ungated_mean() const {
        if (total_count == 0) return SILENCE_THRESHOLD;
        return power_to_lufs(cumulative_power);
    }

    // Get integrated loudness with relative gate
    double get_integrated_loudness() const {
        if (total_count == 0) return SILENCE_THRESHOLD;

        // Relative gate: -10 dB below ungated mean
        double gate_power = cumulative_power * std::pow(10.0, 0.1 * RELATIVE_GATE);

        double sum_power = 0.0;
        uint64_t count = 0;

        for (const auto& bin : bins) {
            if (bin.count > 0 && bin.power_lo > gate_power) {
                sum_power += static_cast<double>(bin.count) * bin.power_lo;
                count += bin.count;
            }
        }

        if (count == 0) return SILENCE_THRESHOLD;
        return power_to_lufs(sum_power / static_cast<double>(count));
    }

    // Get loudness range (LRA) - Per EBU R 128 / ITU-R BS.1770-4
    // Uses short-term histogram with -20 dB relative gate and 10-95 percentile
    // Following lib1770 reference implementation exactly
    double get_loudness_range() const {
        if (total_count == 0) return 0.0;

        // Relative gate for LRA: -20 dB below ungated mean (EBU R 128 tech doc)
        // gate = pass1.wmsq * pow(10, 0.1 * gate_db)
        double gate_power = cumulative_power * std::pow(10.0, 0.1 * (-20.0));

        // Count total blocks above gate (first pass)
        uint64_t gated_count = 0;
        for (const auto& bin : bins) {
            // gate < rp->x && 0 < rp->count
            if (gate_power < bin.power_lo && bin.count > 0) {
                gated_count += bin.count;
            }
        }

        if (gated_count == 0) return 0.0;

        // Per EBU R 128: 10th and 95th percentile
        // lower_count = count * lower, upper_count = count * upper
        uint64_t lower_count = static_cast<uint64_t>(gated_count * LRA_LOWER_PERCENTILE);
        uint64_t upper_count = static_cast<uint64_t>(gated_count * LRA_UPPER_PERCENTILE);

        double min_db = std::nan("");
        double max_db = std::nan("");

        uint64_t count = 0;
        uint64_t prev_count = static_cast<uint64_t>(-1);  // Same as lib1770: -1 cast to unsigned

        // Second pass: find percentiles
        for (const auto& bin : bins) {
            // Only include loudness levels above gate threshold
            // gate < rp->x (note: we don't check count here, same as lib1770)
            if (gate_power < bin.power_lo) {
                count += bin.count;

                // Initialize min/max if not done yet
                if (std::isnan(min_db) || std::isnan(max_db)) {
                    min_db = bin.db;
                    max_db = bin.db;
                    prev_count = count;
                    continue;
                }

                // Check for lower percentile crossing
                // prev_count < lower_count && lower_count <= count
                if (prev_count < lower_count && lower_count <= count) {
                    min_db = bin.db;
                }

                // Check for upper percentile crossing
                // prev_count < upper_count && upper_count <= count
                if (prev_count < upper_count && upper_count <= count) {
                    max_db = bin.db;
                    break;
                }

                prev_count = count;
            }
        }

        // Return range, or 0 if calculation failed
        if (std::isnan(min_db) || std::isnan(max_db)) {
            return 0.0;
        }

        return max_db - min_db;
    }

    // Get maximum loudness
    double get_max_loudness() const {
        return power_to_lufs(max_power);
    }
};

// ============================================================================
// Sliding Block Aggregator
//
// Implements overlapping window measurement per BS.1770-4
// ============================================================================

class BlockAggregator {
public:
    double length_ms;
    int partition;        // Number of sub-blocks for overlap

    double sample_rate;
    size_t overlap_size;  // Samples per partition
    size_t block_size;    // Total samples in block
    double scale;         // 1/block_size for averaging

    // Ring buffer for partial sums
    std::vector<double> ring;
    size_t ring_used;
    size_t ring_offset;
    size_t sample_count;

    // Silence gate threshold (power domain)
    double silence_gate;

    // Maximum block power seen
    double max_power;

    BlockAggregator(double rate, double ms, int part)
        : length_ms(ms), partition(part), sample_rate(rate), max_power(0.0) {

        overlap_size = static_cast<size_t>(std::round((ms / 1000.0) * rate / partition));
        block_size = partition * overlap_size;
        scale = 1.0 / static_cast<double>(block_size);

        ring.resize(partition, 0.0);
        ring_used = 1;
        ring_offset = 0;
        sample_count = 0;

        silence_gate = lufs_to_power(SILENCE_THRESHOLD);
    }

    // Add weighted squared sample to current block
    // Returns true if a block was completed, with the block power in out_power
    bool add_sample(double weighted_sq, double& out_power) {
        bool completed = false;
        out_power = 0.0;

        // Accumulate to all active partitions
        if (weighted_sq >= 1.0e-15) {
            double scaled = weighted_sq * scale;
            for (size_t i = 0; i < ring_used; ++i) {
                ring[i] += scaled;
            }
        }

        // Check if overlap period completed
        if (++sample_count >= overlap_size) {
            size_t next_offset = (ring_offset + 1) % partition;

            // If buffer is full, output the oldest block
            if (ring_used == static_cast<size_t>(partition)) {
                double block_power = ring[next_offset];

                // Apply silence gate
                if (block_power > silence_gate) {
                    out_power = block_power;
                    completed = true;

                    if (block_power > max_power) {
                        max_power = block_power;
                    }
                }
            }

            // Reset next partition and advance
            ring[next_offset] = 0.0;
            sample_count = 0;
            ring_offset = next_offset;

            if (ring_used < static_cast<size_t>(partition)) {
                ++ring_used;
            }
        }

        return completed;
    }

    // Add a batch of weighted squared samples to current block.
    void add_samples(const double* weighted_sq, size_t count, LoudnessHistogram& histogram) {
        size_t ring_used_local = ring_used;
        size_t ring_offset_local = ring_offset;
        size_t sample_count_local = sample_count;
        double max_power_local = max_power;
        const size_t partition_size = static_cast<size_t>(partition);

        double* ring_data = ring.data();

        for (size_t i = 0; i < count; ++i) {
            double wsq = weighted_sq[i];
            if (wsq >= 1.0e-15) {
                double scaled = wsq * scale;
                for (size_t r = 0; r < ring_used_local; ++r) {
                    ring_data[r] += scaled;
                }
            }

            if (++sample_count_local >= overlap_size) {
                size_t next_offset = (ring_offset_local + 1) % partition_size;

                if (ring_used_local == partition_size) {
                    double block_power = ring_data[next_offset];
                    if (block_power > silence_gate) {
                        histogram.add_block(block_power);
                        if (block_power > max_power_local) {
                            max_power_local = block_power;
                        }
                    }
                }

                ring_data[next_offset] = 0.0;
                sample_count_local = 0;
                ring_offset_local = next_offset;

                if (ring_used_local < partition_size) {
                    ++ring_used_local;
                }
            }
        }

        ring_used = ring_used_local;
        ring_offset = ring_offset_local;
        sample_count = sample_count_local;
        max_power = max_power_local;
    }

    double get_max_loudness() const {
        return power_to_lufs(max_power);
    }
};

// ============================================================================
// K-weighting Pre-filter
// ============================================================================

class KWeightingFilter {
public:
    BiquadFilter high_shelf;
    BiquadFilter high_pass;

    std::vector<BiquadFilter::State> shelf_states;
    std::vector<BiquadFilter::State> hp_states;

    int channels;
    std::vector<double> channel_weights;

    KWeightingFilter(double sample_rate, int num_channels)
        : channels(num_channels) {

        // Initialize filters from 48kHz reference
        BiquadFilter ref_shelf, ref_hp;
        get_reference_high_shelf(ref_shelf);
        get_reference_high_pass(ref_hp);

        requantize_biquad(ref_shelf, 48000.0, high_shelf, sample_rate);
        requantize_biquad(ref_hp, 48000.0, high_pass, sample_rate);

        // Initialize per-channel state
        shelf_states.resize(num_channels);
        hp_states.resize(num_channels);

        // Initialize channel weights
        channel_weights.resize(num_channels);
        for (int i = 0; i < num_channels; ++i) {
            channel_weights[i] = get_channel_weight(i, num_channels);
        }
    }

    // Process interleaved samples and return weighted sum of squares
    double process_frame(const float* samples) {
        double weighted_sq = 0.0;

        for (int ch = 0; ch < channels; ++ch) {
            if (channel_weights[ch] == 0.0) {
                continue;
            }

            double x = static_cast<double>(samples[ch]);

            // Apply high shelf, then high pass
            double y = high_shelf.process(x, shelf_states[ch]);
            double z = high_pass.process(y, hp_states[ch]);

            weighted_sq += channel_weights[ch] * z * z;
        }

        return weighted_sq;
    }

    // バッチ処理: 複数フレームを一度に処理（4xループアンローリング）
    void process_frames_batch(const float* samples, size_t frame_count, double* weighted_sq_out) {
        if (channels == 2) {
            // ステレオ専用最化パス
            const double w0 = high_shelf.b0, w1 = high_shelf.b1, w2 = high_shelf.b2;
            const double w3 = high_shelf.a1, w4 = high_shelf.a2;
            const double h0 = high_pass.b0, h1 = high_pass.b1, h2 = high_pass.b2;
            const double h3 = high_pass.a1, h4 = high_pass.a2;

            double sx1_0 = shelf_states[0].x1, sx2_0 = shelf_states[0].x2, sy1_0 = shelf_states[0].y1, sy2_0 = shelf_states[0].y2;
            double sx1_1 = shelf_states[1].x1, sx2_1 = shelf_states[1].x2, sy1_1 = shelf_states[1].y1, sy2_1 = shelf_states[1].y2;
            double hx1_0 = hp_states[0].x1, hx2_0 = hp_states[0].x2, hy1_0 = hp_states[0].y1, hy2_0 = hp_states[0].y2;
            double hx1_1 = hp_states[1].x1, hx2_1 = hp_states[1].x2, hy1_1 = hp_states[1].y1, hy2_1 = hp_states[1].y2;

            size_t f = 0;
            while (f + 4 <= frame_count) {
                const float* frame = samples + f * 2;
                double x0, y0, x1, y1, z0, z1;

                x0 = static_cast<double>(frame[0]); y0 = denormalize(w0*x0 + w1*sx1_0 + w2*sx2_0 - w3*sy1_0 - w4*sy2_0);
                sx2_0=sx1_0; sx1_0=x0; sy2_0=sy1_0; sy1_0=y0;
                x1 = static_cast<double>(frame[1]); y1 = denormalize(w0*x1 + w1*sx1_1 + w2*sx2_1 - w3*sy1_1 - w4*sy2_1);
                sx2_1=sx1_1; sx1_1=x1; sy2_1=sy1_1; sy1_1=y1;
                z0 = denormalize(h0*y0 + h1*hx1_0 + h2*hx2_0 - h3*hy1_0 - h4*hy2_0);
                hx2_0=hx1_0; hx1_0=y0; hy2_0=hy1_0; hy1_0=z0;
                z1 = denormalize(h0*y1 + h1*hx1_1 + h2*hx2_1 - h3*hy1_1 - h4*hy2_1);
                hx2_1=hx1_1; hx1_1=y1; hy2_1=hy1_1; hy1_1=z1;
                weighted_sq_out[f] = z0*z0 + z1*z1;

                x0 = static_cast<double>(frame[2]); y0 = denormalize(w0*x0 + w1*sx1_0 + w2*sx2_0 - w3*sy1_0 - w4*sy2_0);
                sx2_0=sx1_0; sx1_0=x0; sy2_0=sy1_0; sy1_0=y0;
                x1 = static_cast<double>(frame[3]); y1 = denormalize(w0*x1 + w1*sx1_1 + w2*sx2_1 - w3*sy1_1 - w4*sy2_1);
                sx2_1=sx1_1; sx1_1=x1; sy2_1=sy1_1; sy1_1=y1;
                z0 = denormalize(h0*y0 + h1*hx1_0 + h2*hx2_0 - h3*hy1_0 - h4*hy2_0);
                hx2_0=hx1_0; hx1_0=y0; hy2_0=hy1_0; hy1_0=z0;
                z1 = denormalize(h0*y1 + h1*hx1_1 + h2*hx2_1 - h3*hy1_1 - h4*hy2_1);
                hx2_1=hx1_1; hx1_1=y1; hy2_1=hy1_1; hy1_1=z1;
                weighted_sq_out[f+1] = z0*z0 + z1*z1;

                x0 = static_cast<double>(frame[4]); y0 = denormalize(w0*x0 + w1*sx1_0 + w2*sx2_0 - w3*sy1_0 - w4*sy2_0);
                sx2_0=sx1_0; sx1_0=x0; sy2_0=sy1_0; sy1_0=y0;
                x1 = static_cast<double>(frame[5]); y1 = denormalize(w0*x1 + w1*sx1_1 + w2*sx2_1 - w3*sy1_1 - w4*sy2_1);
                sx2_1=sx1_1; sx1_1=x1; sy2_1=sy1_1; sy1_1=y1;
                z0 = denormalize(h0*y0 + h1*hx1_0 + h2*hx2_0 - h3*hy1_0 - h4*hy2_0);
                hx2_0=hx1_0; hx1_0=y0; hy2_0=hy1_0; hy1_0=z0;
                z1 = denormalize(h0*y1 + h1*hx1_1 + h2*hx2_1 - h3*hy1_1 - h4*hy2_1);
                hx2_1=hx1_1; hx1_1=y1; hy2_1=hy1_1; hy1_1=z1;
                weighted_sq_out[f+2] = z0*z0 + z1*z1;

                x0 = static_cast<double>(frame[6]); y0 = denormalize(w0*x0 + w1*sx1_0 + w2*sx2_0 - w3*sy1_0 - w4*sy2_0);
                sx2_0=sx1_0; sx1_0=x0; sy2_0=sy1_0; sy1_0=y0;
                x1 = static_cast<double>(frame[7]); y1 = denormalize(w0*x1 + w1*sx1_1 + w2*sx2_1 - w3*sy1_1 - w4*sy2_1);
                sx2_1=sx1_1; sx1_1=x1; sy2_1=sy1_1; sy1_1=y1;
                z0 = denormalize(h0*y0 + h1*hx1_0 + h2*hx2_0 - h3*hy1_0 - h4*hy2_0);
                hx2_0=hx1_0; hx1_0=y0; hy2_0=hy1_0; hy1_0=z0;
                z1 = denormalize(h0*y1 + h1*hx1_1 + h2*hx2_1 - h3*hy1_1 - h4*hy2_1);
                hx2_1=hx1_1; hx1_1=y1; hy2_1=hy1_1; hy1_1=z1;
                weighted_sq_out[f+3] = z0*z0 + z1*z1;

                f += 4;
            }

            while (f < frame_count) {
                const float* frame = samples + f * 2;
                double x0 = static_cast<double>(frame[0]); double y0 = denormalize(w0*x0 + w1*sx1_0 + w2*sx2_0 - w3*sy1_0 - w4*sy2_0);
                sx2_0=sx1_0; sx1_0=x0; sy2_0=sy1_0; sy1_0=y0;
                double x1 = static_cast<double>(frame[1]); double y1 = denormalize(w0*x1 + w1*sx1_1 + w2*sx2_1 - w3*sy1_1 - w4*sy2_1);
                sx2_1=sx1_1; sx1_1=x1; sy2_1=sy1_1; sy1_1=y1;
                double z0 = denormalize(h0*y0 + h1*hx1_0 + h2*hx2_0 - h3*hy1_0 - h4*hy2_0);
                hx2_0=hx1_0; hx1_0=y0; hy2_0=hy1_0; hy1_0=z0;
                double z1 = denormalize(h0*y1 + h1*hx1_1 + h2*hx2_1 - h3*hy1_1 - h4*hy2_1);
                hx2_1=hx1_1; hx1_1=y1; hy2_1=hy1_1; hy1_1=z1;
                weighted_sq_out[f] = z0*z0 + z1*z1;
                f++;
            }

            shelf_states[0].x1 = sx1_0; shelf_states[0].x2 = sx2_0; shelf_states[0].y1 = sy1_0; shelf_states[0].y2 = sy2_0;
            shelf_states[1].x1 = sx1_1; shelf_states[1].x2 = sx2_1; shelf_states[1].y1 = sy1_1; shelf_states[1].y2 = sy2_1;
            hp_states[0].x1 = hx1_0; hp_states[0].x2 = hx2_0; hp_states[0].y1 = hy1_0; hp_states[0].y2 = hy2_0;
            hp_states[1].x1 = hx1_1; hp_states[1].x2 = hx2_1; hp_states[1].y1 = hy1_1; hp_states[1].y2 = hy2_1;
        } else {
            for (size_t f = 0; f < frame_count; ++f) {
                weighted_sq_out[f] = process_frame(samples + f * channels);
            }
        }
    }

    void reset() {
        for (int ch = 0; ch < channels; ++ch) {
            high_shelf.reset(shelf_states[ch]);
            high_pass.reset(hp_states[ch]);
        }
    }
};

// ============================================================================
// Public API: LoudnessMeter::calc_high_shelf / calc_high_pass
// ============================================================================

LoudnessMeter::BiquadCoeffs LoudnessMeter::calc_high_shelf(double sample_rate) {
    BiquadFilter ref, out;
    get_reference_high_shelf(ref);
    requantize_biquad(ref, 48000.0, out, sample_rate);
    return {out.b0, out.b1, out.b2, out.a1, out.a2};
}

LoudnessMeter::BiquadCoeffs LoudnessMeter::calc_high_pass(double sample_rate) {
    BiquadFilter ref, out;
    get_reference_high_pass(ref);
    requantize_biquad(ref, 48000.0, out, sample_rate);
    return {out.b0, out.b1, out.b2, out.a1, out.a2};
}

// ============================================================================
// Helper: Ensure minimum audio length for accurate measurement
//
// Short-term block requires 3000ms, so we loop audio to at least 4 seconds
// to ensure proper BS.1770-4 measurement (same as bs1770gain behavior)
// ============================================================================

static const float* ensure_minimum_length(const std::vector<float>& samples,
                                          size_t& total_frames,
                                          int channels,
                                          uint32_t sample_rate,
                                          std::vector<float>& looped_storage) {
    double duration_ms = (static_cast<double>(total_frames) * 1000.0) / sample_rate;
    constexpr double MIN_DURATION_MS = 4000.0;  // 4 seconds minimum for shortterm measurement

    if (duration_ms >= MIN_DURATION_MS) {
        return samples.data();  // Already long enough, no copy
    }

    size_t original_frames = total_frames;
    size_t min_frames = static_cast<size_t>((MIN_DURATION_MS / 1000.0) * sample_rate);

    // Calculate how many times we need to loop
    size_t loops_needed = (min_frames + original_frames - 1) / original_frames;
    size_t new_total_frames = original_frames * loops_needed;

    looped_storage.resize(new_total_frames * channels);

    // Copy and loop the audio
    for (size_t loop = 0; loop < loops_needed; ++loop) {
        std::memcpy(looped_storage.data() + (loop * original_frames * channels),
                    samples.data(),
                    original_frames * channels * sizeof(float));
    }

    total_frames = new_total_frames;
    return looped_storage.data();
}

// ============================================================================
// Public API: LoudnessMeter::measure
// ============================================================================

LoudnessMeter::Result LoudnessMeter::measure(const AudioData& audio) {
    Result result = {
        SILENCE_THRESHOLD,  // integrated
        SILENCE_THRESHOLD,  // momentary_max
        SILENCE_THRESHOLD,  // shortterm_max
        0.0,                // range
        -std::numeric_limits<double>::infinity()  // sample_peak
    };

    if (audio.samples.empty() || audio.channels == 0 || audio.sample_rate == 0) {
        return result;
    }

    const double sample_rate = static_cast<double>(audio.sample_rate);
    const int channels = audio.channels;

    // Track sample peak from ORIGINAL audio (pre-loop, pre-filter)
    // Use SIMD-optimized peak detection for all interleaved samples
    const size_t total_samples = audio.total_frames * static_cast<size_t>(channels);
    float sample_peak_linear = simd::find_peak_abs(audio.samples.data(), total_samples);

    // Convert sample peak to dBFS
    if (sample_peak_linear > 0.0f) {
        result.sample_peak = 20.0 * std::log10(static_cast<double>(sample_peak_linear));
    } else {
        result.sample_peak = -std::numeric_limits<double>::infinity();
    }

    // Loop short audio to minimum length for accurate BS.1770-4 measurement
    size_t total_frames = audio.total_frames;
    std::vector<float> looped_samples;
    const float* samples = ensure_minimum_length(
        audio.samples, total_frames, channels, audio.sample_rate, looped_samples);

    // Initialize K-weighting filter
    KWeightingFilter kfilter(sample_rate, channels);

    // Initialize block aggregators
    BlockAggregator momentary(sample_rate, MOMENTARY_BLOCK_MS, MOMENTARY_PARTITION);
    BlockAggregator shortterm(sample_rate, SHORTTERM_BLOCK_MS, SHORTTERM_PARTITION);

    // Histogram for integrated loudness (uses momentary blocks)
    LoudnessHistogram momentary_histogram;

    // Separate histogram for LRA (uses short-term blocks)
    LoudnessHistogram shortterm_histogram;

    // バッチ処理化: 4096フレームずつ処理
    constexpr size_t BATCH_SIZE = 4096;
    std::vector<double> weighted_sq_batch(BATCH_SIZE);

    size_t frame = 0;
    while (frame < total_frames) {
        size_t batch_frames = std::min(BATCH_SIZE, total_frames - frame);
        const float* batch_samples = samples + frame * channels;

        if (channels == 2) {
            kfilter.process_frames_batch(batch_samples, batch_frames, weighted_sq_batch.data());
        } else {
            for (size_t i = 0; i < batch_frames; ++i) {
                weighted_sq_batch[i] = kfilter.process_frame(batch_samples + i * channels);
            }
        }

        momentary.add_samples(weighted_sq_batch.data(), batch_frames, momentary_histogram);
        shortterm.add_samples(weighted_sq_batch.data(), batch_frames, shortterm_histogram);

        frame += batch_frames;
    }

    // Get results from histograms and aggregators
    result.integrated = momentary_histogram.get_integrated_loudness();
    result.momentary_max = momentary.get_max_loudness();
    result.shortterm_max = shortterm.get_max_loudness();

    // LRA is calculated from short-term blocks (per EBU R 128)
    result.range = shortterm_histogram.get_loudness_range();

    // Handle edge cases
    if (std::isinf(result.momentary_max) && momentary_histogram.total_count > 0) {
        result.momentary_max = momentary_histogram.get_max_loudness();
    }

    if (std::isinf(result.shortterm_max) && shortterm_histogram.total_count > 0) {
        result.shortterm_max = shortterm_histogram.get_max_loudness();
    }

    // Fallback for edge cases
    if (std::isinf(result.momentary_max)) {
        result.momentary_max = result.integrated;
    }

    if (std::isinf(result.shortterm_max)) {
        result.shortterm_max = result.momentary_max;
    }

    // If LRA is zero or negative and we don't have enough short-term blocks
    if (result.range <= 0.0 && shortterm_histogram.total_count < 2) {
        result.range = 20.0;
    }

    return result;
}

// ============================================================================
// Public API: LoudnessMeter::measure_with_rms
// ============================================================================

LoudnessMeter::ExtendedResult LoudnessMeter::measure_with_rms(const AudioData& audio, double window_ms) {
    ExtendedResult out = {};
    out.loudness = {
        SILENCE_THRESHOLD,
        SILENCE_THRESHOLD,
        SILENCE_THRESHOLD,
        0.0,
        -std::numeric_limits<double>::infinity()
    };
    out.rms_min = -96.0;
    out.rms_max = -96.0;
    out.rms_average = -96.0;

    if (audio.samples.empty() || audio.channels == 0 || audio.sample_rate == 0) {
        return out;
    }

    const double sample_rate = static_cast<double>(audio.sample_rate);
    const int channels = audio.channels;
    const size_t original_frames = audio.total_frames;

    // RMS parameters (SOX compatible)
    const double time_constant = window_ms / 1000.0;
    const double mult = std::exp(-1.0 / (time_constant * sample_rate));
    const double one_minus_mult = 1.0 - mult;
    const uint64_t tc_samples = static_cast<uint64_t>(5.0 * time_constant * sample_rate);

    // Loop short audio to minimum length for accurate BS.1770-4 measurement
    size_t total_frames = audio.total_frames;
    std::vector<float> looped_samples;
    const float* samples = ensure_minimum_length(
        audio.samples, total_frames, channels, audio.sample_rate, looped_samples);

    // Initialize K-weighting filter
    KWeightingFilter kfilter(sample_rate, channels);

    // Initialize block aggregators
    BlockAggregator momentary(sample_rate, MOMENTARY_BLOCK_MS, MOMENTARY_PARTITION);
    BlockAggregator shortterm(sample_rate, SHORTTERM_BLOCK_MS, SHORTTERM_PARTITION);

    // Histogram for integrated loudness (uses momentary blocks)
    LoudnessHistogram momentary_histogram;

    // Separate histogram for LRA (uses short-term blocks)
    LoudnessHistogram shortterm_histogram;

    float sample_peak_linear = 0.0f;

    std::vector<double> avg_sigma_x2(channels, 0.0);
    std::vector<double> max_sigma_x2(channels, 0.0);
    std::vector<double> min_sigma_x2(channels, std::numeric_limits<double>::max());
    std::vector<double> sum_sq(channels, 0.0);

    // Process original frames (RMS + loudness)
    const size_t gate_start = std::min(original_frames, static_cast<size_t>(tc_samples));

    for (size_t frame = 0; frame < gate_start; ++frame) {
        const float* frame_samples = samples + frame * channels;
        for (int ch = 0; ch < channels; ++ch) {
            float s = frame_samples[ch];
            float abs_s = std::fabs(s);
            if (abs_s > sample_peak_linear) {
                sample_peak_linear = abs_s;
            }

            double sample_sq = static_cast<double>(s) * static_cast<double>(s);
            sum_sq[ch] += sample_sq;
            avg_sigma_x2[ch] = avg_sigma_x2[ch] * mult + one_minus_mult * sample_sq;
        }

        double weighted_sq = kfilter.process_frame(frame_samples);

        double block_power;
        if (momentary.add_sample(weighted_sq, block_power)) {
            momentary_histogram.add_block(block_power);
        }

        if (shortterm.add_sample(weighted_sq, block_power)) {
            shortterm_histogram.add_block(block_power);
        }
    }

    for (size_t frame = gate_start; frame < original_frames; ++frame) {
        const float* frame_samples = samples + frame * channels;
        for (int ch = 0; ch < channels; ++ch) {
            float s = frame_samples[ch];
            float abs_s = std::fabs(s);
            if (abs_s > sample_peak_linear) {
                sample_peak_linear = abs_s;
            }

            double sample_sq = static_cast<double>(s) * static_cast<double>(s);
            sum_sq[ch] += sample_sq;
            avg_sigma_x2[ch] = avg_sigma_x2[ch] * mult + one_minus_mult * sample_sq;

            if (avg_sigma_x2[ch] > max_sigma_x2[ch]) {
                max_sigma_x2[ch] = avg_sigma_x2[ch];
            }
            if (avg_sigma_x2[ch] < min_sigma_x2[ch]) {
                min_sigma_x2[ch] = avg_sigma_x2[ch];
            }
        }

        double weighted_sq = kfilter.process_frame(frame_samples);

        double block_power;
        if (momentary.add_sample(weighted_sq, block_power)) {
            momentary_histogram.add_block(block_power);
        }

        if (shortterm.add_sample(weighted_sq, block_power)) {
            shortterm_histogram.add_block(block_power);
        }
    }

    // Process looped frames for loudness only
    for (size_t frame = original_frames; frame < total_frames; ++frame) {
        const float* frame_samples = samples + frame * channels;
        double weighted_sq = kfilter.process_frame(frame_samples);

        double block_power;
        if (momentary.add_sample(weighted_sq, block_power)) {
            momentary_histogram.add_block(block_power);
        }

        if (shortterm.add_sample(weighted_sq, block_power)) {
            shortterm_histogram.add_block(block_power);
        }
    }

    // Loudness results
    out.loudness.integrated = momentary_histogram.get_integrated_loudness();
    out.loudness.momentary_max = momentary.get_max_loudness();
    out.loudness.shortterm_max = shortterm.get_max_loudness();
    out.loudness.range = shortterm_histogram.get_loudness_range();

    if (sample_peak_linear > 0.0f) {
        out.loudness.sample_peak = 20.0 * std::log10(static_cast<double>(sample_peak_linear));
    } else {
        out.loudness.sample_peak = -std::numeric_limits<double>::infinity();
    }

    if (std::isinf(out.loudness.momentary_max) && momentary_histogram.total_count > 0) {
        out.loudness.momentary_max = momentary_histogram.get_max_loudness();
    }

    if (std::isinf(out.loudness.shortterm_max) && shortterm_histogram.total_count > 0) {
        out.loudness.shortterm_max = shortterm_histogram.get_max_loudness();
    }

    if (std::isinf(out.loudness.momentary_max)) {
        out.loudness.momentary_max = out.loudness.integrated;
    }

    if (std::isinf(out.loudness.shortterm_max)) {
        out.loudness.shortterm_max = out.loudness.momentary_max;
    }

    if (out.loudness.range <= 0.0 && shortterm_histogram.total_count < 2) {
        out.loudness.range = 20.0;
    }

    // RMS results (SOX behavior)
    if (original_frames < static_cast<size_t>(tc_samples)) {
        for (int ch = 0; ch < channels; ++ch) {
            double avg_power = sum_sq[ch] / static_cast<double>(original_frames);
            max_sigma_x2[ch] = avg_power;
            min_sigma_x2[ch] = avg_power;
        }
    }

    double total_sum_sq = 0.0;
    const uint64_t total_sample_count =
        static_cast<uint64_t>(original_frames) * static_cast<uint64_t>(channels);

    for (int ch = 0; ch < channels; ++ch) {
        total_sum_sq += sum_sq[ch];
    }

    if (total_sample_count > 0 && total_sum_sq > 0.0) {
        double avg_rms = std::sqrt(total_sum_sq / static_cast<double>(total_sample_count));
        out.rms_average = 20.0 * std::log10(avg_rms);
    }

    double overall_max_sigma_x2 = 0.0;
    for (int ch = 0; ch < channels; ++ch) {
        if (max_sigma_x2[ch] > overall_max_sigma_x2) {
            overall_max_sigma_x2 = max_sigma_x2[ch];
        }
    }
    if (overall_max_sigma_x2 > 0.0) {
        double max_rms = std::sqrt(overall_max_sigma_x2);
        out.rms_max = 20.0 * std::log10(max_rms);
    }

    double overall_min_sigma_x2 = std::numeric_limits<double>::max();
    for (int ch = 0; ch < channels; ++ch) {
        if (min_sigma_x2[ch] < overall_min_sigma_x2) {
            overall_min_sigma_x2 = min_sigma_x2[ch];
        }
    }
    if (overall_min_sigma_x2 > 0.0 && overall_min_sigma_x2 < std::numeric_limits<double>::max()) {
        double min_rms = std::sqrt(overall_min_sigma_x2);
        double min_db = 20.0 * std::log10(min_rms);
        if (std::isinf(min_db)) {
            out.rms_min = -96.0;
        } else {
            out.rms_min = min_db;
        }
    } else {
        out.rms_min = -96.0;
    }

    return out;
}

// ============================================================================
// Public API: LoudnessMeter::measure_stream
// ============================================================================

LoudnessMeter::ExtendedResult LoudnessMeter::measure_stream(AudioStream& stream, double window_ms) {
    ExtendedResult out = {};
    out.loudness = {
        SILENCE_THRESHOLD,
        SILENCE_THRESHOLD,
        SILENCE_THRESHOLD,
        0.0,
        -std::numeric_limits<double>::infinity()
    };
    out.rms_min = -96.0;
    out.rms_max = -96.0;
    out.rms_average = -96.0;

    const auto info = stream.info;
    if (info.channels == 0 || info.sample_rate == 0) {
        return out;
    }

    const double sample_rate = static_cast<double>(info.sample_rate);
    const int channels = info.channels;

    const double time_constant = window_ms / 1000.0;
    const double mult = std::exp(-1.0 / (time_constant * sample_rate));
    const double one_minus_mult = 1.0 - mult;
    const uint64_t tc_samples = static_cast<uint64_t>(5.0 * time_constant * sample_rate);

    float sample_peak_linear = 0.0f;

    std::vector<double> avg_sigma_x2(channels, 0.0);
    std::vector<double> max_sigma_x2(channels, 0.0);
    std::vector<double> min_sigma_x2(channels, std::numeric_limits<double>::max());
    std::vector<double> sum_sq(channels, 0.0);

    uint64_t total_frames = 0;

    KWeightingFilter kfilter(sample_rate, channels);
    BlockAggregator momentary(sample_rate, MOMENTARY_BLOCK_MS, MOMENTARY_PARTITION);
    BlockAggregator shortterm(sample_rate, SHORTTERM_BLOCK_MS, SHORTTERM_PARTITION);
    LoudnessHistogram momentary_histogram;
    LoudnessHistogram shortterm_histogram;

    constexpr double MIN_DURATION_MS = 4000.0;
    const size_t min_frames = static_cast<size_t>((MIN_DURATION_MS / 1000.0) * sample_rate);

    const size_t frames_per_chunk = 4096;
    std::vector<float> chunk(frames_per_chunk * channels);
    std::vector<float> short_buffer;
    size_t buffered_frames = 0;
    bool loudness_started = false;

    auto process_loudness_frames = [&](const float* data, size_t frames) {
        for (size_t frame = 0; frame < frames; ++frame) {
            const float* frame_samples = data + frame * channels;
            double weighted_sq = kfilter.process_frame(frame_samples);

            double block_power;
            if (momentary.add_sample(weighted_sq, block_power)) {
                momentary_histogram.add_block(block_power);
            }
            if (shortterm.add_sample(weighted_sq, block_power)) {
                shortterm_histogram.add_block(block_power);
            }
        }
    };

    while (true) {
        size_t frames_read = stream.read_frames(chunk.data(), frames_per_chunk);
        if (frames_read == 0) {
            break;
        }

        float chunk_peak = simd::find_peak_abs(chunk.data(), frames_read * static_cast<size_t>(channels));
        if (chunk_peak > sample_peak_linear) {
            sample_peak_linear = chunk_peak;
        }

        uint64_t frames_seen = total_frames;
        size_t pre_frames = 0;
        if (frames_seen < tc_samples) {
            uint64_t remaining = tc_samples - frames_seen;
            pre_frames = static_cast<size_t>(
                std::min<uint64_t>(static_cast<uint64_t>(frames_read), remaining));
        }

        const float* frame_samples = chunk.data();
        for (size_t frame = 0; frame < pre_frames; ++frame) {
            for (int ch = 0; ch < channels; ++ch) {
                float s = frame_samples[ch];
                double sample_sq = static_cast<double>(s) * static_cast<double>(s);
                sum_sq[ch] += sample_sq;
                avg_sigma_x2[ch] = avg_sigma_x2[ch] * mult + one_minus_mult * sample_sq;
            }
            frame_samples += channels;
        }

        for (size_t frame = pre_frames; frame < frames_read; ++frame) {
            for (int ch = 0; ch < channels; ++ch) {
                float s = frame_samples[ch];
                double sample_sq = static_cast<double>(s) * static_cast<double>(s);
                sum_sq[ch] += sample_sq;
                avg_sigma_x2[ch] = avg_sigma_x2[ch] * mult + one_minus_mult * sample_sq;

                if (avg_sigma_x2[ch] > max_sigma_x2[ch]) {
                    max_sigma_x2[ch] = avg_sigma_x2[ch];
                }
                if (avg_sigma_x2[ch] < min_sigma_x2[ch]) {
                    min_sigma_x2[ch] = avg_sigma_x2[ch];
                }
            }
            frame_samples += channels;
        }

        total_frames += frames_read;

        if (!loudness_started) {
            size_t needed = (buffered_frames < min_frames) ? (min_frames - buffered_frames) : 0;
            size_t to_buffer = std::min(needed, frames_read);
            if (to_buffer > 0) {
                size_t old_size = short_buffer.size();
                short_buffer.resize(old_size + to_buffer * channels);
                std::memcpy(short_buffer.data() + old_size,
                            chunk.data(),
                            to_buffer * channels * sizeof(float));
                buffered_frames += to_buffer;
            }

            if (buffered_frames >= min_frames) {
                process_loudness_frames(short_buffer.data(), buffered_frames);
                loudness_started = true;

                if (frames_read > to_buffer) {
                    process_loudness_frames(chunk.data() + (to_buffer * channels), frames_read - to_buffer);
                }
            }
        } else {
            process_loudness_frames(chunk.data(), frames_read);
        }
    }

    if (!loudness_started && buffered_frames > 0) {
        size_t loops_needed = (min_frames + buffered_frames - 1) / buffered_frames;
        for (size_t loop = 0; loop < loops_needed; ++loop) {
            process_loudness_frames(short_buffer.data(), buffered_frames);
        }
    }

    out.loudness.integrated = momentary_histogram.get_integrated_loudness();
    out.loudness.momentary_max = momentary.get_max_loudness();
    out.loudness.shortterm_max = shortterm.get_max_loudness();
    out.loudness.range = shortterm_histogram.get_loudness_range();

    if (sample_peak_linear > 0.0f) {
        out.loudness.sample_peak = 20.0 * std::log10(static_cast<double>(sample_peak_linear));
    } else {
        out.loudness.sample_peak = -std::numeric_limits<double>::infinity();
    }

    if (std::isinf(out.loudness.momentary_max) && momentary_histogram.total_count > 0) {
        out.loudness.momentary_max = momentary_histogram.get_max_loudness();
    }

    if (std::isinf(out.loudness.shortterm_max) && shortterm_histogram.total_count > 0) {
        out.loudness.shortterm_max = shortterm_histogram.get_max_loudness();
    }

    if (std::isinf(out.loudness.momentary_max)) {
        out.loudness.momentary_max = out.loudness.integrated;
    }

    if (std::isinf(out.loudness.shortterm_max)) {
        out.loudness.shortterm_max = out.loudness.momentary_max;
    }

    if (out.loudness.range <= 0.0 && shortterm_histogram.total_count < 2) {
        out.loudness.range = 20.0;
    }

    if (total_frames < tc_samples && total_frames > 0) {
        for (int ch = 0; ch < channels; ++ch) {
            double avg_power = sum_sq[ch] / static_cast<double>(total_frames);
            max_sigma_x2[ch] = avg_power;
            min_sigma_x2[ch] = avg_power;
        }
    }

    double total_sum_sq = 0.0;
    for (int ch = 0; ch < channels; ++ch) {
        total_sum_sq += sum_sq[ch];
    }

    const uint64_t total_sample_count =
        total_frames * static_cast<uint64_t>(channels);

    if (total_sample_count > 0 && total_sum_sq > 0.0) {
        double avg_rms = std::sqrt(total_sum_sq / static_cast<double>(total_sample_count));
        out.rms_average = 20.0 * std::log10(avg_rms);
    }

    double overall_max_sigma_x2 = 0.0;
    for (int ch = 0; ch < channels; ++ch) {
        if (max_sigma_x2[ch] > overall_max_sigma_x2) {
            overall_max_sigma_x2 = max_sigma_x2[ch];
        }
    }
    if (overall_max_sigma_x2 > 0.0) {
        double max_rms = std::sqrt(overall_max_sigma_x2);
        out.rms_max = 20.0 * std::log10(max_rms);
    }

    double overall_min_sigma_x2 = std::numeric_limits<double>::max();
    for (int ch = 0; ch < channels; ++ch) {
        if (min_sigma_x2[ch] < overall_min_sigma_x2) {
            overall_min_sigma_x2 = min_sigma_x2[ch];
        }
    }
    if (overall_min_sigma_x2 > 0.0 && overall_min_sigma_x2 < std::numeric_limits<double>::max()) {
        double min_rms = std::sqrt(overall_min_sigma_x2);
        double min_db = 20.0 * std::log10(min_rms);
        if (std::isinf(min_db)) {
            out.rms_min = -96.0;
        } else {
            out.rms_min = min_db;
        }
    } else {
        out.rms_min = -96.0;
    }

    return out;
}

// ============================================================================
// Utility Functions
// ============================================================================

double linear_to_db(double linear) {
    if (linear <= 0.0) return -std::numeric_limits<double>::infinity();
    return 20.0 * std::log10(linear);
}

double db_to_linear(double db) {
    return std::pow(10.0, db / 20.0);
}

std::string format_duration(double seconds) {
    int hours = static_cast<int>(seconds / 3600);
    int mins = static_cast<int>(std::fmod(seconds, 3600) / 60);
    double secs = std::fmod(seconds, 60);
    int whole_secs = static_cast<int>(secs);
    int millis = static_cast<int>((secs - whole_secs) * 1000 + 0.5);

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", hours, mins, whole_secs, millis);
    return std::string(buf);
}

} // namespace pb_audio
