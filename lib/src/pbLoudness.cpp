/*
 * pbLoudness.cpp - BS.1770-4 Loudness Measurement
 *
 * Implements:
 * - K-weighting filter (high shelf + high pass biquads)
 * - Momentary loudness (400ms block, 75% overlap)
 * - Short-term loudness (3000ms block, 67% overlap)
 * - Integrated loudness (with -10 LU relative gate)
 * - Loudness Range (LRA) - 10-95 percentile (EBU Tech 3342)
 * - Sample Peak / True Peak (streaming path)
 *
 * Performance architecture (all measurement paths share one batched core):
 * - K-weighting runs chunk-batched with filter state in registers
 *   (specialized stereo kernel + generic N-channel kernel).
 * - Block aggregation is O(1)/sample via per-partition segment sums
 *   (one SIMD partial sum per segment instead of `partition` ring adds
 *   per sample). Mathematically identical to the previous ring buffer.
 * - SOX-compatible RMS EMA runs chunk-batched per channel with the
 *   recurrence kept in registers (the EMA itself is inherently serial).
 * - True peak is computed in the streaming path with the same polyphase
 *   scanner as TruePeakMeter (single pass, exact).
 * - Denormals are flushed in hardware (FTZ/DAZ) on x86; on other targets
 *   the biquad output is flushed at 1e-30 as before.
 *
 * Reference: ITU-R BS.1770-4, EBU R 128, EBU Tech 3341/3342, lib1770-2
 */

#include "pbAudioStats.h"
#include "pbSimd.h"
#include "pbTruePeakScanner.h"

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

// Batch size (frames) for the chunked measurement core
static constexpr size_t BATCH_FRAMES = 4096;

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

// Flush small biquad outputs to zero to avoid denormal slowdowns.
// On x86 (AVX2 builds) hardware FTZ/DAZ is enabled for the duration of each
// measurement (see DenormalGuard), so this is a no-op there. Other targets
// keep the 1e-30 flush (well below any audible level, ~-300 dB).
static inline double denormalize(double x) {
#if defined(PB_SIMD_AVX2)
    return x;
#else
    return (std::fabs(x) < 1.0e-30) ? 0.0 : x;
#endif
}

namespace {

// RAII guard: set FTZ (flush-to-zero) + DAZ (denormals-are-zero) on x86 so the
// IIR/EMA decay tails cannot hit denormal slowpaths; restores MXCSR on exit.
// Values this small are ~-300 dB and far below the -70 LUFS absolute gate and
// the displayed precision, so measurement results are unaffected.
struct DenormalGuard {
#if defined(PB_SIMD_AVX2)
    unsigned int old_csr;
    DenormalGuard() : old_csr(_mm_getcsr()) {
        _mm_setcsr(old_csr | 0x8040u);  // FTZ (bit 15) | DAZ (bit 6)
    }
    ~DenormalGuard() { _mm_setcsr(old_csr); }
#else
    DenormalGuard() {}
#endif
    DenormalGuard(const DenormalGuard&) = delete;
    DenormalGuard& operator=(const DenormalGuard&) = delete;
};

}  // namespace

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

    out.a1 = (2.0 * (k_sq - 1.0)) / a0;
    out.a2 = (1.0 - k_by_q + k_sq) / a0;
    out.b0 = (p.vh + p.vb * k_by_q + p.vl * k_sq) / a0;
    out.b1 = (2.0 * (p.vl * k_sq - p.vh)) / a0;
    out.b2 = (p.vh - p.vb * k_by_q + p.vl * k_sq) / a0;
}

// ============================================================================
// Channel Weights (per BS.1770-4)
//
// G values by channel position. Channel order assumptions follow the
// standard WAV channel masks for each common count:
//   1ch  mono            : C                          -> 1.0
//   2ch  stereo          : L R                        -> 1.0 1.0
//   3ch  L R C           : 1.0 x3
//   4ch  quad            : L R Ls Rs                  -> surrounds weighted
//   5ch  5.0             : L R C Ls Rs                -> surrounds weighted
//   6ch  5.1             : L R C LFE Ls Rs            -> LFE excluded
//   7ch  6.1             : L R C LFE Cs Ls Rs         -> LFE excluded
//   8ch  7.1             : L R C LFE Lrs Rrs Ls Rs    -> LFE excluded
//   >8ch                 : all 1.0 (unknown layout)
// ============================================================================

static double get_channel_weight(int channel_index, int total_channels) {
    // BS.1770-4: surrounds get +1.5 dB = 10^0.15 = 1.41253754...
    static const double SURROUND_WEIGHT = std::pow(10.0, 0.15);

    switch (total_channels) {
        case 1:
        case 2:
        case 3:
            return 1.0;
        case 4:  // quad: L R Ls Rs
            return (channel_index >= 2) ? SURROUND_WEIGHT : 1.0;
        case 5:  // 5.0: L R C Ls Rs
            return (channel_index >= 3) ? SURROUND_WEIGHT : 1.0;
        case 6:  // 5.1: L R C LFE Ls Rs
            if (channel_index == 3) return 0.0;
            return (channel_index >= 4) ? SURROUND_WEIGHT : 1.0;
        case 7:  // 6.1: L R C LFE Cs Ls Rs
            if (channel_index == 3) return 0.0;
            return (channel_index >= 4) ? SURROUND_WEIGHT : 1.0;
        case 8:  // 7.1: L R C LFE Lrs Rrs Ls Rs
            if (channel_index == 3) return 0.0;
            return (channel_index >= 4) ? SURROUND_WEIGHT : 1.0;
        default:
            return 1.0;
    }
}

// ============================================================================
// Histogram for Integrated Loudness & LRA
// ============================================================================

class LoudnessHistogram {
public:
    // Per-bin block count (compact: ~60KB total, fits in L2)
    std::vector<uint64_t> count;

    // Sum of block powers, computed with Neumaier compensated summation
    // for stability over very long signals.
    double sum_power = 0.0;
    double sum_compensation = 0.0;
    uint64_t total_count = 0;
    double max_power = 0.0;

    LoudnessHistogram() : count(HIST_NBINS, 0) {}

    static inline int power_to_bin(double power) {
        if (power <= 0.0) return -1;
        double db = -0.691 + 10.0 * std::log10(power);
        if (db < HIST_MIN_DB) return -1;
        if (db >= HIST_MAX_DB) return HIST_NBINS - 1;
        return static_cast<int>((db - HIST_MIN_DB) * HIST_GRAIN);
    }

    void add_block(double power) {
        if (power > max_power) max_power = power;
        int bin = power_to_bin(power);
        if (bin < 0) return;
        ++count[bin];
        ++total_count;

        // Neumaier compensated summation
        double t = sum_power + power;
        if (std::fabs(sum_power) >= std::fabs(power)) {
            sum_compensation += (sum_power - t) + power;
        } else {
            sum_compensation += (power - t) + sum_power;
        }
        sum_power = t;
    }

    double ungated_sum() const { return sum_power + sum_compensation; }

    double get_ungated_mean() const {
        if (total_count == 0) return SILENCE_THRESHOLD;
        return power_to_lufs(ungated_sum() / static_cast<double>(total_count));
    }

    // Integrated loudness with relative gate at -10 LU below ungated mean
    // Uses bin-center power to avoid the systematic negative bias of using
    // the lower bin edge.
    double get_integrated_loudness() const {
        if (total_count == 0) return SILENCE_THRESHOLD;

        double mean_power = ungated_sum() / static_cast<double>(total_count);
        double gate_power = mean_power * std::pow(10.0, 0.1 * RELATIVE_GATE);

        const double step = 1.0 / HIST_GRAIN;
        double sum = 0.0;
        uint64_t cnt = 0;

        for (int i = 0; i < HIST_NBINS; ++i) {
            if (count[i] == 0) continue;
            double bin_power_lo = lufs_to_power(HIST_MIN_DB + step * i);
            if (bin_power_lo <= gate_power) continue;
            double bin_power_mid = lufs_to_power(HIST_MIN_DB + step * (i + 0.5));
            sum += static_cast<double>(count[i]) * bin_power_mid;
            cnt += count[i];
        }

        if (cnt == 0) return SILENCE_THRESHOLD;
        return power_to_lufs(sum / static_cast<double>(cnt));
    }

    // Loudness Range (EBU Tech 3342): -20 LU relative gate, 10-95 percentile
    // (nearest-rank, ceil-based).
    double get_loudness_range() const {
        if (total_count < 2) return 0.0;

        double mean_power = ungated_sum() / static_cast<double>(total_count);
        double gate_power = mean_power * std::pow(10.0, 0.1 * (-20.0));

        const double step = 1.0 / HIST_GRAIN;
        uint64_t gated = 0;
        for (int i = 0; i < HIST_NBINS; ++i) {
            if (count[i] == 0) continue;
            double bin_power_lo = lufs_to_power(HIST_MIN_DB + step * i);
            if (bin_power_lo > gate_power) gated += count[i];
        }
        if (gated < 2) return 0.0;

        uint64_t lower_rank = static_cast<uint64_t>(
            std::ceil(static_cast<double>(gated) * LRA_LOWER_PERCENTILE));
        uint64_t upper_rank = static_cast<uint64_t>(
            std::ceil(static_cast<double>(gated) * LRA_UPPER_PERCENTILE));
        if (lower_rank == 0) lower_rank = 1;
        if (upper_rank == 0) upper_rank = 1;

        uint64_t cum = 0;
        double low_db = std::nan("");
        double high_db = std::nan("");
        for (int i = 0; i < HIST_NBINS; ++i) {
            if (count[i] == 0) continue;
            double bin_power_lo = lufs_to_power(HIST_MIN_DB + step * i);
            if (bin_power_lo <= gate_power) continue;
            cum += count[i];
            double bin_db_mid = HIST_MIN_DB + step * (static_cast<double>(i) + 0.5);
            if (std::isnan(low_db) && cum >= lower_rank) {
                low_db = bin_db_mid;
            }
            if (cum >= upper_rank) {
                high_db = bin_db_mid;
                break;
            }
        }

        if (std::isnan(low_db) || std::isnan(high_db)) return 0.0;
        return high_db - low_db;
    }

    double get_max_loudness() const { return power_to_lufs(max_power); }
};

// ============================================================================
// Sliding Block Aggregator
//
// Implements overlapping window measurement per BS.1770-4.
//
// O(1)/sample formulation: instead of adding every sample to all active
// partitions (`partition` memory RMWs per sample), keep one running segment
// sum; at each partition boundary (every overlap_size samples) store it in a
// small ring. A completed block's power is the sum of the last `partition`
// segment sums (times 1/block_size). This is algebraically identical to the
// previous per-partition ring accumulation.
//
// Note: max_power tracks the maximum block power UNGATED, per EBU Tech 3341
// (momentary/short-term maxima are defined without the -70 LUFS gate). The
// histogram (used for integrated loudness / LRA) still receives only blocks
// above the absolute gate, as required for gating.
// ============================================================================

class BlockAggregator {
public:
    double sample_rate;
    size_t overlap_size;  // Samples per partition
    size_t partition;     // Number of sub-blocks for overlap
    double scale;         // 1/block_size for averaging

    // Segment-sum ring
    std::vector<double> seg_sums;
    size_t seg_head;
    uint64_t seg_count;
    double seg_accum;
    size_t sample_count;

    // Silence gate threshold (power domain)
    double silence_gate;

    // Maximum block power seen (ungated)
    double max_power;

    BlockAggregator(double rate, double ms, int part)
        : sample_rate(rate), max_power(0.0) {
        partition = static_cast<size_t>(part);
        overlap_size = static_cast<size_t>(std::round((ms / 1000.0) * rate / part));
        if (overlap_size == 0) overlap_size = 1;
        scale = 1.0 / static_cast<double>(partition * overlap_size);

        seg_sums.assign(partition, 0.0);
        seg_head = 0;
        seg_count = 0;
        seg_accum = 0.0;
        sample_count = 0;

        silence_gate = lufs_to_power(SILENCE_THRESHOLD);
    }

    // Add a batch of weighted squared samples; completed gated blocks are
    // registered into `histogram`.
    void add_batch(const double* weighted_sq, size_t count, LoudnessHistogram& histogram) {
        size_t i = 0;
        while (i < count) {
            size_t take = overlap_size - sample_count;
            size_t remain = count - i;
            if (take > remain) take = remain;

            seg_accum += simd::sum_pd(weighted_sq + i, take);
            sample_count += take;
            i += take;

            if (sample_count == overlap_size) {
                seg_sums[seg_head] = seg_accum;
                seg_head = (seg_head + 1 == partition) ? 0 : seg_head + 1;
                seg_accum = 0.0;
                sample_count = 0;
                ++seg_count;

                if (seg_count >= partition) {
                    double block_power = 0.0;
                    for (size_t k = 0; k < partition; ++k) block_power += seg_sums[k];
                    block_power *= scale;

                    if (block_power > max_power) max_power = block_power;
                    if (block_power > silence_gate) histogram.add_block(block_power);
                }
            }
        }
    }

    double get_max_loudness() const {
        return power_to_lufs(max_power);
    }
};

// ============================================================================
// K-weighting Pre-filter (chunk-batched)
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

    // バッチ処理: 複数フレームを一括処理。
    // ステレオは専用カーネル（状態をレジスタ常駐、4xアンローリング）、
    // それ以外はチャンネル外側ループの汎用カーネル（状態レジスタ常駐）。
    void process_frames_batch(const float* samples, size_t frame_count, double* weighted_sq_out) {
        if (frame_count == 0) return;
        if (channels == 2) {
            process_frames_batch_stereo(samples, frame_count, weighted_sq_out);
        } else {
            process_frames_batch_generic(samples, frame_count, weighted_sq_out);
        }
    }

    void reset() {
        for (int ch = 0; ch < channels; ++ch) {
            high_shelf.reset(shelf_states[ch]);
            high_pass.reset(hp_states[ch]);
        }
    }

private:
    void process_frames_batch_stereo(const float* samples, size_t frame_count,
                                     double* weighted_sq_out) {
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
    }

    // Generic N-channel kernel: channel-outer loop so the filter state lives
    // in registers across the whole chunk. Channel contributions are added
    // in channel order (same summation order as process_frame).
    void process_frames_batch_generic(const float* samples, size_t frame_count,
                                      double* weighted_sq_out) {
        const double w0 = high_shelf.b0, w1 = high_shelf.b1, w2 = high_shelf.b2;
        const double w3 = high_shelf.a1, w4 = high_shelf.a2;
        const double h0 = high_pass.b0, h1 = high_pass.b1, h2 = high_pass.b2;
        const double h3 = high_pass.a1, h4 = high_pass.a2;
        const int nch = channels;

        std::memset(weighted_sq_out, 0, frame_count * sizeof(double));

        for (int ch = 0; ch < nch; ++ch) {
            const double weight = channel_weights[ch];
            if (weight == 0.0) continue;

            double sx1 = shelf_states[ch].x1, sx2 = shelf_states[ch].x2;
            double sy1 = shelf_states[ch].y1, sy2 = shelf_states[ch].y2;
            double hx1 = hp_states[ch].x1, hx2 = hp_states[ch].x2;
            double hy1 = hp_states[ch].y1, hy2 = hp_states[ch].y2;

            const float* src = samples + ch;
            for (size_t f = 0; f < frame_count; ++f) {
                double x = static_cast<double>(src[f * nch]);
                double y = denormalize(w0*x + w1*sx1 + w2*sx2 - w3*sy1 - w4*sy2);
                sx2 = sx1; sx1 = x; sy2 = sy1; sy1 = y;
                double z = denormalize(h0*y + h1*hx1 + h2*hx2 - h3*hy1 - h4*hy2);
                hx2 = hx1; hx1 = y; hy2 = hy1; hy1 = z;
                weighted_sq_out[f] += weight * z * z;
            }

            shelf_states[ch].x1 = sx1; shelf_states[ch].x2 = sx2;
            shelf_states[ch].y1 = sy1; shelf_states[ch].y2 = sy2;
            hp_states[ch].x1 = hx1; hp_states[ch].x2 = hx2;
            hp_states[ch].y1 = hy1; hp_states[ch].y2 = hy2;
        }
    }
};

// ============================================================================
// SOX-compatible RMS EMA state (chunk-batched, recurrence kept in registers)
// ============================================================================

namespace {

struct RmsChannelState {
    double avg_sigma_x2 = 0.0;
    double max_sigma_x2 = 0.0;
    double min_sigma_x2 = std::numeric_limits<double>::max();
    double sum_sq = 0.0;
};

class RmsBatch {
public:
    RmsBatch(int channels, double sample_rate, double window_ms)
        : channels_(channels), states_(channels) {
        const double time_constant = window_ms / 1000.0;
        mult_ = std::exp(-1.0 / (time_constant * sample_rate));
        one_minus_mult_ = 1.0 - mult_;
        tc_samples_ = static_cast<uint64_t>(5.0 * time_constant * sample_rate);
    }

    uint64_t tc_samples() const { return tc_samples_; }

    // Process `frames` interleaved frames whose first frame has measurement
    // index `start_index` (frame index within the measured region). Min/max
    // tracking begins at index >= tc_samples (SOX settling behavior).
    void process(const float* samples, size_t frames, uint64_t start_index) {
        // Split at the settling boundary so the inner loops are branch-free.
        size_t pre = 0;
        if (start_index < tc_samples_) {
            uint64_t remaining = tc_samples_ - start_index;
            pre = static_cast<size_t>(
                std::min<uint64_t>(remaining, static_cast<uint64_t>(frames)));
        }

        const int nch = channels_;
        for (int ch = 0; ch < nch; ++ch) {
            RmsChannelState& st = states_[ch];
            double avg = st.avg_sigma_x2;
            double mx = st.max_sigma_x2;
            double mn = st.min_sigma_x2;
            double ssq = st.sum_sq;
            const float* src = samples + ch;

            size_t f = 0;
            for (; f < pre; ++f) {
                double s = static_cast<double>(src[f * nch]);
                double sq = s * s;
                ssq += sq;
                avg = avg * mult_ + one_minus_mult_ * sq;
            }
            for (; f < frames; ++f) {
                double s = static_cast<double>(src[f * nch]);
                double sq = s * s;
                ssq += sq;
                avg = avg * mult_ + one_minus_mult_ * sq;
                if (avg > mx) mx = avg;
                if (avg < mn) mn = avg;
            }

            st.avg_sigma_x2 = avg;
            st.max_sigma_x2 = mx;
            st.min_sigma_x2 = mn;
            st.sum_sq = ssq;
        }
    }

    // SOX behavior for inputs shorter than the settling time: use the overall
    // average power as both peak and trough.
    void finalize(uint64_t measured_frames) {
        if (measured_frames > 0 && measured_frames < tc_samples_) {
            for (int ch = 0; ch < channels_; ++ch) {
                double avg_power = states_[ch].sum_sq / static_cast<double>(measured_frames);
                states_[ch].max_sigma_x2 = avg_power;
                states_[ch].min_sigma_x2 = avg_power;
            }
        }
    }

    void fill_results(uint64_t measured_frames, double& rms_min, double& rms_max,
                      double& rms_average) const {
        rms_min = -96.0;
        rms_max = -96.0;
        rms_average = -96.0;

        double total_sum_sq = 0.0;
        for (int ch = 0; ch < channels_; ++ch) total_sum_sq += states_[ch].sum_sq;

        const uint64_t total_sample_count =
            measured_frames * static_cast<uint64_t>(channels_);
        if (total_sample_count > 0 && total_sum_sq > 0.0) {
            double avg_rms = std::sqrt(total_sum_sq / static_cast<double>(total_sample_count));
            rms_average = 20.0 * std::log10(avg_rms);
        }

        double overall_max = 0.0;
        for (int ch = 0; ch < channels_; ++ch) {
            if (states_[ch].max_sigma_x2 > overall_max) overall_max = states_[ch].max_sigma_x2;
        }
        if (overall_max > 0.0) {
            rms_max = 20.0 * std::log10(std::sqrt(overall_max));
        }

        double overall_min = std::numeric_limits<double>::max();
        for (int ch = 0; ch < channels_; ++ch) {
            if (states_[ch].min_sigma_x2 < overall_min) overall_min = states_[ch].min_sigma_x2;
        }
        if (overall_min > 0.0 && overall_min < std::numeric_limits<double>::max()) {
            double min_db = 20.0 * std::log10(std::sqrt(overall_min));
            rms_min = std::isinf(min_db) ? -96.0 : min_db;
        }
    }

private:
    int channels_;
    double mult_;
    double one_minus_mult_;
    uint64_t tc_samples_;
    std::vector<RmsChannelState> states_;
};

}  // namespace

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
// to ensure proper BS.1770-4 measurement. For looped buffers we add one
// extra loop at the front to be used as a K-weighting filter warmup; the
// caller feeds those frames through the filter without registering blocks,
// so that the measured region runs on a settled IIR.
// ============================================================================

struct LoudnessBuffer {
    const float* samples;
    size_t total_frames;
    size_t warmup_frames;  // Frames at the start that should be discarded
};

static LoudnessBuffer prepare_loudness_buffer(const std::vector<float>& samples,
                                              size_t original_frames,
                                              int channels,
                                              uint32_t sample_rate,
                                              std::vector<float>& looped_storage) {
    double duration_ms = (static_cast<double>(original_frames) * 1000.0) / sample_rate;
    constexpr double MIN_DURATION_MS = 4000.0;

    if (duration_ms >= MIN_DURATION_MS || original_frames == 0) {
        return {samples.data(), original_frames, 0};
    }

    size_t min_frames = static_cast<size_t>((MIN_DURATION_MS / 1000.0) * sample_rate);
    size_t loops_for_min = (min_frames + original_frames - 1) / original_frames;
    size_t loops_total = loops_for_min + 1;  // extra loop for filter warmup
    size_t new_total_frames = original_frames * loops_total;

    looped_storage.resize(new_total_frames * channels);
    for (size_t loop = 0; loop < loops_total; ++loop) {
        std::memcpy(looped_storage.data() + (loop * original_frames * channels),
                    samples.data(),
                    original_frames * channels * sizeof(float));
    }

    return {looped_storage.data(), new_total_frames, original_frames};
}

// ============================================================================
// Shared result assembly (fallback chains for edge cases)
// ============================================================================

static void finalize_loudness_result(LoudnessMeter::Result& r,
                                     const BlockAggregator& momentary,
                                     const BlockAggregator& shortterm,
                                     const LoudnessHistogram& momentary_histogram,
                                     const LoudnessHistogram& shortterm_histogram) {
    r.integrated = momentary_histogram.get_integrated_loudness();
    r.momentary_max = momentary.get_max_loudness();
    r.shortterm_max = shortterm.get_max_loudness();
    r.range = shortterm_histogram.get_loudness_range();

    if (std::isinf(r.momentary_max) && momentary_histogram.total_count > 0) {
        r.momentary_max = momentary_histogram.get_max_loudness();
    }
    if (std::isinf(r.shortterm_max) && shortterm_histogram.total_count > 0) {
        r.shortterm_max = shortterm_histogram.get_max_loudness();
    }
    if (std::isinf(r.momentary_max)) {
        r.momentary_max = r.integrated;
    }
    if (std::isinf(r.shortterm_max)) {
        r.shortterm_max = r.momentary_max;
    }
    if (r.range <= 0.0 && shortterm_histogram.total_count < 2) {
        r.range = 0.0;
    }
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

    DenormalGuard denormal_guard;

    const double sample_rate = static_cast<double>(audio.sample_rate);
    const int channels = audio.channels;

    // Track sample peak from ORIGINAL audio (pre-loop, pre-filter)
    const size_t total_samples = audio.total_frames * static_cast<size_t>(channels);
    float sample_peak_linear = simd::find_peak_abs(audio.samples.data(), total_samples);

    if (sample_peak_linear > 0.0f) {
        result.sample_peak = 20.0 * std::log10(static_cast<double>(sample_peak_linear));
    } else {
        result.sample_peak = -std::numeric_limits<double>::infinity();
    }

    // Loop short audio to minimum length for accurate BS.1770-4 measurement
    std::vector<float> looped_samples;
    LoudnessBuffer buf = prepare_loudness_buffer(
        audio.samples, audio.total_frames, channels, audio.sample_rate, looped_samples);
    const float* samples = buf.samples;
    const size_t total_frames = buf.total_frames;
    const size_t warmup_frames = buf.warmup_frames;

    KWeightingFilter kfilter(sample_rate, channels);
    BlockAggregator momentary(sample_rate, MOMENTARY_BLOCK_MS, MOMENTARY_PARTITION);
    BlockAggregator shortterm(sample_rate, SHORTTERM_BLOCK_MS, SHORTTERM_PARTITION);
    LoudnessHistogram momentary_histogram;
    LoudnessHistogram shortterm_histogram;

    std::vector<double> weighted_sq_batch(BATCH_FRAMES);

    size_t frame = 0;
    while (frame < total_frames) {
        size_t batch_frames = std::min(BATCH_FRAMES, total_frames - frame);
        const float* batch_samples = samples + frame * channels;

        kfilter.process_frames_batch(batch_samples, batch_frames, weighted_sq_batch.data());

        size_t start = 0;
        if (frame < warmup_frames) {
            // Warmup region: filter state is updated, blocks discarded
            start = std::min(batch_frames, warmup_frames - frame);
        }
        if (start < batch_frames) {
            momentary.add_batch(weighted_sq_batch.data() + start, batch_frames - start,
                                momentary_histogram);
            shortterm.add_batch(weighted_sq_batch.data() + start, batch_frames - start,
                                shortterm_histogram);
        }

        frame += batch_frames;
    }

    finalize_loudness_result(result, momentary, shortterm,
                             momentary_histogram, shortterm_histogram);
    return result;
}

// ============================================================================
// Public API: LoudnessMeter::measure_with_rms
// ============================================================================

LoudnessMeter::ExtendedResult LoudnessMeter::measure_with_rms(const AudioData& audio,
                                                              double window_ms,
                                                              double known_sample_peak_linear) {
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
    out.true_peak = -std::numeric_limits<double>::infinity();

    if (audio.samples.empty() || audio.channels == 0 || audio.sample_rate == 0) {
        return out;
    }

    DenormalGuard denormal_guard;

    const double sample_rate = static_cast<double>(audio.sample_rate);
    const int channels = audio.channels;
    const size_t original_frames = audio.total_frames;

    // Sample peak from original audio only (reuse caller's scan if provided)
    double sample_peak_linear = known_sample_peak_linear;
    if (sample_peak_linear < 0.0) {
        sample_peak_linear = static_cast<double>(simd::find_peak_abs(
            audio.samples.data(), original_frames * static_cast<size_t>(channels)));
    }
    if (sample_peak_linear > 0.0) {
        out.loudness.sample_peak = 20.0 * std::log10(sample_peak_linear);
    }

    // Loop short audio to minimum length for accurate BS.1770-4 measurement
    std::vector<float> looped_samples;
    LoudnessBuffer buf = prepare_loudness_buffer(
        audio.samples, original_frames, channels, audio.sample_rate, looped_samples);
    const float* samples = buf.samples;
    const size_t total_frames = buf.total_frames;
    const size_t warmup_frames = buf.warmup_frames;
    const size_t rms_end = warmup_frames + original_frames;  // RMS covers the real audio window

    KWeightingFilter kfilter(sample_rate, channels);
    BlockAggregator momentary(sample_rate, MOMENTARY_BLOCK_MS, MOMENTARY_PARTITION);
    BlockAggregator shortterm(sample_rate, SHORTTERM_BLOCK_MS, SHORTTERM_PARTITION);
    LoudnessHistogram momentary_histogram;
    LoudnessHistogram shortterm_histogram;
    RmsBatch rms(channels, sample_rate, window_ms);

    std::vector<double> weighted_sq_batch(BATCH_FRAMES);

    size_t frame = 0;
    while (frame < total_frames) {
        size_t batch_frames = std::min(BATCH_FRAMES, total_frames - frame);
        const float* batch_samples = samples + frame * channels;

        kfilter.process_frames_batch(batch_samples, batch_frames, weighted_sq_batch.data());

        // Loudness: skip warmup region
        size_t start = 0;
        if (frame < warmup_frames) {
            start = std::min(batch_frames, warmup_frames - frame);
        }
        if (start < batch_frames) {
            momentary.add_batch(weighted_sq_batch.data() + start, batch_frames - start,
                                momentary_histogram);
            shortterm.add_batch(weighted_sq_batch.data() + start, batch_frames - start,
                                shortterm_histogram);
        }

        // RMS: only over [warmup_frames, rms_end)
        size_t rms_lo = std::max(frame, warmup_frames);
        size_t rms_hi = std::min(frame + batch_frames, rms_end);
        if (rms_lo < rms_hi) {
            rms.process(samples + rms_lo * channels, rms_hi - rms_lo,
                        static_cast<uint64_t>(rms_lo - warmup_frames));
        }

        frame += batch_frames;
    }

    finalize_loudness_result(out.loudness, momentary, shortterm,
                             momentary_histogram, shortterm_histogram);

    rms.finalize(original_frames);
    rms.fill_results(original_frames, out.rms_min, out.rms_max, out.rms_average);

    return out;
}

// ============================================================================
// Public API: LoudnessMeter::measure_stream
//
// Single pass: loudness + RMS + sample peak + exact BS.1770-4 true peak
// (the polyphase scanner is fully streamable).
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
    out.true_peak = -std::numeric_limits<double>::infinity();

    const auto info = stream.info;
    if (info.channels == 0 || info.sample_rate == 0) {
        return out;
    }

    DenormalGuard denormal_guard;

    const double sample_rate = static_cast<double>(info.sample_rate);
    const int channels = info.channels;

    float sample_peak_linear = 0.0f;
    uint64_t total_frames = 0;

    KWeightingFilter kfilter(sample_rate, channels);
    BlockAggregator momentary(sample_rate, MOMENTARY_BLOCK_MS, MOMENTARY_PARTITION);
    BlockAggregator shortterm(sample_rate, SHORTTERM_BLOCK_MS, SHORTTERM_PARTITION);
    LoudnessHistogram momentary_histogram;
    LoudnessHistogram shortterm_histogram;
    RmsBatch rms(channels, sample_rate, window_ms);
    tp_detail::TruePeakScanner truepeak(channels);

    constexpr double MIN_DURATION_MS = 4000.0;
    const size_t min_frames = static_cast<size_t>((MIN_DURATION_MS / 1000.0) * sample_rate);

    const size_t frames_per_chunk = 65536;
    std::vector<float> chunk(frames_per_chunk * channels);
    std::vector<double> weighted_sq_batch(BATCH_FRAMES);
    std::vector<float> short_buffer;
    size_t buffered_frames = 0;
    bool loudness_started = false;

    auto process_loudness_frames = [&](const float* data, size_t frames) {
        size_t f = 0;
        while (f < frames) {
            size_t batch_frames = std::min(BATCH_FRAMES, frames - f);
            kfilter.process_frames_batch(data + f * channels, batch_frames,
                                         weighted_sq_batch.data());
            momentary.add_batch(weighted_sq_batch.data(), batch_frames, momentary_histogram);
            shortterm.add_batch(weighted_sq_batch.data(), batch_frames, shortterm_histogram);
            f += batch_frames;
        }
    };

    while (true) {
        size_t frames_read = stream.read_frames(chunk.data(), frames_per_chunk);
        if (frames_read == 0) {
            break;
        }

        float chunk_peak = simd::find_peak_abs(chunk.data(),
                                               frames_read * static_cast<size_t>(channels));
        if (chunk_peak > sample_peak_linear) {
            sample_peak_linear = chunk_peak;
        }

        // True peak: single streaming pass, exact 4x oversampling
        truepeak.seed_peak(static_cast<double>(chunk_peak));
        truepeak.process(chunk.data(), frames_read);

        // RMS (measurement index = global frame index)
        rms.process(chunk.data(), frames_read, total_frames);

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
                short_buffer.clear();
                short_buffer.shrink_to_fit();

                if (frames_read > to_buffer) {
                    process_loudness_frames(chunk.data() + (to_buffer * channels),
                                            frames_read - to_buffer);
                }
            }
        } else {
            process_loudness_frames(chunk.data(), frames_read);
        }
    }

    if (!loudness_started && buffered_frames > 0) {
        size_t loops_for_min = (min_frames + buffered_frames - 1) / buffered_frames;
        // Warmup loop: pass through K-weighting filter, do not register blocks
        size_t f = 0;
        while (f < buffered_frames) {
            size_t batch_frames = std::min(BATCH_FRAMES, buffered_frames - f);
            kfilter.process_frames_batch(short_buffer.data() + f * channels, batch_frames,
                                         weighted_sq_batch.data());
            f += batch_frames;
        }
        // Measurement loops on a settled filter
        for (size_t loop = 0; loop < loops_for_min; ++loop) {
            process_loudness_frames(short_buffer.data(), buffered_frames);
        }
    }

    finalize_loudness_result(out.loudness, momentary, shortterm,
                             momentary_histogram, shortterm_histogram);

    if (sample_peak_linear > 0.0f) {
        out.loudness.sample_peak = 20.0 * std::log10(static_cast<double>(sample_peak_linear));
    } else {
        out.loudness.sample_peak = -std::numeric_limits<double>::infinity();
    }

    truepeak.flush();
    double tp_linear = truepeak.peak_linear();
    if (tp_linear > 0.0) {
        out.true_peak = 20.0 * std::log10(tp_linear);
    }

    rms.finalize(total_frames);
    rms.fill_results(total_frames, out.rms_min, out.rms_max, out.rms_average);

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
    if (millis >= 1000) {  // carry when fractional part rounds up to 1000ms
        millis -= 1000;
        if (++whole_secs >= 60) {
            whole_secs -= 60;
            if (++mins >= 60) {
                mins -= 60;
                ++hours;
            }
        }
    }

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", hours, mins, whole_secs, millis);
    return std::string(buf);
}

} // namespace pb_audio
