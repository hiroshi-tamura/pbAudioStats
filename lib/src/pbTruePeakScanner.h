/*
 * pbTruePeakScanner.h - Internal streaming true-peak scanner (not public API)
 *
 * ITU-R BS.1770-4 4x polyphase Kaiser-windowed-sinc oversampling, evaluated
 * as a block FIR over channel-planar double buffers:
 *
 *   y_p(n) = sum_{t=0}^{TAPS-1} c[p][t] * x[n-t]
 *          = sum_{k=0}^{TAPS-1} rev[p][k] * x[n-HISTORY+k]   (rev[p][k] = c[p][TAPS-1-k])
 *
 * Key properties:
 *  - No per-sample history shift: an 11-sample carry is kept between chunks.
 *  - AVX2: 4 output frames per iteration via coefficient-broadcast FMA
 *    (12 FMAs/frame/channel). NEON: 2 frames per iteration. Scalar fallback.
 *  - Rigorous chunk pruning: |y_p(n)| <= L1(c_p) * max|x| over the window,
 *    so a chunk whose max|x| * L1max cannot exceed the running peak is
 *    skipped entirely (exact, never changes the result).
 *  - flush() evaluates the trailing windows (zero-padded tail) so
 *    inter-sample peaks in the final samples are not missed.
 *  - Streamable: process() may be called repeatedly; used by both the
 *    in-memory path and measure_stream.
 */

#ifndef PB_TRUE_PEAK_SCANNER_H
#define PB_TRUE_PEAK_SCANNER_H

#include "pbSimd.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace pb_audio {
namespace tp_detail {

constexpr int OVERSAMPLE = 4;
constexpr int TAPS_PER_PHASE = 12;
constexpr int TOTAL_TAPS = OVERSAMPLE * TAPS_PER_PHASE;  // 48
constexpr int HISTORY = TAPS_PER_PHASE - 1;              // 11
constexpr size_t CHUNK_FRAMES = 4096;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    double half_x = 0.5 * x;
    for (int k = 1; k < 50; ++k) {
        term *= (half_x * half_x) / static_cast<double>(k * k);
        sum += term;
        if (term < 1e-15 * sum) break;
    }
    return sum;
}

struct PhaseTable {
    // rev[p][k] = prototype coefficient c[p][TAPS-1-k]; 32-byte aligned rows.
    alignas(32) double rev[OVERSAMPLE][TAPS_PER_PHASE];
    double l1_max;  // max over phases of sum |c[p][t]| (rigorous output bound)
};

inline PhaseTable build_phase_table() {
    double proto[TOTAL_TAPS];
    const double cutoff = M_PI / static_cast<double>(OVERSAMPLE);
    const double center = (static_cast<double>(TOTAL_TAPS) - 1.0) * 0.5;
    const double beta = 9.6;
    const double i0_beta = bessel_i0(beta);

    for (int n = 0; n < TOTAL_TAPS; ++n) {
        double m = static_cast<double>(n) - center;
        double sinc = (std::fabs(m) < 1e-12) ? (cutoff / M_PI)
                                             : (std::sin(cutoff * m) / (M_PI * m));
        double r = (2.0 * n / static_cast<double>(TOTAL_TAPS - 1)) - 1.0;
        double window = bessel_i0(beta * std::sqrt(std::fmax(0.0, 1.0 - r * r))) / i0_beta;
        proto[n] = sinc * window * static_cast<double>(OVERSAMPLE);
    }

    PhaseTable table{};
    table.l1_max = 0.0;
    for (int p = 0; p < OVERSAMPLE; ++p) {
        double l1 = 0.0;
        for (int t = 0; t < TAPS_PER_PHASE; ++t) {
            double c = proto[t * OVERSAMPLE + p];
            table.rev[p][TAPS_PER_PHASE - 1 - t] = c;
            l1 += std::fabs(c);
        }
        if (l1 > table.l1_max) table.l1_max = l1;
    }
    return table;
}

inline const PhaseTable& phase_table() {
    static const PhaseTable table = build_phase_table();
    return table;
}

// Convolve `frames` output positions over the planar buffer x (which has
// HISTORY samples of left context: window n spans x[n .. n+HISTORY]).
// Returns max |y| over all phases and frames.
#if defined(PB_SIMD_AVX2)

inline double convolve_max_abs(const double* x, size_t frames, const PhaseTable& tbl) {
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    __m256d vmax = _mm256_setzero_pd();

    size_t n = 0;
    const size_t simd_frames = frames & ~static_cast<size_t>(3);
    for (; n < simd_frames; n += 4) {
        const double* base = x + n;
        for (int p = 0; p < OVERSAMPLE; ++p) {
            const double* rc = tbl.rev[p];
            __m256d acc0 = _mm256_mul_pd(_mm256_set1_pd(rc[0]), _mm256_loadu_pd(base));
            __m256d acc1 = _mm256_mul_pd(_mm256_set1_pd(rc[1]), _mm256_loadu_pd(base + 1));
            __m256d acc2 = _mm256_mul_pd(_mm256_set1_pd(rc[2]), _mm256_loadu_pd(base + 2));
            __m256d acc3 = _mm256_mul_pd(_mm256_set1_pd(rc[3]), _mm256_loadu_pd(base + 3));
            for (int k = 4; k < TAPS_PER_PHASE; k += 4) {
                acc0 = _mm256_fmadd_pd(_mm256_set1_pd(rc[k]), _mm256_loadu_pd(base + k), acc0);
                acc1 = _mm256_fmadd_pd(_mm256_set1_pd(rc[k + 1]), _mm256_loadu_pd(base + k + 1), acc1);
                acc2 = _mm256_fmadd_pd(_mm256_set1_pd(rc[k + 2]), _mm256_loadu_pd(base + k + 2), acc2);
                acc3 = _mm256_fmadd_pd(_mm256_set1_pd(rc[k + 3]), _mm256_loadu_pd(base + k + 3), acc3);
            }
            __m256d acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
            vmax = _mm256_max_pd(vmax, _mm256_andnot_pd(sign_mask, acc));
        }
    }

    __m128d hi = _mm256_extractf128_pd(vmax, 1);
    __m128d lo = _mm256_castpd256_pd128(vmax);
    __m128d m = _mm_max_pd(lo, hi);
    m = _mm_max_pd(m, _mm_unpackhi_pd(m, m));
    double max_val = _mm_cvtsd_f64(m);

    for (; n < frames; ++n) {
        for (int p = 0; p < OVERSAMPLE; ++p) {
            const double* rc = tbl.rev[p];
            const double* w = x + n;
            double acc = 0.0;
            for (int k = 0; k < TAPS_PER_PHASE; ++k) acc += rc[k] * w[k];
            double a = std::fabs(acc);
            if (a > max_val) max_val = a;
        }
    }
    return max_val;
}

#elif defined(PB_SIMD_NEON)

inline double convolve_max_abs(const double* x, size_t frames, const PhaseTable& tbl) {
    float64x2_t vmax = vdupq_n_f64(0.0);

    size_t n = 0;
    const size_t simd_frames = frames & ~static_cast<size_t>(1);
    for (; n < simd_frames; n += 2) {
        const double* base = x + n;
        for (int p = 0; p < OVERSAMPLE; ++p) {
            const double* rc = tbl.rev[p];
            float64x2_t acc0 = vmulq_n_f64(vld1q_f64(base), rc[0]);
            float64x2_t acc1 = vmulq_n_f64(vld1q_f64(base + 1), rc[1]);
            for (int k = 2; k < TAPS_PER_PHASE; k += 2) {
                acc0 = vfmaq_n_f64(acc0, vld1q_f64(base + k), rc[k]);
                acc1 = vfmaq_n_f64(acc1, vld1q_f64(base + k + 1), rc[k + 1]);
            }
            float64x2_t acc = vaddq_f64(acc0, acc1);
            vmax = vmaxq_f64(vmax, vabsq_f64(acc));
        }
    }

    double max_val = vgetq_lane_f64(vmax, 0);
    double m1 = vgetq_lane_f64(vmax, 1);
    if (m1 > max_val) max_val = m1;

    for (; n < frames; ++n) {
        for (int p = 0; p < OVERSAMPLE; ++p) {
            const double* rc = tbl.rev[p];
            const double* w = x + n;
            double acc = 0.0;
            for (int k = 0; k < TAPS_PER_PHASE; ++k) acc += rc[k] * w[k];
            double a = std::fabs(acc);
            if (a > max_val) max_val = a;
        }
    }
    return max_val;
}

#else

inline double convolve_max_abs(const double* x, size_t frames, const PhaseTable& tbl) {
    double max_val = 0.0;
    for (size_t n = 0; n < frames; ++n) {
        const double* w = x + n;
        for (int p = 0; p < OVERSAMPLE; ++p) {
            const double* rc = tbl.rev[p];
            double acc = 0.0;
            for (int k = 0; k < TAPS_PER_PHASE; ++k) acc += rc[k] * w[k];
            double a = std::fabs(acc);
            if (a > max_val) max_val = a;
        }
    }
    return max_val;
}

#endif

class TruePeakScanner {
public:
    explicit TruePeakScanner(int channels)
        : channels_(channels),
          carry_(static_cast<size_t>(channels) * HISTORY, 0.0),
          plane_(HISTORY + CHUNK_FRAMES + HISTORY, 0.0) {}

    // Seed with a known sample peak (BS.1770: true peak >= sample peak; also
    // makes the pruning bound effective from the first chunk).
    void seed_peak(double sample_peak_abs) {
        if (sample_peak_abs > max_peak_) max_peak_ = sample_peak_abs;
    }

    void process(const float* interleaved, size_t frames) {
        const PhaseTable& tbl = phase_table();
        size_t done = 0;
        while (done < frames) {
            size_t n = frames - done;
            if (n > CHUNK_FRAMES) n = CHUNK_FRAMES;
            const float* src = interleaved + done * static_cast<size_t>(channels_);

            for (int ch = 0; ch < channels_; ++ch) {
                double* carry = carry_.data() + static_cast<size_t>(ch) * HISTORY;
                double* plane = plane_.data();

                // Deinterleave into the planar buffer, tracking the window max
                // (history + new samples) in the same pass.
                double wmax = 0.0;
                for (int k = 0; k < HISTORY; ++k) {
                    plane[k] = carry[k];
                    double a = std::fabs(carry[k]);
                    if (a > wmax) wmax = a;
                }
                for (size_t i = 0; i < n; ++i) {
                    double v = static_cast<double>(src[i * channels_ + ch]);
                    plane[HISTORY + i] = v;
                    double a = std::fabs(v);
                    if (a > wmax) wmax = a;
                }

                scan_plane(plane, n, wmax, tbl);

                // Carry the last HISTORY samples into the next chunk.
                const double* tail = plane + n;  // == plane + (HISTORY + n) - HISTORY
                for (int k = 0; k < HISTORY; ++k) carry[k] = tail[k];
            }
            done += n;
        }
    }

    // Evaluate the trailing windows (the final HISTORY output positions with
    // zero-padded input) so inter-sample peaks at end-of-stream are counted.
    void flush() {
        if (flushed_) return;
        flushed_ = true;
        const PhaseTable& tbl = phase_table();
        for (int ch = 0; ch < channels_; ++ch) {
            double* carry = carry_.data() + static_cast<size_t>(ch) * HISTORY;
            double* plane = plane_.data();
            double wmax = 0.0;
            for (int k = 0; k < HISTORY; ++k) {
                plane[k] = carry[k];
                double a = std::fabs(carry[k]);
                if (a > wmax) wmax = a;
            }
            for (int k = 0; k < HISTORY; ++k) plane[HISTORY + k] = 0.0;
            scan_plane(plane, HISTORY, wmax, tbl);
        }
    }

    double peak_linear() const { return max_peak_; }

private:
    void scan_plane(const double* plane, size_t frames, double wmax, const PhaseTable& tbl) {
        // Seed with the raw sample max (idempotent for the carried prefix).
        if (wmax > max_peak_) max_peak_ = wmax;

        // Rigorous pruning: no interpolated output of this chunk can exceed
        // wmax * l1_max. Skip the convolution if that bound cannot beat the
        // current peak.
        if (wmax * tbl.l1_max <= max_peak_) return;

        double m = convolve_max_abs(plane, frames, tbl);
        if (m > max_peak_) max_peak_ = m;
    }

    int channels_;
    double max_peak_ = 0.0;
    bool flushed_ = false;
    std::vector<double> carry_;
    std::vector<double> plane_;
};

}  // namespace tp_detail
}  // namespace pb_audio

#endif  // PB_TRUE_PEAK_SCANNER_H
