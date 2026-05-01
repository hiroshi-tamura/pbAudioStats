/*
 * pbSimd.h - SIMD Abstraction Layer
 * Supports: AVX2 (x86-64), NEON (ARM64), Scalar fallback
 */

#ifndef PB_SIMD_H
#define PB_SIMD_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>

// Platform detection
#if defined(__AVX2__)
    #define PB_SIMD_AVX2 1
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define PB_SIMD_NEON 1
    #include <arm_neon.h>
#else
    #define PB_SIMD_SCALAR 1
#endif

// Alignment for SIMD operations (32 bytes for AVX2)
#define PB_SIMD_ALIGN 32

namespace pb_audio {
namespace simd {

// ============================================================================
// Memory Alignment Utilities
// ============================================================================

inline void* aligned_alloc(size_t size) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, PB_SIMD_ALIGN);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, PB_SIMD_ALIGN, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

inline void aligned_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Aligned vector wrapper
template<typename T>
class AlignedBuffer {
public:
    T* data;
    size_t size;

    AlignedBuffer() : data(nullptr), size(0) {}

    explicit AlignedBuffer(size_t n) : size(n) {
        data = static_cast<T*>(aligned_alloc(n * sizeof(T)));
        if (data) std::memset(data, 0, n * sizeof(T));
    }

    ~AlignedBuffer() {
        if (data) aligned_free(data);
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            if (data) aligned_free(data);
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    void resize(size_t n) {
        if (data) aligned_free(data);
        size = n;
        data = static_cast<T*>(aligned_alloc(n * sizeof(T)));
        if (data) std::memset(data, 0, n * sizeof(T));
    }

    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }
};

// ============================================================================
// SIMD Peak Detection (find max absolute value)
// Uses FMA and aligned loads for maximum performance
// ============================================================================

#if defined(PB_SIMD_AVX2)

inline float find_peak_abs(const float* data, size_t count) {
    if (count == 0) return 0.0f;

    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 max_vec = _mm256_setzero_ps();

    size_t i = 0;
    const size_t simd_count = count & ~7;

    for (; i < simd_count; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        __m256 abs_v = _mm256_andnot_ps(sign_mask, v);
        max_vec = _mm256_max_ps(max_vec, abs_v);
    }

    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 max128 = _mm_max_ps(lo, hi);
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));

    float max_val = _mm_cvtss_f32(max128);

    for (; i < count; ++i) {
        float abs_val = std::fabs(data[i]);
        if (abs_val > max_val) max_val = abs_val;
    }

    return max_val;
}

// Sum of squares
inline double sum_of_squares(const float* data, size_t count) {
    if (count == 0) return 0.0;

    __m256d sum_vec = _mm256_setzero_pd();

    size_t i = 0;
    size_t simd_count = count & ~3;  // Process 4 floats at a time (convert to double)

    for (; i < simd_count; i += 4) {
        __m128 v = _mm_loadu_ps(data + i);
        __m256d vd = _mm256_cvtps_pd(v);
        __m256d sq = _mm256_mul_pd(vd, vd);
        sum_vec = _mm256_add_pd(sum_vec, sq);
    }

    // Horizontal sum
    __m128d hi = _mm256_extractf128_pd(sum_vec, 1);
    __m128d lo = _mm256_castpd256_pd128(sum_vec);
    __m128d sum128 = _mm_add_pd(lo, hi);
    sum128 = _mm_hadd_pd(sum128, sum128);

    double sum = _mm_cvtsd_f64(sum128);

    // Handle remaining elements
    for (; i < count; ++i) {
        double v = static_cast<double>(data[i]);
        sum += v * v;
    }

    return sum;
}

// Weighted sum of squares for stereo (interleaved L, R with weight 1.0)
inline double weighted_sum_of_squares_stereo(const float* data, size_t frames) {
    if (frames == 0) return 0.0;

    __m256d sum_vec = _mm256_setzero_pd();

    size_t i = 0;
    size_t simd_frames = frames & ~1;  // Process 2 frames (4 samples) at a time

    for (; i < simd_frames; i += 2) {
        __m128 v = _mm_loadu_ps(data + i * 2);
        __m256d vd = _mm256_cvtps_pd(v);
        __m256d sq = _mm256_mul_pd(vd, vd);
        sum_vec = _mm256_add_pd(sum_vec, sq);
    }

    // Horizontal sum
    __m128d hi = _mm256_extractf128_pd(sum_vec, 1);
    __m128d lo = _mm256_castpd256_pd128(sum_vec);
    __m128d sum128 = _mm_add_pd(lo, hi);
    sum128 = _mm_hadd_pd(sum128, sum128);

    double sum = _mm_cvtsd_f64(sum128);

    // Handle remaining frames
    for (; i < frames; ++i) {
        double l = static_cast<double>(data[i * 2]);
        double r = static_cast<double>(data[i * 2 + 1]);
        sum += l * l + r * r;
    }

    return sum;
}

#elif defined(PB_SIMD_NEON)

inline float find_peak_abs(const float* data, size_t count) {
    if (count == 0) return 0.0f;

    float32x4_t max_vec = vdupq_n_f32(0.0f);

    size_t i = 0;
    size_t simd_count = count & ~3;

    for (; i < simd_count; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        float32x4_t abs_v = vabsq_f32(v);
        max_vec = vmaxq_f32(max_vec, abs_v);
    }

    // Horizontal max
    float32x2_t max2 = vpmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
    max2 = vpmax_f32(max2, max2);
    float max_val = vget_lane_f32(max2, 0);

    for (; i < count; ++i) {
        float abs_val = std::fabs(data[i]);
        if (abs_val > max_val) max_val = abs_val;
    }

    return max_val;
}

inline double sum_of_squares(const float* data, size_t count) {
    if (count == 0) return 0.0;

    float64x2_t sum_vec = vdupq_n_f64(0.0);

    size_t i = 0;
    size_t simd_count = count & ~3;

    for (; i < simd_count; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        // Convert to double and accumulate
        float64x2_t lo = vcvt_f64_f32(vget_low_f32(v));
        float64x2_t hi = vcvt_f64_f32(vget_high_f32(v));
        sum_vec = vfmaq_f64(sum_vec, lo, lo);
        sum_vec = vfmaq_f64(sum_vec, hi, hi);
    }

    double sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);

    for (; i < count; ++i) {
        double v = static_cast<double>(data[i]);
        sum += v * v;
    }

    return sum;
}

inline double weighted_sum_of_squares_stereo(const float* data, size_t frames) {
    if (frames == 0) return 0.0;

    float64x2_t sum_vec = vdupq_n_f64(0.0);

    size_t i = 0;
    size_t simd_frames = frames & ~1;

    for (; i < simd_frames; i += 2) {
        float32x4_t v = vld1q_f32(data + i * 2);
        float64x2_t lo = vcvt_f64_f32(vget_low_f32(v));
        float64x2_t hi = vcvt_f64_f32(vget_high_f32(v));
        sum_vec = vfmaq_f64(sum_vec, lo, lo);
        sum_vec = vfmaq_f64(sum_vec, hi, hi);
    }

    double sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);

    for (; i < frames; ++i) {
        double l = static_cast<double>(data[i * 2]);
        double r = static_cast<double>(data[i * 2 + 1]);
        sum += l * l + r * r;
    }

    return sum;
}

#else  // Scalar fallback

inline float find_peak_abs(const float* data, size_t count) {
    float max_val = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float abs_val = std::fabs(data[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    return max_val;
}

inline double sum_of_squares(const float* data, size_t count) {
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double v = static_cast<double>(data[i]);
        sum += v * v;
    }
    return sum;
}

inline double weighted_sum_of_squares_stereo(const float* data, size_t frames) {
    double sum = 0.0;
    for (size_t i = 0; i < frames; ++i) {
        double l = static_cast<double>(data[i * 2]);
        double r = static_cast<double>(data[i * 2 + 1]);
        sum += l * l + r * r;
    }
    return sum;
}

#endif

// ============================================================================
// SIMD FIR Filter (for True Peak oversampling)
// ============================================================================

#if defined(PB_SIMD_AVX2)

// 4x oversampling FIR filter kernel (48 taps per phase)
// Processes 8 input samples at once
inline void fir_upsample_4x_avx2(const float* input, float* output,
                                  const float* coeffs, size_t input_count,
                                  size_t taps_per_phase) {
    // Simplified: process one sample at a time but use SIMD for coefficient multiplication
    for (size_t i = 0; i < input_count; ++i) {
        for (int phase = 0; phase < 4; ++phase) {
            __m256 sum_vec = _mm256_setzero_ps();
            const float* phase_coeffs = coeffs + phase * taps_per_phase;

            size_t j = 0;
            size_t simd_taps = taps_per_phase & ~7;

            for (; j < simd_taps && i >= j / 4; j += 8) {
                // Load coefficients
                __m256 c = _mm256_loadu_ps(phase_coeffs + j);
                // Load input samples (need to handle boundary)
                float in[8] = {0};
                for (int k = 0; k < 8 && i >= (j + k) / 4; ++k) {
                    size_t idx = i - (j + k) / 4;
                    if (idx < input_count) in[k] = input[idx];
                }
                __m256 v = _mm256_loadu_ps(in);
                sum_vec = _mm256_fmadd_ps(c, v, sum_vec);
            }

            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
            __m128 lo = _mm256_castps256_ps128(sum_vec);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float sum = _mm_cvtss_f32(sum128);

            // Remaining taps
            for (; j < taps_per_phase; ++j) {
                size_t idx = i - j / 4;
                if (idx < input_count) {
                    sum += phase_coeffs[j] * input[idx];
                }
            }

            output[i * 4 + phase] = sum;
        }
    }
}

#endif

// ============================================================================
// Batch operations for RMS calculation
// ============================================================================

#if defined(PB_SIMD_AVX2)

inline void exp_smooth_batch_avx2(const float* samples, size_t count,
                                   double mult, double one_minus_mult,
                                   double& avg_sigma) {
    const __m256d mult_vec = _mm256_set1_pd(mult);
    const __m256d one_minus_mult_vec = _mm256_set1_pd(one_minus_mult);
    const __m128 zero = _mm_setzero_ps();

    size_t i = 0;
    const size_t simd_count = count & ~7;

    __m256d avg_vec = _mm256_set1_pd(avg_sigma);

    for (; i < simd_count; i += 8) {
        __m128 v0 = _mm_loadu_ps(samples + i);
        __m128 v1 = _mm_loadu_ps(samples + i + 4);

        __m256d vd0 = _mm256_cvtps_pd(v0);
        __m256d vd1 = _mm256_cvtps_pd(v1);

        __m256d sq0 = _mm256_mul_pd(vd0, vd0);
        __m256d sq1 = _mm256_mul_pd(vd1, vd1);

        avg_vec = _mm256_add_pd(_mm256_mul_pd(avg_vec, mult_vec),
                                 _mm256_mul_pd(sq0, one_minus_mult_vec));
        avg_vec = _mm256_add_pd(_mm256_mul_pd(avg_vec, mult_vec),
                                 _mm256_mul_pd(sq1, one_minus_mult_vec));
    }

    double avg_sum = _mm256_cvtsd_f64(_mm256_hadd_pd(avg_vec, avg_vec));

    for (; i < count; ++i) {
        double sq = static_cast<double>(samples[i]) * static_cast<double>(samples[i]);
        avg_sum = avg_sum * mult + one_minus_mult * sq;
    }

    avg_sigma = avg_sum;
}

inline void sum_squares_batch_avx2(const float* samples, size_t count,
                                    double& sum_out) {
    __m256d sum_vec = _mm256_setzero_pd();

    size_t i = 0;
    const size_t simd_count = count & ~7;

    for (; i < simd_count; i += 8) {
        __m128 v0 = _mm_loadu_ps(samples + i);
        __m128 v1 = _mm_loadu_ps(samples + i + 4);

        __m256d vd0 = _mm256_cvtps_pd(v0);
        __m256d vd1 = _mm256_cvtps_pd(v1);

        sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(vd0, vd0));
        sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(vd1, vd1));
    }

    double sum = _mm256_cvtsd_f64(_mm256_hadd_pd(sum_vec, sum_vec));

    for (; i < count; ++i) {
        double v = static_cast<double>(samples[i]);
        sum += v * v;
    }

    sum_out = sum;
}

#endif

inline void exp_smooth_batch(const float* samples, size_t count,
                              double* avg_sigma, double mult, double one_minus_mult) {
#if defined(PB_SIMD_AVX2)
    exp_smooth_batch_avx2(samples, count, mult, one_minus_mult, *avg_sigma);
#else
    double avg = *avg_sigma;
    for (size_t i = 0; i < count; ++i) {
        double sq = static_cast<double>(samples[i]) * static_cast<double>(samples[i]);
        avg = avg * mult + one_minus_mult * sq;
    }
    *avg_sigma = avg;
#endif
}

} // namespace simd
} // namespace pb_audio

#endif // PB_SIMD_H
