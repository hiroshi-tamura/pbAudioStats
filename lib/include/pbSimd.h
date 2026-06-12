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
#include <algorithm>

#if defined(__AVX2__)
    #define PB_SIMD_AVX2 1
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define PB_SIMD_NEON 1
    #include <arm_neon.h>
#else
    #define PB_SIMD_SCALAR 1
#endif

namespace pb_audio {
namespace simd {

// ============================================================================
// Peak detection: max(|x|) over float array
// ============================================================================

#if defined(PB_SIMD_AVX2)

inline float find_peak_abs(const float* data, size_t count) {
    if (count == 0) return 0.0f;

    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    // Four independent accumulators break the loop-carried VMAXPS dependency
    // chain (3-cycle latency) and let the loop run at load throughput.
    __m256 m0 = _mm256_setzero_ps();
    __m256 m1 = _mm256_setzero_ps();
    __m256 m2 = _mm256_setzero_ps();
    __m256 m3 = _mm256_setzero_ps();

    size_t i = 0;
    const size_t simd_count32 = count & ~static_cast<size_t>(31);
    for (; i < simd_count32; i += 32) {
        m0 = _mm256_max_ps(m0, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(data + i)));
        m1 = _mm256_max_ps(m1, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(data + i + 8)));
        m2 = _mm256_max_ps(m2, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(data + i + 16)));
        m3 = _mm256_max_ps(m3, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(data + i + 24)));
    }
    const size_t simd_count8 = count & ~static_cast<size_t>(7);
    for (; i < simd_count8; i += 8) {
        m0 = _mm256_max_ps(m0, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(data + i)));
    }
    __m256 max_vec = _mm256_max_ps(_mm256_max_ps(m0, m1), _mm256_max_ps(m2, m3));

    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_val = _mm_cvtss_f32(m);

    for (; i < count; ++i) {
        float a = std::fabs(data[i]);
        if (a > max_val) max_val = a;
    }
    return max_val;
}

// 4-lane horizontal sum of __m256d
static inline double hsum_pd(__m256d v) {
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d s = _mm_add_pd(lo, hi);
    s = _mm_add_pd(s, _mm_unpackhi_pd(s, s));
    return _mm_cvtsd_f64(s);
}

inline double sum_of_squares(const float* data, size_t count) {
    if (count == 0) return 0.0;

    // Two independent FMA accumulators hide the 4-cycle FMA latency.
    __m256d s0 = _mm256_setzero_pd();
    __m256d s1 = _mm256_setzero_pd();
    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(7);
    for (; i < simd_count; i += 8) {
        __m256d d0 = _mm256_cvtps_pd(_mm_loadu_ps(data + i));
        __m256d d1 = _mm256_cvtps_pd(_mm_loadu_ps(data + i + 4));
        s0 = _mm256_fmadd_pd(d0, d0, s0);
        s1 = _mm256_fmadd_pd(d1, d1, s1);
    }
    double sum = hsum_pd(_mm256_add_pd(s0, s1));
    for (; i < count; ++i) {
        double v = static_cast<double>(data[i]);
        sum += v * v;
    }
    return sum;
}

inline double weighted_sum_of_squares_stereo(const float* data, size_t frames) {
    if (frames == 0) return 0.0;

    __m256d s0 = _mm256_setzero_pd();
    __m256d s1 = _mm256_setzero_pd();
    size_t i = 0;
    const size_t simd_frames = frames & ~static_cast<size_t>(3);
    for (; i < simd_frames; i += 4) {
        __m256d d0 = _mm256_cvtps_pd(_mm_loadu_ps(data + i * 2));
        __m256d d1 = _mm256_cvtps_pd(_mm_loadu_ps(data + i * 2 + 4));
        s0 = _mm256_fmadd_pd(d0, d0, s0);
        s1 = _mm256_fmadd_pd(d1, d1, s1);
    }
    double sum = hsum_pd(_mm256_add_pd(s0, s1));
    for (; i < frames; ++i) {
        double l = static_cast<double>(data[i * 2]);
        double r = static_cast<double>(data[i * 2 + 1]);
        sum += l * l + r * r;
    }
    return sum;
}

inline void apply_gain_and_clip(float* data, size_t count, float gain,
                                 float clip_min, float clip_max) {
    const __m256 g = _mm256_set1_ps(gain);
    const __m256 lo = _mm256_set1_ps(clip_min);
    const __m256 hi = _mm256_set1_ps(clip_max);

    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(7);
    for (; i < simd_count; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        v = _mm256_mul_ps(v, g);
        v = _mm256_max_ps(_mm256_min_ps(v, hi), lo);
        _mm256_storeu_ps(data + i, v);
    }
    for (; i < count; ++i) {
        float v = data[i] * gain;
        if (v > clip_max) v = clip_max;
        else if (v < clip_min) v = clip_min;
        data[i] = v;
    }
}

#elif defined(PB_SIMD_NEON)

inline float find_peak_abs(const float* data, size_t count) {
    if (count == 0) return 0.0f;
    // Four independent accumulators break the loop-carried max dependency.
    float32x4_t m0 = vdupq_n_f32(0.0f);
    float32x4_t m1 = vdupq_n_f32(0.0f);
    float32x4_t m2v = vdupq_n_f32(0.0f);
    float32x4_t m3 = vdupq_n_f32(0.0f);
    size_t i = 0;
    const size_t simd_count16 = count & ~static_cast<size_t>(15);
    for (; i < simd_count16; i += 16) {
        m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(data + i)));
        m1 = vmaxq_f32(m1, vabsq_f32(vld1q_f32(data + i + 4)));
        m2v = vmaxq_f32(m2v, vabsq_f32(vld1q_f32(data + i + 8)));
        m3 = vmaxq_f32(m3, vabsq_f32(vld1q_f32(data + i + 12)));
    }
    const size_t simd_count4 = count & ~static_cast<size_t>(3);
    for (; i < simd_count4; i += 4) {
        m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(data + i)));
    }
    float32x4_t max_vec = vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2v, m3));
    float32x2_t m2 = vpmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
    m2 = vpmax_f32(m2, m2);
    float max_val = vget_lane_f32(m2, 0);
    for (; i < count; ++i) {
        float a = std::fabs(data[i]);
        if (a > max_val) max_val = a;
    }
    return max_val;
}

inline double sum_of_squares(const float* data, size_t count) {
    if (count == 0) return 0.0;
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(3);
    for (; i < simd_count; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
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
    const size_t simd_frames = frames & ~static_cast<size_t>(1);
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

inline void apply_gain_and_clip(float* data, size_t count, float gain,
                                 float clip_min, float clip_max) {
    const float32x4_t g = vdupq_n_f32(gain);
    const float32x4_t lo = vdupq_n_f32(clip_min);
    const float32x4_t hi = vdupq_n_f32(clip_max);
    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(3);
    for (; i < simd_count; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        v = vmulq_f32(v, g);
        v = vmaxq_f32(vminq_f32(v, hi), lo);
        vst1q_f32(data + i, v);
    }
    for (; i < count; ++i) {
        float v = data[i] * gain;
        if (v > clip_max) v = clip_max;
        else if (v < clip_min) v = clip_min;
        data[i] = v;
    }
}

#else

inline float find_peak_abs(const float* data, size_t count) {
    float max_val = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float a = std::fabs(data[i]);
        if (a > max_val) max_val = a;
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

inline void apply_gain_and_clip(float* data, size_t count, float gain,
                                 float clip_min, float clip_max) {
    for (size_t i = 0; i < count; ++i) {
        float v = data[i] * gain;
        if (v > clip_max) v = clip_max;
        else if (v < clip_min) v = clip_min;
        data[i] = v;
    }
}

#endif

// ============================================================================
// Bit-depth conversion (integer PCM -> float [-1, 1))
// AVX2 fast paths fall back to scalar for the trailing samples and on
// platforms without AVX2.
// ============================================================================

namespace detail {

inline float i16_le_scalar(const uint8_t* src) {
    int16_t v = static_cast<int16_t>(src[0] | (src[1] << 8));
    return static_cast<float>(v) * (1.0f / 32768.0f);
}

inline float i16_be_scalar(const uint8_t* src) {
    int16_t v = static_cast<int16_t>((src[0] << 8) | src[1]);
    return static_cast<float>(v) * (1.0f / 32768.0f);
}

inline float i24_le_scalar(const uint8_t* src) {
    int32_t v = src[0] | (src[1] << 8) | (src[2] << 16);
    if (v & 0x800000) v |= 0xFF000000;
    return static_cast<float>(v) * (1.0f / 8388608.0f);
}

inline float i24_be_scalar(const uint8_t* src) {
    int32_t v = (src[0] << 16) | (src[1] << 8) | src[2];
    if (v & 0x800000) v |= 0xFF000000;
    return static_cast<float>(v) * (1.0f / 8388608.0f);
}

inline float i32_le_scalar(const uint8_t* src) {
    int32_t v = src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24);
    return static_cast<float>(static_cast<double>(v) / 2147483648.0);
}

inline float i32_be_scalar(const uint8_t* src) {
    int32_t v = (src[0] << 24) | (src[1] << 16) | (src[2] << 8) | src[3];
    return static_cast<float>(static_cast<double>(v) / 2147483648.0);
}

}  // namespace detail

#if defined(PB_SIMD_AVX2)

inline void convert_i16_le_to_float(const uint8_t* src, float* dst, size_t count) {
    const __m256 scale = _mm256_set1_ps(1.0f / 32768.0f);
    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(7);
    for (; i < simd_count; i += 8) {
        __m128i v16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i * 2));
        __m256i v32 = _mm256_cvtepi16_epi32(v16);
        __m256 f = _mm256_cvtepi32_ps(v32);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(f, scale));
    }
    for (; i < count; ++i) dst[i] = detail::i16_le_scalar(src + i * 2);
}

inline void convert_i16_be_to_float(const uint8_t* src, float* dst, size_t count) {
    const __m256 scale = _mm256_set1_ps(1.0f / 32768.0f);
    const __m128i bswap = _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9,
                                        6, 7, 4, 5, 2, 3, 0, 1);
    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(7);
    for (; i < simd_count; i += 8) {
        __m128i v16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i * 2));
        v16 = _mm_shuffle_epi8(v16, bswap);
        __m256i v32 = _mm256_cvtepi16_epi32(v16);
        __m256 f = _mm256_cvtepi32_ps(v32);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(f, scale));
    }
    for (; i < count; ++i) dst[i] = detail::i16_be_scalar(src + i * 2);
}

// 24-bit needs care: each iteration reads 16 bytes but consumes only 12
// (4 samples). Hold back the last full SIMD batch to avoid over-reading
// past the input by 4 bytes.
inline void convert_i24_le_to_float(const uint8_t* src, float* dst, size_t count) {
    const __m128 scale = _mm_set1_ps(1.0f / 8388608.0f);
    const __m128i mask = _mm_set_epi8(11, 10, 9, -1, 8, 7, 6, -1,
                                       5, 4, 3, -1, 2, 1, 0, -1);
    size_t simd_iter = count / 4;
    if (simd_iter > 0) --simd_iter;
    size_t simd_done = simd_iter * 4;
    size_t i = 0;
    for (; i < simd_done; i += 4) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i * 3));
        __m128i shuffled = _mm_shuffle_epi8(v, mask);
        __m128i shifted = _mm_srai_epi32(shuffled, 8);  // sign extend
        __m128 f = _mm_cvtepi32_ps(shifted);
        _mm_storeu_ps(dst + i, _mm_mul_ps(f, scale));
    }
    for (; i < count; ++i) dst[i] = detail::i24_le_scalar(src + i * 3);
}

inline void convert_i24_be_to_float(const uint8_t* src, float* dst, size_t count) {
    const __m128 scale = _mm_set1_ps(1.0f / 8388608.0f);
    // For BE, bytes per sample are MSB,MID,LSB; we want LE int32 byte order
    // 0,LSB,MID,MSB so that >>8 sign-extends from the MSB.
    const __m128i mask = _mm_set_epi8(9, 10, 11, -1, 6, 7, 8, -1,
                                       3, 4, 5, -1, 0, 1, 2, -1);
    size_t simd_iter = count / 4;
    if (simd_iter > 0) --simd_iter;
    size_t simd_done = simd_iter * 4;
    size_t i = 0;
    for (; i < simd_done; i += 4) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i * 3));
        __m128i shuffled = _mm_shuffle_epi8(v, mask);
        __m128i shifted = _mm_srai_epi32(shuffled, 8);
        __m128 f = _mm_cvtepi32_ps(shifted);
        _mm_storeu_ps(dst + i, _mm_mul_ps(f, scale));
    }
    for (; i < count; ++i) dst[i] = detail::i24_be_scalar(src + i * 3);
}

inline void convert_i32_le_to_float(const uint8_t* src, float* dst, size_t count) {
    const __m256 scale = _mm256_set1_ps(1.0f / 2147483648.0f);
    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(7);
    for (; i < simd_count; i += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i * 4));
        __m256 f = _mm256_cvtepi32_ps(v);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(f, scale));
    }
    for (; i < count; ++i) dst[i] = detail::i32_le_scalar(src + i * 4);
}

inline void convert_i32_be_to_float(const uint8_t* src, float* dst, size_t count) {
    const __m256 scale = _mm256_set1_ps(1.0f / 2147483648.0f);
    // Per-lane byte swap for 32-bit big-endian -> little-endian.
    const __m256i bswap = _mm256_setr_epi8(
        3, 2, 1, 0,  7, 6, 5, 4,  11, 10, 9, 8,  15, 14, 13, 12,
        3, 2, 1, 0,  7, 6, 5, 4,  11, 10, 9, 8,  15, 14, 13, 12);
    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(7);
    for (; i < simd_count; i += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i * 4));
        v = _mm256_shuffle_epi8(v, bswap);
        __m256 f = _mm256_cvtepi32_ps(v);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(f, scale));
    }
    for (; i < count; ++i) dst[i] = detail::i32_be_scalar(src + i * 4);
}

#else

inline void convert_i16_le_to_float(const uint8_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] = detail::i16_le_scalar(src + i * 2);
}
inline void convert_i16_be_to_float(const uint8_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] = detail::i16_be_scalar(src + i * 2);
}
inline void convert_i24_le_to_float(const uint8_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] = detail::i24_le_scalar(src + i * 3);
}
inline void convert_i24_be_to_float(const uint8_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] = detail::i24_be_scalar(src + i * 3);
}
inline void convert_i32_le_to_float(const uint8_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] = detail::i32_le_scalar(src + i * 4);
}
inline void convert_i32_be_to_float(const uint8_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] = detail::i32_be_scalar(src + i * 4);
}

#endif

inline void convert_u8_to_float(const uint8_t* src, float* dst, size_t count) {
    // 8-bit unsigned WAV: center is 128, range [-128, 127]/128
    constexpr float SCALE = 1.0f / 128.0f;
    for (size_t i = 0; i < count; ++i) {
        dst[i] = (static_cast<float>(src[i]) - 128.0f) * SCALE;
    }
}

inline void convert_s8_to_float(const uint8_t* src, float* dst, size_t count) {
    constexpr float SCALE = 1.0f / 128.0f;
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<float>(static_cast<int8_t>(src[i])) * SCALE;
    }
}

// ============================================================================
// Plain sum of doubles (fixed deterministic association: 4-way split)
// Used for loudness block segment sums; order is part of the contract.
// ============================================================================

#if defined(PB_SIMD_AVX2)
inline double sum_pd(const double* w, size_t n) {
    __m256d a0 = _mm256_setzero_pd();
    __m256d a1 = _mm256_setzero_pd();
    size_t i = 0;
    const size_t simd_n = n & ~static_cast<size_t>(7);
    for (; i < simd_n; i += 8) {
        a0 = _mm256_add_pd(a0, _mm256_loadu_pd(w + i));
        a1 = _mm256_add_pd(a1, _mm256_loadu_pd(w + i + 4));
    }
    double s = hsum_pd(_mm256_add_pd(a0, a1));
    for (; i < n; ++i) s += w[i];
    return s;
}
#elif defined(PB_SIMD_NEON)
inline double sum_pd(const double* w, size_t n) {
    float64x2_t a0 = vdupq_n_f64(0.0);
    float64x2_t a1 = vdupq_n_f64(0.0);
    size_t i = 0;
    const size_t simd_n = n & ~static_cast<size_t>(3);
    for (; i < simd_n; i += 4) {
        a0 = vaddq_f64(a0, vld1q_f64(w + i));
        a1 = vaddq_f64(a1, vld1q_f64(w + i + 2));
    }
    float64x2_t a = vaddq_f64(a0, a1);
    double s = vgetq_lane_f64(a, 0) + vgetq_lane_f64(a, 1);
    for (; i < n; ++i) s += w[i];
    return s;
}
#else
inline double sum_pd(const double* w, size_t n) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += w[i];
        s1 += w[i + 1];
        s2 += w[i + 2];
        s3 += w[i + 3];
    }
    double s = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) s += w[i];
    return s;
}
#endif

// ============================================================================
// double -> float conversion (for 64-bit float WAV)
// ============================================================================

#if defined(PB_SIMD_AVX2)
inline void convert_f64_to_f32(const double* src, float* dst, size_t count) {
    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(7);
    for (; i < simd_count; i += 8) {
        __m128 lo = _mm256_cvtpd_ps(_mm256_loadu_pd(src + i));
        __m128 hi = _mm256_cvtpd_ps(_mm256_loadu_pd(src + i + 4));
        _mm256_storeu_ps(dst + i, _mm256_set_m128(hi, lo));
    }
    for (; i < count; ++i) dst[i] = static_cast<float>(src[i]);
}
#elif defined(PB_SIMD_NEON)
inline void convert_f64_to_f32(const double* src, float* dst, size_t count) {
    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(3);
    for (; i < simd_count; i += 4) {
        float32x2_t lo = vcvt_f32_f64(vld1q_f64(src + i));
        float32x2_t hi = vcvt_f32_f64(vld1q_f64(src + i + 2));
        vst1q_f32(dst + i, vcombine_f32(lo, hi));
    }
    for (; i < count; ++i) dst[i] = static_cast<float>(src[i]);
}
#else
inline void convert_f64_to_f32(const double* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] = static_cast<float>(src[i]);
}
#endif

// ============================================================================
// float -> integer PCM conversion with fused gain + clip (for writers).
//
// Semantics (identical across scalar/AVX2/NEON paths, bit-exact):
//   s' = clamp(s * gain, -1.0f, 1.0f)        (single-precision multiply)
//   v  = round_to_nearest_even(s' * scale)   (scale = 32767 / 8388607)
// 32-bit uses double precision internally (float cannot represent 2^31-1).
// Scalar paths use lrintf/llrint which honor the default round-to-nearest
// mode and therefore match the SIMD cvt instructions exactly.
// ============================================================================

namespace detail {

inline int32_t f32_to_pcm_scalar(float s, float gain, float scale) {
    float v = s * gain;
    if (v > 1.0f) v = 1.0f;
    else if (v < -1.0f) v = -1.0f;
    return static_cast<int32_t>(std::lrint(v * scale));
}

inline int32_t f32_to_pcm32_scalar(float s, float gain) {
    float v = s * gain;
    if (v > 1.0f) v = 1.0f;
    else if (v < -1.0f) v = -1.0f;
    double d = static_cast<double>(v) * 2147483647.0;
    return static_cast<int32_t>(std::llrint(d));
}

}  // namespace detail

#if defined(PB_SIMD_AVX2)

// Convert count floats to interleaved 16-bit PCM. `big_endian` selects byte order.
inline void convert_float_to_i16(const float* src, uint8_t* dst, size_t count,
                                 float gain, bool big_endian) {
    const __m256 g = _mm256_set1_ps(gain);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    const __m256 scale = _mm256_set1_ps(32767.0f);
    const __m256i bswap = _mm256_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(15);
    for (; i < simd_count; i += 16) {
        __m256 v0 = _mm256_mul_ps(_mm256_loadu_ps(src + i), g);
        __m256 v1 = _mm256_mul_ps(_mm256_loadu_ps(src + i + 8), g);
        v0 = _mm256_max_ps(_mm256_min_ps(v0, one), neg_one);
        v1 = _mm256_max_ps(_mm256_min_ps(v1, one), neg_one);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, scale));  // round-to-nearest-even
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, scale));
        __m256i packed = _mm256_packs_epi32(i0, i1);                 // lane-interleaved
        packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
        if (big_endian) packed = _mm256_shuffle_epi8(packed, bswap);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i * 2), packed);
    }
    for (; i < count; ++i) {
        int32_t v = detail::f32_to_pcm_scalar(src[i], gain, 32767.0f);
        if (big_endian) {
            dst[i * 2] = static_cast<uint8_t>((v >> 8) & 0xFF);
            dst[i * 2 + 1] = static_cast<uint8_t>(v & 0xFF);
        } else {
            dst[i * 2] = static_cast<uint8_t>(v & 0xFF);
            dst[i * 2 + 1] = static_cast<uint8_t>((v >> 8) & 0xFF);
        }
    }
}

// Convert count floats to interleaved 24-bit PCM.
inline void convert_float_to_i24(const float* src, uint8_t* dst, size_t count,
                                 float gain, bool big_endian) {
    const __m256 g = _mm256_set1_ps(gain);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    const __m256 scale = _mm256_set1_ps(8388607.0f);

    alignas(32) int32_t tmp[8];
    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(7);
    for (; i < simd_count; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(src + i), g);
        v = _mm256_max_ps(_mm256_min_ps(v, one), neg_one);
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp),
                           _mm256_cvtps_epi32(_mm256_mul_ps(v, scale)));
        uint8_t* d = dst + i * 3;
        if (big_endian) {
            for (int k = 0; k < 8; ++k) {
                int32_t x = tmp[k];
                d[k * 3]     = static_cast<uint8_t>((x >> 16) & 0xFF);
                d[k * 3 + 1] = static_cast<uint8_t>((x >> 8) & 0xFF);
                d[k * 3 + 2] = static_cast<uint8_t>(x & 0xFF);
            }
        } else {
            for (int k = 0; k < 8; ++k) {
                int32_t x = tmp[k];
                d[k * 3]     = static_cast<uint8_t>(x & 0xFF);
                d[k * 3 + 1] = static_cast<uint8_t>((x >> 8) & 0xFF);
                d[k * 3 + 2] = static_cast<uint8_t>((x >> 16) & 0xFF);
            }
        }
    }
    for (; i < count; ++i) {
        int32_t x = detail::f32_to_pcm_scalar(src[i], gain, 8388607.0f);
        uint8_t* d = dst + i * 3;
        if (big_endian) {
            d[0] = static_cast<uint8_t>((x >> 16) & 0xFF);
            d[1] = static_cast<uint8_t>((x >> 8) & 0xFF);
            d[2] = static_cast<uint8_t>(x & 0xFF);
        } else {
            d[0] = static_cast<uint8_t>(x & 0xFF);
            d[1] = static_cast<uint8_t>((x >> 8) & 0xFF);
            d[2] = static_cast<uint8_t>((x >> 16) & 0xFF);
        }
    }
}

// Convert count floats to interleaved 32-bit PCM (double-precision scaling:
// float cannot represent 2147483647 exactly; the old float path overflowed
// +1.0 to INT32_MIN).
inline void convert_float_to_i32(const float* src, uint8_t* dst, size_t count,
                                 float gain, bool big_endian) {
    const __m128 g = _mm_set1_ps(gain);
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 neg_one = _mm_set1_ps(-1.0f);
    const __m256d scale = _mm256_set1_pd(2147483647.0);
    const __m128i bswap = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11,
                                       4, 5, 6, 7, 0, 1, 2, 3);

    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(3);
    for (; i < simd_count; i += 4) {
        __m128 v = _mm_mul_ps(_mm_loadu_ps(src + i), g);
        v = _mm_max_ps(_mm_min_ps(v, one), neg_one);
        __m256d d = _mm256_mul_pd(_mm256_cvtps_pd(v), scale);
        __m128i x = _mm256_cvtpd_epi32(d);  // round-to-nearest-even, in range by clamp
        if (big_endian) x = _mm_shuffle_epi8(x, bswap);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i * 4), x);
    }
    for (; i < count; ++i) {
        int32_t x = detail::f32_to_pcm32_scalar(src[i], gain);
        uint8_t* d = dst + i * 4;
        if (big_endian) {
            d[0] = static_cast<uint8_t>((x >> 24) & 0xFF);
            d[1] = static_cast<uint8_t>((x >> 16) & 0xFF);
            d[2] = static_cast<uint8_t>((x >> 8) & 0xFF);
            d[3] = static_cast<uint8_t>(x & 0xFF);
        } else {
            d[0] = static_cast<uint8_t>(x & 0xFF);
            d[1] = static_cast<uint8_t>((x >> 8) & 0xFF);
            d[2] = static_cast<uint8_t>((x >> 16) & 0xFF);
            d[3] = static_cast<uint8_t>((x >> 24) & 0xFF);
        }
    }
}

#elif defined(PB_SIMD_NEON)

inline void convert_float_to_i16(const float* src, uint8_t* dst, size_t count,
                                 float gain, bool big_endian) {
    const float32x4_t g = vdupq_n_f32(gain);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t neg_one = vdupq_n_f32(-1.0f);
    const float32x4_t scale = vdupq_n_f32(32767.0f);

    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(7);
    for (; i < simd_count; i += 8) {
        float32x4_t v0 = vmulq_f32(vld1q_f32(src + i), g);
        float32x4_t v1 = vmulq_f32(vld1q_f32(src + i + 4), g);
        v0 = vmaxq_f32(vminq_f32(v0, one), neg_one);
        v1 = vmaxq_f32(vminq_f32(v1, one), neg_one);
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(v0, scale));  // round-to-nearest-even
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(v1, scale));
        int16x8_t packed = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        if (big_endian) {
            packed = vreinterpretq_s16_u8(vrev16q_u8(vreinterpretq_u8_s16(packed)));
        }
        vst1q_s16(reinterpret_cast<int16_t*>(dst + i * 2), packed);
    }
    for (; i < count; ++i) {
        int32_t v = detail::f32_to_pcm_scalar(src[i], gain, 32767.0f);
        if (big_endian) {
            dst[i * 2] = static_cast<uint8_t>((v >> 8) & 0xFF);
            dst[i * 2 + 1] = static_cast<uint8_t>(v & 0xFF);
        } else {
            dst[i * 2] = static_cast<uint8_t>(v & 0xFF);
            dst[i * 2 + 1] = static_cast<uint8_t>((v >> 8) & 0xFF);
        }
    }
}

inline void convert_float_to_i24(const float* src, uint8_t* dst, size_t count,
                                 float gain, bool big_endian) {
    for (size_t i = 0; i < count; ++i) {
        int32_t x = detail::f32_to_pcm_scalar(src[i], gain, 8388607.0f);
        uint8_t* d = dst + i * 3;
        if (big_endian) {
            d[0] = static_cast<uint8_t>((x >> 16) & 0xFF);
            d[1] = static_cast<uint8_t>((x >> 8) & 0xFF);
            d[2] = static_cast<uint8_t>(x & 0xFF);
        } else {
            d[0] = static_cast<uint8_t>(x & 0xFF);
            d[1] = static_cast<uint8_t>((x >> 8) & 0xFF);
            d[2] = static_cast<uint8_t>((x >> 16) & 0xFF);
        }
    }
}

inline void convert_float_to_i32(const float* src, uint8_t* dst, size_t count,
                                 float gain, bool big_endian) {
    for (size_t i = 0; i < count; ++i) {
        int32_t x = detail::f32_to_pcm32_scalar(src[i], gain);
        uint8_t* d = dst + i * 4;
        if (big_endian) {
            d[0] = static_cast<uint8_t>((x >> 24) & 0xFF);
            d[1] = static_cast<uint8_t>((x >> 16) & 0xFF);
            d[2] = static_cast<uint8_t>((x >> 8) & 0xFF);
            d[3] = static_cast<uint8_t>(x & 0xFF);
        } else {
            d[0] = static_cast<uint8_t>(x & 0xFF);
            d[1] = static_cast<uint8_t>((x >> 8) & 0xFF);
            d[2] = static_cast<uint8_t>((x >> 16) & 0xFF);
            d[3] = static_cast<uint8_t>((x >> 24) & 0xFF);
        }
    }
}

#else

inline void convert_float_to_i16(const float* src, uint8_t* dst, size_t count,
                                 float gain, bool big_endian) {
    for (size_t i = 0; i < count; ++i) {
        int32_t v = detail::f32_to_pcm_scalar(src[i], gain, 32767.0f);
        if (big_endian) {
            dst[i * 2] = static_cast<uint8_t>((v >> 8) & 0xFF);
            dst[i * 2 + 1] = static_cast<uint8_t>(v & 0xFF);
        } else {
            dst[i * 2] = static_cast<uint8_t>(v & 0xFF);
            dst[i * 2 + 1] = static_cast<uint8_t>((v >> 8) & 0xFF);
        }
    }
}

inline void convert_float_to_i24(const float* src, uint8_t* dst, size_t count,
                                 float gain, bool big_endian) {
    for (size_t i = 0; i < count; ++i) {
        int32_t x = detail::f32_to_pcm_scalar(src[i], gain, 8388607.0f);
        uint8_t* d = dst + i * 3;
        if (big_endian) {
            d[0] = static_cast<uint8_t>((x >> 16) & 0xFF);
            d[1] = static_cast<uint8_t>((x >> 8) & 0xFF);
            d[2] = static_cast<uint8_t>(x & 0xFF);
        } else {
            d[0] = static_cast<uint8_t>(x & 0xFF);
            d[1] = static_cast<uint8_t>((x >> 8) & 0xFF);
            d[2] = static_cast<uint8_t>((x >> 16) & 0xFF);
        }
    }
}

inline void convert_float_to_i32(const float* src, uint8_t* dst, size_t count,
                                 float gain, bool big_endian) {
    for (size_t i = 0; i < count; ++i) {
        int32_t x = detail::f32_to_pcm32_scalar(src[i], gain);
        uint8_t* d = dst + i * 4;
        if (big_endian) {
            d[0] = static_cast<uint8_t>((x >> 24) & 0xFF);
            d[1] = static_cast<uint8_t>((x >> 16) & 0xFF);
            d[2] = static_cast<uint8_t>((x >> 8) & 0xFF);
            d[3] = static_cast<uint8_t>(x & 0xFF);
        } else {
            d[0] = static_cast<uint8_t>(x & 0xFF);
            d[1] = static_cast<uint8_t>((x >> 8) & 0xFF);
            d[2] = static_cast<uint8_t>((x >> 16) & 0xFF);
            d[3] = static_cast<uint8_t>((x >> 24) & 0xFF);
        }
    }
}

#endif

// 8-bit writers (rare; scalar everywhere, same rounding semantics)
inline void convert_float_to_u8(const float* src, uint8_t* dst, size_t count, float gain) {
    for (size_t i = 0; i < count; ++i) {
        int32_t v = detail::f32_to_pcm_scalar(src[i], gain, 127.0f);
        dst[i] = static_cast<uint8_t>(v + 128);
    }
}

inline void convert_float_to_s8(const float* src, uint8_t* dst, size_t count, float gain) {
    for (size_t i = 0; i < count; ++i) {
        int32_t v = detail::f32_to_pcm_scalar(src[i], gain, 127.0f);
        dst[i] = static_cast<uint8_t>(static_cast<int8_t>(v));
    }
}

} // namespace simd
} // namespace pb_audio

#endif // PB_SIMD_H
