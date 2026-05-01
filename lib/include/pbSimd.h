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
    __m256 max_vec = _mm256_setzero_ps();

    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(7);
    for (; i < simd_count; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        __m256 abs_v = _mm256_andnot_ps(sign_mask, v);
        max_vec = _mm256_max_ps(max_vec, abs_v);
    }

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

    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(3);
    for (; i < simd_count; i += 4) {
        __m128 v = _mm_loadu_ps(data + i);
        __m256d vd = _mm256_cvtps_pd(v);
        sum_vec = _mm256_fmadd_pd(vd, vd, sum_vec);
    }
    double sum = hsum_pd(sum_vec);
    for (; i < count; ++i) {
        double v = static_cast<double>(data[i]);
        sum += v * v;
    }
    return sum;
}

inline double weighted_sum_of_squares_stereo(const float* data, size_t frames) {
    if (frames == 0) return 0.0;

    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    const size_t simd_frames = frames & ~static_cast<size_t>(1);
    for (; i < simd_frames; i += 2) {
        __m128 v = _mm_loadu_ps(data + i * 2);
        __m256d vd = _mm256_cvtps_pd(v);
        sum_vec = _mm256_fmadd_pd(vd, vd, sum_vec);
    }
    double sum = hsum_pd(sum_vec);
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
    float32x4_t max_vec = vdupq_n_f32(0.0f);
    size_t i = 0;
    const size_t simd_count = count & ~static_cast<size_t>(3);
    for (; i < simd_count; i += 4) {
        max_vec = vmaxq_f32(max_vec, vabsq_f32(vld1q_f32(data + i)));
    }
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

} // namespace simd
} // namespace pb_audio

#endif // PB_SIMD_H
