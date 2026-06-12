# pbAudioStats

High-performance C++17 audio analysis library and CLI with ITU-R BS.1770-4 loudness measurement, true peak detection, RMS analysis, and audio normalization. Built for **uncompromising accuracy and speed** — every measurement is verified against EBU Tech 3341/3342 theoretical values and cross-checked against `ffmpeg ebur128`.

[日本語版 README](README_ja.md)

## Highlights

- **Standards-grade accuracy** — BS.1770-4 / EBU R 128 / EBU Tech 3341/3342 compliant, SOX-compatible RMS. Verified bit-stable against a golden test-signal suite.
- **Extreme performance** — AVX2 (x86-64) and NEON (ARM64) hand-tuned kernels with a streaming block-FIR true-peak scanner, fully batched K-weighting, and O(1)/sample loudness aggregation.
- **Robust I/O** — native WAV/AIFF readers/writers (PCM 8/16/24/32-bit, IEEE float 32/64-bit, `WAVE_FORMAT_EXTENSIBLE`, AIFF/AIFC incl. `sowt`), MP3 via dr_mp3, full UTF-8 / Japanese filename support on Windows.
- **Parallel batch processing** — scales across all CPU cores for analysis, with a memory-budgeted admission scheme for normalization.

## Features

- **BS.1770-4 Loudness Measurement**
  - Integrated Loudness (LUFS)
  - Short-term Loudness Maximum (LUFS)
  - Momentary Loudness Maximum (LUFS)
  - Loudness Range LRA (LU)
- **True Peak Detection** — 4× oversampling per ITU-R BS.1770-4 (48-tap polyphase Kaiser-windowed sinc, ~89 dB stopband). Computed exactly even on the large-file streaming path.
- **Sample Peak** (dBFS)
- **RMS Analysis** — SOX-compatible exponential-moving-average implementation (min / max / average)
- **Audio Normalization** — Peak, True Peak, Integrated / Short-term / Momentary Loudness, RMS min/max/average targets
- **Multi-format Support** — WAV, AIFF/AIFC, MP3
- **Cross-platform** — Windows (x64, AVX2) and macOS (Apple Silicon / Intel)

## Requirements

- C++17 compatible compiler
- CMake 3.16+
- **Windows**: MSVC 2022, **CPU with AVX2 + FMA** (Intel Haswell 2013+ / AMD Excavator 2015+). The binary detects unsupported CPUs at startup and prints a diagnostic instead of crashing.
- **macOS**: Clang/GCC (Apple Silicon NEON, or x86-64 AVX2)

## Building

### Windows

```batch
build_win.bat Release
```

or manually:

```batch
mkdir build_win && cd build_win
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

### macOS

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0
cmake --build build -j
```

Universal binary:

```bash
cmake -B build -G Xcode -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0
cmake --build build --config Release
```

## CLI Usage

```bash
# Analyze a single file (integrated loudness + true peak)
pbAudioStats -i -tp input.wav

# Analyze a directory with every column
pbAudioStats -f -fe -fea -sr -bt -ch -tm -du -i -s -m -l -pk -tp -rn -rm -ra ./audio_folder/

# Export results to UTF-8 CSV (with BOM, Excel-friendly)
pbAudioStats -i -tp -pk ./audio_folder/ results.csv

# Normalize to -23 LUFS (integrated)
pbAudioStats -norm-i:-23.0 input.wav output.wav

# Normalize a whole folder to -1 dBTP (mirrors the input tree into the output dir)
pbAudioStats -norm-tp:-1.0 ./in_folder/ ./out_folder/

# Limit worker threads
pbAudioStats -j8 -i ./audio_folder/
```

With no analysis options and no `-norm-*`, every column is emitted. If the output path ends in `.csv`, results are written as UTF-8 CSV (with BOM) instead of stdout.

### Analysis Options

| Option | Description | Unit |
|--------|-------------|------|
| `-f` | File name (without extension) | - |
| `-fe` | File name (with extension) | - |
| `-fea` | Full file path | - |
| `-sr` | Sample rate | Hz |
| `-bt` | Bit depth | bits |
| `-ch` | Number of channels | - |
| `-tm` | Total time | HH:MM:SS.mmm |
| `-du` | Duration | seconds |
| `-i` | Integrated Loudness | LUFS |
| `-s` | Short-term Loudness Maximum | LUFS |
| `-m` | Momentary Loudness Maximum | LUFS |
| `-l` | Loudness Range (LRA) | LU |
| `-pk` | Sample Peak | dBFS |
| `-tp` | True Peak | dBTP |
| `-rn` | RMS Minimum | dB |
| `-rm` | RMS Maximum | dB |
| `-ra` | RMS Average | dB |

### Normalization Options

Mutually exclusive with analysis options. The gain needed to hit the target is computed, applied with clipping, and written out. **MP3 output is not supported** — normalizing an MP3 requires a `.wav` or `.aiff` output path.

| Option | Description | Example |
|--------|-------------|---------|
| `-norm-pk:<value>` | Normalize to Peak | `-norm-pk:-1.0` |
| `-norm-tp:<value>` | Normalize to True Peak | `-norm-tp:-1.0` |
| `-norm-i:<value>` | Normalize to Integrated Loudness | `-norm-i:-23.0` |
| `-norm-s:<value>` | Normalize to Short-term Max | `-norm-s:-18.0` |
| `-norm-m:<value>` | Normalize to Momentary Max | `-norm-m:-18.0` |
| `-norm-rn:<value>` | Normalize to RMS Min | `-norm-rn:-12.0` |
| `-norm-rm:<value>` | Normalize to RMS Max | `-norm-rm:-12.0` |
| `-norm-ra:<value>` | Normalize to RMS Average | `-norm-ra:-20.0` |

### Other Options

| Option | Description |
|--------|-------------|
| `-j<N>` | Number of worker threads (default: all cores) |
| `-h` | Show help |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Usage / argument / I/O error |
| `2` | Completed, but one or more files failed to load (placeholder values emitted) |
| `3` | CPU does not support the required AVX2/FMA instruction set |

## Library Usage

```cpp
#include "pbAudioStats.h"
using namespace pb_audio;

int main() {
    // Analyze a file
    AudioStats stats = analyze("input.wav");

    std::cout << "Integrated: " << stats.integrated_loudness << " LUFS\n";
    std::cout << "True Peak:  " << stats.true_peak << " dBTP\n";
    std::cout << "RMS Avg:    " << stats.rms_average << " dB\n";

    // Normalize to -23 LUFS
    Normalizer::normalize_and_save(
        "input.wav", "output.wav",
        Normalizer::Target::Integrated, -23.0);

    return 0;
}
```

Key API surface (`lib/include/pbAudioStats.h`):

- `analyze(path)` / `analyze(AudioData&, source_path)` — full analysis, returns `AudioStats`.
- `AudioReader::load(path)` / `open_stream(path)` — decode to memory or stream.
- `LoudnessMeter::measure / measure_with_rms / measure_stream` — loudness (+RMS, +true peak in the streaming path).
- `TruePeakMeter::measure(audio[, known_sample_peak])` — exact 4× oversampled true peak.
- `RMSMeter::measure` — SOX-compatible RMS.
- `Normalizer::calculate_gain / apply_gain / normalize_and_save`.

## Technical Specifications

### BS.1770-4 Compliance

| Item | Specification |
|------|---------------|
| K-weighting | High Shelf + High Pass biquad filters |
| Reference rate | 48 kHz (requantized for other rates, ≤0.0011 dB deviation) |
| Momentary | 400 ms block, 75% overlap |
| Short-term | 3000 ms block, 67% overlap |
| Absolute gate | -70 LUFS |
| Relative gate | -10 LU below ungated mean |
| LRA | 10–95 percentile, -20 LU relative gate (EBU Tech 3342) |
| True peak | 4× oversampling, 48-tap polyphase Kaiser sinc (β=9.6), end-of-stream flushed |
| Channel weights | Mono/stereo/3.0 = 1.0; surrounds +1.5 dB; LFE excluded (4.0–7.1 layouts) |

Conforms to ITU-R **BS.1770-5** for channel-based mono/stereo/5.1 measurement (no algorithmic change from BS.1770-4 for these layouts).

### SOX Compatibility

| Item | Specification |
|------|---------------|
| RMS method | Exponential moving average |
| Time constant | 50 ms (default) |
| Settling time | 5 × time_constant × sample_rate |

### Performance Architecture

- **True peak**: streaming block-FIR with AVX2/NEON FMA kernels and a rigorous L1-norm bound that prunes chunks which provably cannot beat the running peak (exact, never changes the result).
- **Loudness**: chunk-batched K-weighting with filter state kept in registers; block aggregation reduced to O(1) per sample via per-partition segment sums.
- **RMS / peak**: single shared sample-peak scan feeds both the loudness meter and the true-peak seed; SOX EMA batched per channel.
- **Denormals**: hardware FTZ/DAZ on x86 removes per-sample flush branches from the IIR hot path.
- **Batch**: lock-free work-stealing across all cores; normalization is gated by a RAM byte-budget to avoid OOM.
- **Build**: `/fp:contract` (MSVC) / `-ffp-contract=fast` (GCC/Clang) for FMA contraction (single rounding — *more* accurate), without `/fp:fast` so compensated summation stays intact.

## License

Proprietary

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-06 | Initial release |
| 1.1.1 | 2026-05-02 | Renamed project to pbAudioStats |
| 1.2.0 | 2026-05-02 | Accuracy + performance overhaul (BS.1770-4 4× oversampling, SIMD bit-depth conversion) |
| 1.3.0 | 2026-06-12 | Extreme optimization: streaming block-FIR true peak (AVX2/NEON + L1 pruning + EOF flush), fully batched loudness/RMS, O(1) block aggregation, FTZ/DAZ, all-core scheduling, memory-budgeted normalization. Correctness fixes: 32-bit PCM full-scale overflow, exact true peak on the streaming path, `WAVE_FORMAT_EXTENSIBLE` support, UTF-8 / Japanese filenames, round-to-nearest PCM writes, AVX2 CPU guard, robust header parsing. |
