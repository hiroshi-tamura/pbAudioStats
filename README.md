# pb_audio_stats

High-performance C++ audio analysis library with BS.1770-4 loudness measurement, true peak detection, RMS analysis, and audio normalization.

[日本語版 README](README_ja.md)

## Features

- **BS.1770-4 Loudness Measurement**
  - Integrated Loudness (LUFS)
  - Short-term Loudness Maximum (LUFS)
  - Momentary Loudness Maximum (LUFS)
  - Loudness Range LRA (LU)
- **True Peak Detection** - 4x oversampling per ITU-R BS.1770-4
- **RMS Analysis** - SOX compatible implementation
- **Audio Normalization** - Peak, True Peak, Loudness, RMS targets
- **Multi-format Support** - WAV, AIFF, MP3
- **Cross-platform** - macOS and Windows

## Requirements

- C++17 compatible compiler
- CMake 3.16+
- macOS: Clang/GCC
- Windows: MSVC 2022

## Building

### macOS

```bash
mkdir build_mac && cd build_mac
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Windows

```batch
mkdir build_win && cd build_win
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

## CLI Usage

```bash
# Analyze single file
pb_audio_stats -i -tp input.wav

# Analyze directory with all options
pb_audio_stats -f -fe -fea -sr -bt -ch -tm -du -i -s -m -l -pk -tp -rn -rm -ra ./audio_folder/

# Export to CSV
pb_audio_stats -i -tp -pk ./audio_folder/ results.csv

# Normalize to -23 LUFS
pb_audio_stats -norm-i:-23.0 input.wav output.wav
```

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
| `-tp` | True Peak | dBFS |
| `-rn` | RMS Minimum | dB |
| `-rm` | RMS Maximum | dB |
| `-ra` | RMS Average | dB |

### Normalization Options

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

## Library Usage

```cpp
#include "pb_audio_stats.h"
using namespace pb_audio;

int main() {
    // Analyze file
    AudioStats stats = analyze("input.wav");

    std::cout << "Integrated: " << stats.integrated_loudness << " LUFS\n";
    std::cout << "True Peak: " << stats.true_peak << " dBFS\n";
    std::cout << "RMS Avg: " << stats.rms_average << " dB\n";

    // Normalize to -23 LUFS
    Normalizer::normalize_and_save(
        "input.wav",
        "output.wav",
        Normalizer::Target::Integrated,
        -23.0
    );

    return 0;
}
```

## Technical Specifications

### BS.1770-4 Compliance

| Item | Specification |
|------|---------------|
| K-weighting | High Shelf + High Pass biquad filters |
| Reference rate | 48kHz (requantized for other rates) |
| Momentary | 400ms block, 75% overlap |
| Short-term | 3000ms block, 67% overlap |
| Absolute gate | -70 LUFS |
| Relative gate | -10 LU below ungated mean |
| LRA | 10-95 percentile, -20dB relative gate |

### SOX Compatibility

| Item | Specification |
|------|---------------|
| RMS method | Exponential moving average |
| Time constant | 50ms (default) |
| Settling time | 5 × time_constant × sample_rate |

## License

Proprietary

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-06 | Initial release |
