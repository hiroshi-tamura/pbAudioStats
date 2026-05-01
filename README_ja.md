# pbAudioStats

BS.1770-4ラウドネス計測、トゥルーピーク検出、RMS分析、オーディオノーマライズ機能を備えた高性能C++オーディオ解析ライブラリ。

[English README](README.md)

## 機能

- **BS.1770-4 ラウドネス計測**
  - 統合ラウドネス (LUFS)
  - 短期ラウドネス最大値 (LUFS)
  - モメンタリーラウドネス最大値 (LUFS)
  - ラウドネスレンジ LRA (LU)
- **トゥルーピーク検出** - ITU-R BS.1770-4準拠 4倍オーバーサンプリング
- **RMS分析** - SOX互換実装
- **オーディオノーマライズ** - ピーク、トゥルーピーク、ラウドネス、RMSターゲット
- **マルチフォーマット対応** - WAV, AIFF, MP3
- **クロスプラットフォーム** - macOS / Windows

## 必要環境

- C++17対応コンパイラ
- CMake 3.16以上
- macOS: Clang/GCC
- Windows: MSVC 2022

## ビルド方法

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

## CLIの使い方

```bash
# 単一ファイルを解析
pbAudioStats -i -tp input.wav

# ディレクトリを全オプションで解析
pbAudioStats -f -fe -fea -sr -bt -ch -tm -du -i -s -m -l -pk -tp -rn -rm -ra ./audio_folder/

# CSVにエクスポート
pbAudioStats -i -tp -pk ./audio_folder/ results.csv

# -23 LUFSにノーマライズ
pbAudioStats -norm-i:-23.0 input.wav output.wav
```

### 解析オプション

| オプション | 説明 | 単位 |
|-----------|------|------|
| `-f` | ファイル名（拡張子なし） | - |
| `-fe` | ファイル名（拡張子付き） | - |
| `-fea` | フルファイルパス | - |
| `-sr` | サンプルレート | Hz |
| `-bt` | ビット深度 | bits |
| `-ch` | チャンネル数 | - |
| `-tm` | 総時間 | HH:MM:SS.mmm |
| `-du` | デュレーション | 秒 |
| `-i` | 統合ラウドネス | LUFS |
| `-s` | 短期ラウドネス最大値 | LUFS |
| `-m` | モメンタリーラウドネス最大値 | LUFS |
| `-l` | ラウドネスレンジ (LRA) | LU |
| `-pk` | サンプルピーク | dBFS |
| `-tp` | トゥルーピーク | dBFS |
| `-rn` | RMS最小値 | dB |
| `-rm` | RMS最大値 | dB |
| `-ra` | RMS平均値 | dB |

### ノーマライズオプション

| オプション | 説明 | 例 |
|-----------|------|-----|
| `-norm-pk:<値>` | ピークにノーマライズ | `-norm-pk:-1.0` |
| `-norm-tp:<値>` | トゥルーピークにノーマライズ | `-norm-tp:-1.0` |
| `-norm-i:<値>` | 統合ラウドネスにノーマライズ | `-norm-i:-23.0` |
| `-norm-s:<値>` | 短期最大値にノーマライズ | `-norm-s:-18.0` |
| `-norm-m:<値>` | モメンタリー最大値にノーマライズ | `-norm-m:-18.0` |
| `-norm-rn:<値>` | RMS最小値にノーマライズ | `-norm-rn:-12.0` |
| `-norm-rm:<値>` | RMS最大値にノーマライズ | `-norm-rm:-12.0` |
| `-norm-ra:<値>` | RMS平均値にノーマライズ | `-norm-ra:-20.0` |

## ライブラリの使い方

```cpp
#include "pbAudioStats.h"
using namespace pb_audio;

int main() {
    // ファイル解析
    AudioStats stats = analyze("input.wav");

    std::cout << "統合ラウドネス: " << stats.integrated_loudness << " LUFS\n";
    std::cout << "トゥルーピーク: " << stats.true_peak << " dBFS\n";
    std::cout << "RMS平均: " << stats.rms_average << " dB\n";

    // -23 LUFSにノーマライズ
    Normalizer::normalize_and_save(
        "input.wav",
        "output.wav",
        Normalizer::Target::Integrated,
        -23.0
    );

    return 0;
}
```

## 技術仕様

### BS.1770-4準拠

| 項目 | 仕様 |
|------|------|
| Kウェイティング | High Shelf + High Pass バイクワッドフィルタ |
| 参照サンプルレート | 48kHz（他レートは再量子化） |
| モメンタリー | 400msブロック、75%オーバーラップ |
| 短期 | 3000msブロック、67%オーバーラップ |
| 絶対ゲート | -70 LUFS |
| 相対ゲート | アンゲート平均から-10 LU |
| LRA | 10-95パーセンタイル、-20dB相対ゲート |

### SOX互換性

| 項目 | 仕様 |
|------|------|
| RMS方式 | 指数移動平均 |
| 時定数 | 50ms（デフォルト） |
| セトリング時間 | 5 × 時定数 × サンプルレート |

## ライセンス

プロプライエタリ

## バージョン履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| 1.0.0 | 2026-01-06 | 初回リリース |
| 1.1.1 | 2026-05-02 | プロジェクト名を pbAudioStats へ変更 |
