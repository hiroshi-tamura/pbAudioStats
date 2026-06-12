cmake_minimum_required(VERSION 3.16)
project(pbAudioStats VERSION 1.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC: Use static runtime library (/MT)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# ============================================================================
# SIMD / FP compiler flags (applied per-target so the CPU-guard translation
# unit can be compiled WITHOUT AVX2 — it must run on pre-AVX2 CPUs to print
# a diagnostic instead of crashing with an illegal instruction).
#
# /fp:contract (MSVC) and -ffp-contract=fast (GCC/Clang) allow FMA contraction
# in the scalar double hot loops. Contraction performs a*b+c with a single
# rounding, which is MORE accurate than separate multiply+add. Deliberately
# NOT /fp:fast or -ffast-math: those would break the Neumaier compensated
# summation used for long-file loudness accumulation.
# ============================================================================
set(PB_SIMD_FLAGS "")
set(PB_IS_X86 FALSE)
if(MSVC)
    add_compile_options(/MP)      # parallel compilation
    add_compile_options(/utf-8)   # sources are UTF-8 (Japanese comments)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
        set(PB_IS_X86 TRUE)
        list(APPEND PB_SIMD_FLAGS /arch:AVX2)
        if(MSVC_VERSION GREATER_EQUAL 1930)  # VS2022+: /fp:contract
            list(APPEND PB_SIMD_FLAGS /fp:contract)
        endif()
    endif()
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(PB_IS_X86 TRUE)
        list(APPEND PB_SIMD_FLAGS -mavx2 -mfma -ffp-contract=fast)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
        # ARM64: NEON is enabled by default
        list(APPEND PB_SIMD_FLAGS -ffp-contract=fast)
    endif()
endif()

# ============================================================================
# Optimization flags and LTO (Link Time Optimization)
# ============================================================================
if(MSVC)
    # MSVC Release optimization
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
    # MSVC LTO (Whole Program Optimization) - use with caution for header-only libs
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
else()
    # GCC/Clang Release optimization
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    # Enable LTO for Release builds (GCC/Clang)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

# External headers (dr_libs)
set(EXT_DIR ${CMAKE_SOURCE_DIR}/ext)

# ============================================================================
# Output directories: lib/ for library, cli/ for CLI executable
# ============================================================================
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cli)

# ============================================================================
# Library: pbAudioStatsLib
# ============================================================================

add_library(pbAudioStatsLib STATIC
    lib/src/pbAudioReader.cpp
    lib/src/pbLoudness.cpp
    lib/src/pbTruePeak.cpp
    lib/src/pbRms.cpp
    lib/src/pbAnalyze.cpp
    lib/src/pbNormalize.cpp
)

target_include_directories(pbAudioStatsLib PUBLIC
    ${CMAKE_SOURCE_DIR}/lib/include
    ${EXT_DIR}
)

target_compile_options(pbAudioStatsLib PRIVATE ${PB_SIMD_FLAGS})

# Define DR_MP3_IMPLEMENTATION for MP3 support
target_compile_definitions(pbAudioStatsLib PRIVATE DR_MP3_IMPLEMENTATION)

# Explicitly set library output directory
set_target_properties(pbAudioStatsLib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# ============================================================================
# CPU guard: compiled WITHOUT AVX2 so it can run anywhere; contains main().
# ============================================================================

add_library(pbCpuGuard STATIC cli/cpu_check.cpp)
if(PB_IS_X86)
    target_compile_definitions(pbCpuGuard PRIVATE PB_REQUIRE_AVX2)
endif()

# ============================================================================
# CLI Executable: pbAudioStats
# ============================================================================

add_executable(pbAudioStats
    cli/main.cpp
)

target_compile_options(pbAudioStats PRIVATE ${PB_SIMD_FLAGS})

target_link_libraries(pbAudioStats PRIVATE pbAudioStatsLib pbCpuGuard)

target_include_directories(pbAudioStats PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/include
    ${EXT_DIR}
)

# Explicitly set CLI output directory
set_target_properties(pbAudioStats PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cli
)

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(pbAudioStats PRIVATE Threads::Threads)

# Platform-specific settings
if(APPLE)
    # macOS specific
    target_compile_definitions(pbAudioStats PRIVATE _DARWIN_C_SOURCE)
elseif(WIN32)
    # Windows specific
    target_compile_definitions(pbAudioStats PRIVATE _WIN32_WINNT=0x0601 NOMINMAX)
endif()

# ============================================================================
# Dev tool: per-stage timing micro-benchmark (not built by default)
#   cmake --build . --config Release --target stage_bench
# ============================================================================
add_executable(stage_bench EXCLUDE_FROM_ALL tools/stage_bench.cpp)
target_compile_options(stage_bench PRIVATE ${PB_SIMD_FLAGS})
target_link_libraries(stage_bench PRIVATE pbAudioStatsLib)
target_include_directories(stage_bench PRIVATE ${CMAKE_SOURCE_DIR}/lib/include ${EXT_DIR})
set_target_properties(stage_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cli)

# Install targets
install(TARGETS pbAudioStats
    RUNTIME DESTINATION bin
)

install(TARGETS pbAudioStatsLib
    ARCHIVE DESTINATION lib
)

install(FILES lib/include/pbAudioStats.h
    DESTINATION include
)
