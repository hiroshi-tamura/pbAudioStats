cmake_minimum_required(VERSION 3.16)
project(pb_audio_stats VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC: Use static runtime library (/MT)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# ============================================================================
# SIMD Compiler Flags
# ============================================================================
if(MSVC)
    # MSVC: Enable AVX2 for x86-64
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        add_compile_options(/arch:AVX2)
    endif()
else()
    # GCC/Clang
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        # x86-64: Enable AVX2 and FMA
        add_compile_options(-mavx2 -mfma)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
        # ARM64: NEON is enabled by default, no additional flags needed
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
# Library: pb_audio_stats
# ============================================================================

add_library(pb_audio_stats_lib STATIC
    lib/src/pb_audio_reader.cpp
    lib/src/pb_loudness.cpp
    lib/src/pb_truepeak.cpp
    lib/src/pb_rms.cpp
    lib/src/pb_analyze.cpp
    lib/src/pb_normalize.cpp
)

target_include_directories(pb_audio_stats_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/lib/include
    ${EXT_DIR}
)

# Define DR_MP3_IMPLEMENTATION for MP3 support
target_compile_definitions(pb_audio_stats_lib PRIVATE DR_MP3_IMPLEMENTATION)

# Explicitly set library output directory
set_target_properties(pb_audio_stats_lib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# ============================================================================
# CLI Executable: pb_audio_stats
# ============================================================================

add_executable(pb_audio_stats
    cli/main.cpp
)

target_link_libraries(pb_audio_stats PRIVATE pb_audio_stats_lib)

target_include_directories(pb_audio_stats PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/include
    ${EXT_DIR}
)

# Explicitly set CLI output directory
set_target_properties(pb_audio_stats PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cli
)

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(pb_audio_stats PRIVATE Threads::Threads)

# Platform-specific settings
if(APPLE)
    # macOS specific
    target_compile_definitions(pb_audio_stats PRIVATE _DARWIN_C_SOURCE)
elseif(WIN32)
    # Windows specific
    target_compile_definitions(pb_audio_stats PRIVATE _WIN32_WINNT=0x0601 NOMINMAX)
endif()

# Install targets
install(TARGETS pb_audio_stats
    RUNTIME DESTINATION bin
)

install(TARGETS pb_audio_stats_lib
    ARCHIVE DESTINATION lib
)

install(FILES lib/include/pb_audio_stats.h
    DESTINATION include
)
