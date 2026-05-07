# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)
project(jetson_voice_edgellm_workers LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)

set(EDGE_LLM_SOURCE_DIR
    "$ENV{EDGE_LLM_BASE}"
    CACHE PATH "Path to a TensorRT-Edge-LLM source checkout")
set(EDGE_LLM_BUILD_DIR
    "$ENV{EDGE_LLM_BUILD}"
    CACHE PATH "Path to a configured TensorRT-Edge-LLM build directory")

if(NOT EDGE_LLM_SOURCE_DIR)
  set(EDGE_LLM_SOURCE_DIR "$ENV{HOME}/project/tensorrt-edge-llm")
endif()
if(NOT EDGE_LLM_BUILD_DIR)
  set(EDGE_LLM_BUILD_DIR "${EDGE_LLM_SOURCE_DIR}/build_sm87")
endif()

if(NOT EXISTS "${EDGE_LLM_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "EDGE_LLM_SOURCE_DIR must point to a TensorRT-Edge-LLM checkout")
endif()
if(NOT EXISTS "${EDGE_LLM_BUILD_DIR}/cpp/libedgellmCore.a")
  message(
    FATAL_ERROR
      "EDGE_LLM_BUILD_DIR must point to a built TensorRT-Edge-LLM tree with cpp/libedgellmCore.a"
  )
endif()

function(read_edgellm_cache_entry out_var cache_name)
  set(${out_var}
      ""
      PARENT_SCOPE)
  if(NOT EXISTS "${EDGE_LLM_BUILD_DIR}/CMakeCache.txt")
    return()
  endif()
  file(STRINGS "${EDGE_LLM_BUILD_DIR}/CMakeCache.txt" _cache_lines
       REGEX "^${cache_name}:[^=]*=")
  if(NOT _cache_lines)
    return()
  endif()
  list(GET _cache_lines 0 _cache_line)
  string(REGEX REPLACE "^${cache_name}:[^=]*=" "" _cache_value
                       "${_cache_line}")
  set(${out_var}
      "${_cache_value}"
      PARENT_SCOPE)
endfunction()

macro(set_ifndef var val)
  if(NOT DEFINED ${var})
    set(${var} ${val})
  endif()
endmacro()

macro(add_cross_build_link_options target)
  if(DEFINED AARCH64_BUILD)
    target_link_options(${target} PRIVATE "-L${CUDA_DIR}/lib")
    target_link_options(${target} PRIVATE "-L${CUDA_DIR}/lib/stubs")
  endif()
endmacro()

set_ifndef(CUDA_CTK_VERSION 12.6)
set_ifndef(CUDA_DIR /usr/local/cuda-${CUDA_CTK_VERSION})

find_path(
  CUDA_RUNTIME_API_INCLUDE_DIR cuda_runtime_api.h
  HINTS ${CUDA_DIR}
  PATH_SUFFIXES include)
find_path(
  CURAND_KERNEL_INCLUDE_DIR
  NAMES curand_kernel.h
  HINTS ${CUDA_DIR}
  PATH_SUFFIXES include)
find_library(
  CUDART_LIB cudart
  HINTS ${CUDA_DIR}
  PATH_SUFFIXES lib lib64)
find_library(
  CUDA_DRIVER_LIB cuda
  HINTS ${CUDA_DIR}
  PATH_SUFFIXES lib lib64 lib/stubs)
find_library(
  CUBLAS_LIB cublas
  HINTS ${CUDA_DIR}
  PATH_SUFFIXES lib lib64)
find_library(
  CUBLASLT_LIB cublasLt
  HINTS ${CUDA_DIR}
  PATH_SUFFIXES lib lib64)

read_edgellm_cache_entry(EDGE_LLM_BUILD_ENABLE_CUTE_DSL ENABLE_CUTE_DSL)
read_edgellm_cache_entry(EDGE_LLM_BUILD_CUTE_DSL_ARTIFACT_TAG
                         CUTE_DSL_ARTIFACT_TAG)
read_edgellm_cache_entry(EDGE_LLM_BUILD_EMBEDDED_TARGET EMBEDDED_TARGET)
set_ifndef(ENABLE_CUTE_DSL
           "${EDGE_LLM_BUILD_ENABLE_CUTE_DSL}")
if(NOT ENABLE_CUTE_DSL)
  set(ENABLE_CUTE_DSL OFF)
endif()
set_ifndef(CUTE_DSL_ARTIFACT_TAG
           "${EDGE_LLM_BUILD_CUTE_DSL_ARTIFACT_TAG}")
set_ifndef(EMBEDDED_TARGET "${EDGE_LLM_BUILD_EMBEDDED_TARGET}")

list(APPEND CMAKE_MODULE_PATH "${EDGE_LLM_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED COMPONENTS OnnxParser)

set(COMMON_INCLUDE_DIRS
    ${CUDA_RUNTIME_API_INCLUDE_DIR}
    ${CURAND_KERNEL_INCLUDE_DIR}
    ${EDGE_LLM_SOURCE_DIR}/cpp
    ${EDGE_LLM_SOURCE_DIR}/examples/multimodal
    ${EDGE_LLM_SOURCE_DIR}/3rdParty/nlohmannJson/include
    ${EDGE_LLM_SOURCE_DIR}/3rdParty/stb)

add_library(edgellmCore STATIC IMPORTED GLOBAL)
set_target_properties(
  edgellmCore PROPERTIES IMPORTED_LOCATION
                         "${EDGE_LLM_BUILD_DIR}/cpp/libedgellmCore.a")

add_library(commonLibraryExt INTERFACE)
target_link_libraries(
  commonLibraryExt INTERFACE TensorRT::TensorRT TensorRT::OnnxParser
                             ${CUDART_LIB} ${CUDA_DRIVER_LIB} ${CUBLAS_LIB}
                             ${CUBLASLT_LIB} ${CMAKE_DL_LIBS})

add_library(voiceWorkerUtils STATIC
            "${EDGE_LLM_SOURCE_DIR}/examples/utils/audioWriter.cpp"
            "${EDGE_LLM_SOURCE_DIR}/examples/utils/requestFileParser.cpp")
target_include_directories(
  voiceWorkerUtils PUBLIC ${EDGE_LLM_SOURCE_DIR}/examples/utils
                          ${COMMON_INCLUDE_DIRS})
target_link_libraries(voiceWorkerUtils PRIVATE edgellmCore commonLibraryExt)

set(VOICE_WORKER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/workers")

add_executable(qwen3_tts_worker qwen3_tts_worker.cpp)
target_include_directories(
  qwen3_tts_worker PRIVATE ${COMMON_INCLUDE_DIRS}
                           ${EDGE_LLM_SOURCE_DIR}/examples/utils)
target_link_libraries(qwen3_tts_worker PRIVATE voiceWorkerUtils edgellmCore
                                               commonLibraryExt)
add_cross_build_link_options(qwen3_tts_worker)
set_target_properties(qwen3_tts_worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  ${VOICE_WORKER_OUTPUT_DIR}
                                                  LINKER_LANGUAGE CUDA
                                                  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(qwen3_asr_worker qwen3_asr_worker.cpp)
target_include_directories(
  qwen3_asr_worker PRIVATE ${COMMON_INCLUDE_DIRS}
                           ${EDGE_LLM_SOURCE_DIR}/examples/utils)
target_link_libraries(qwen3_asr_worker PRIVATE voiceWorkerUtils edgellmCore
                                               commonLibraryExt)
add_cross_build_link_options(qwen3_asr_worker)
set_target_properties(qwen3_asr_worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  ${VOICE_WORKER_OUTPUT_DIR}
                                                  LINKER_LANGUAGE CUDA
                                                  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

function(link_edgellm_cutedsl_if_enabled target)
  string(TOUPPER "${ENABLE_CUTE_DSL}" _enable_cutedsl)
  if(_enable_cutedsl STREQUAL "" OR _enable_cutedsl STREQUAL "OFF")
    return()
  endif()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_cutedsl_arch aarch64)
  else()
    set(_cutedsl_arch x86_64)
  endif()

  set(_artifact_tag "${CUTE_DSL_ARTIFACT_TAG}")
  if(_artifact_tag STREQUAL "" AND _cutedsl_arch STREQUAL "aarch64")
    string(TOLOWER "${EMBEDDED_TARGET}" _embedded_target)
    string(REPLACE "-" "_" _embedded_target "${_embedded_target}")
    if(_embedded_target STREQUAL "jetson_orin")
      set(_artifact_tag sm_87)
    elseif(_embedded_target STREQUAL "jetson_thor"
           OR _embedded_target STREQUAL "auto_thor")
      set(_artifact_tag sm_110)
    elseif(_embedded_target STREQUAL "gb10")
      set(_artifact_tag sm_121)
    endif()
  endif()

  if(_artifact_tag STREQUAL "")
    message(FATAL_ERROR "ENABLE_CUTE_DSL=${ENABLE_CUTE_DSL}, but no "
                        "CUTE_DSL_ARTIFACT_TAG was found. Pass "
                        "-DCUTE_DSL_ARTIFACT_TAG=<tag> when configuring the "
                        "worker.")
  endif()

  set(_cutedsl_dir
      "${EDGE_LLM_SOURCE_DIR}/cpp/kernels/cuteDSLArtifact/${_cutedsl_arch}/${_artifact_tag}"
  )
  set(_cutedsl_lib "${_cutedsl_dir}/libcutedsl_${_cutedsl_arch}.a")
  if(NOT EXISTS "${_cutedsl_lib}")
    message(FATAL_ERROR "CuTe DSL artifact library not found: ${_cutedsl_lib}")
  endif()

  if(NOT TARGET edgellmCuteDslArtifact)
    add_library(edgellmCuteDslArtifact STATIC IMPORTED GLOBAL)
    set_target_properties(edgellmCuteDslArtifact
                          PROPERTIES IMPORTED_LOCATION "${_cutedsl_lib}")
  endif()

  if(NOT TARGET edgellmCuteDslCudartShim)
    if(CUDA_CTK_VERSION VERSION_LESS 12.8)
      add_library(
        edgellmCuteDslCudartShim STATIC
        "${EDGE_LLM_SOURCE_DIR}/cpp/kernels/gdnKernels/cutedsl_cuda_runtime_library_shim.c"
      )
      target_include_directories(edgellmCuteDslCudartShim
                                 PRIVATE ${CUDA_RUNTIME_API_INCLUDE_DIR})
      if(CUDA_CTK_VERSION VERSION_GREATER_EQUAL 12.0)
        target_compile_definitions(edgellmCuteDslCudartShim
                                   PRIVATE CUTEDSL_WRAP_LAUNCH_KERNEL_EX)
      endif()
    else()
      add_library(edgellmCuteDslCudartShim INTERFACE)
    endif()
  endif()

  target_link_libraries(${target} PRIVATE edgellmCuteDslArtifact
                                          edgellmCuteDslCudartShim)
  if(CUDA_DRIVER_LIB AND NOT CUDA_DRIVER_LIB MATCHES "-NOTFOUND$")
    target_link_libraries(${target} PRIVATE ${CUDA_DRIVER_LIB})
  endif()
  if(CUDA_CTK_VERSION VERSION_LESS 12.8)
    target_link_options(${target} PRIVATE "-Wl,--wrap=_cudaLaunchKernelEx")
  endif()
endfunction()

link_edgellm_cutedsl_if_enabled(qwen3_tts_worker)
link_edgellm_cutedsl_if_enabled(qwen3_asr_worker)
