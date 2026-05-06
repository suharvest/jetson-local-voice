# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)
project(jetson_voice_edgellm_workers LANGUAGES CXX CUDA)

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
