cmake_minimum_required(VERSION 3.18)
project(qwen3_tts_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CUDA ---
set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA root")
set(CUDA_INCLUDE_DIR "${CUDA_ROOT}/include")
find_library(CUDA_RUNTIME_LIB cudart HINTS "${CUDA_ROOT}/lib64" "${CUDA_ROOT}/lib")
if(NOT CUDA_RUNTIME_LIB)
    message(FATAL_ERROR "cudart not found. Set CUDA_ROOT.")
endif()
find_library(CURAND_LIB curand HINTS "${CUDA_ROOT}/lib64" "${CUDA_ROOT}/lib")
if(NOT CURAND_LIB)
    message(FATAL_ERROR "curand not found. Set CUDA_ROOT.")
endif()
include_directories(${CUDA_INCLUDE_DIR})

# CUDA architecture: SM 8.7 for Jetson Orin
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_STANDARD 17)

# --- TensorRT ---
# On Jetson, headers are in /usr/include/aarch64-linux-gnu/
# Libs are in /usr/lib/aarch64-linux-gnu/
find_path(TRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include /usr/local/include)
find_library(TRT_LIB nvinfer
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib)

if(NOT TRT_INCLUDE_DIR OR NOT TRT_LIB)
    message(FATAL_ERROR "TensorRT not found. Set TRT_INCLUDE_DIR and TRT_LIB.")
endif()
message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib: ${TRT_LIB}")

# --- ONNX Runtime ---
# Download: https://github.com/microsoft/onnxruntime/releases
# Or set ORT_ROOT to extracted directory
set(ORT_ROOT "" CACHE PATH "ONNX Runtime root directory")
if(ORT_ROOT)
    set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
    find_library(ORT_LIB onnxruntime HINTS "${ORT_ROOT}/lib")
else()
    # Try system paths and Python site-packages
    find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS /usr/include /usr/local/include
              /usr/local/lib/python3.10/dist-packages/onnxruntime/capi/include)
    find_library(ORT_LIB onnxruntime
        HINTS /usr/lib /usr/local/lib
              /usr/local/lib/python3.10/dist-packages/onnxruntime/capi)
endif()

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found. Set -DORT_ROOT=/path/to/onnxruntime\n"
        "Download from: https://github.com/microsoft/onnxruntime/releases")
endif()
message(STATUS "ORT include: ${ORT_INCLUDE_DIR}")
message(STATUS "ORT lib: ${ORT_LIB}")

# --- Build ---
include_directories(
    ${TRT_INCLUDE_DIR}
    ${ORT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(qwen3_tts
    main.cpp
    tts_trt_engine.cpp
    tts_ort_models.cpp
    tts_pipeline.cpp
    asr_pipeline.cpp
    cp_sample_kernel.cu
)

target_link_libraries(qwen3_tts
    ${TRT_LIB}
    ${ORT_LIB}
    ${CUDA_RUNTIME_LIB}
    ${CURAND_LIB}
    pthread
)

# --- pybind11 module (optional) ---
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # Try pip-installed pybind11
    execute_process(
        COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYBIND11_CMAKE_DIR)
        set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 REQUIRED)
    endif()
endif()

if(pybind11_FOUND)
    message(STATUS "pybind11 found — building Python module")
    pybind11_add_module(qwen3_speech_engine
        tts_binding.cpp
        tts_trt_engine.cpp
        tts_ort_models.cpp
        tts_pipeline.cpp
        asr_pipeline.cpp
        cp_sample_kernel.cu
    )
    target_link_libraries(qwen3_speech_engine PRIVATE
        ${TRT_LIB}
        ${ORT_LIB}
        ${CUDA_RUNTIME_LIB}
        ${CURAND_LIB}
    )
else()
    message(STATUS "pybind11 not found — skipping Python module")
endif()

# Install
install(TARGETS qwen3_tts DESTINATION bin)
