/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/trtUtils.h"
#include "requestFileParser.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{
struct Args
{
    std::string engineDir;
    std::string multimodalEngineDir;
    bool debug{false};
};

enum OptionId : int
{
    HELP = 1000,
    ENGINE_DIR,
    MULTIMODAL_ENGINE_DIR,
    DEBUG,
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " --engineDir=<path> --multimodalEngineDir=<path> [--debug]\n\n"
              << "Reads llm_inference-compatible JSON lines from stdin and writes JSON lines to stdout.\n";
}

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[] = {{"help", no_argument, 0, HELP},
        {"engineDir", required_argument, 0, ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, MULTIMODAL_ENGINE_DIR}, {"debug", no_argument, 0, DEBUG},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: printUsage(argv[0]); std::exit(EXIT_SUCCESS);
        case ENGINE_DIR: args.engineDir = optarg; break;
        case MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
        case DEBUG: args.debug = true; break;
        default: return false;
        }
    }

    return !args.engineDir.empty() && !args.multimodalEngineDir.empty();
}

std::filesystem::path writeTempInput(Json const& input, std::string const& id)
{
    std::string safeId = id.empty() ? "request" : id;
    for (auto& ch : safeId)
    {
        bool const ok = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_'
            || ch == '-';
        if (!ok)
        {
            ch = '_';
        }
    }
    auto path = std::filesystem::temp_directory_path()
        / ("qwen3_asr_worker_" + safeId + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
            + ".json");
    std::ofstream file(path);
    if (!file)
    {
        throw std::runtime_error("Failed to open temp input file: " + path.string());
    }
    file << input.dump();
    return path;
}
} // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    gLogger.setLevel(args.debug ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING);
    auto pluginHandles = loadEdgellmPluginLib();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto const initStart = std::chrono::steady_clock::now();
    std::unordered_map<std::string, std::string> loraWeightsMap;
    auto runtime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(
        args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
    bool const enableGraph = std::getenv("EDGE_LLM_ASR_CUDA_GRAPH") == nullptr
        || std::string(std::getenv("EDGE_LLM_ASR_CUDA_GRAPH")) != "0";
    if (enableGraph && !runtime->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("CUDA graph capture failed for ASR worker, proceeding without.");
    }
    double const initMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - initStart).count();
    std::cout << Json{{"event", "ready"}, {"init_ms", initMs}}.dump() << std::endl;

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            continue;
        }

        Json response;
        std::filesystem::path tempPath;
        auto const requestStart = std::chrono::steady_clock::now();
        try
        {
            Json input = Json::parse(line);
            std::string const id = input.value("id", "");
            input.erase("id");
            int32_t const batchSizeOverride = input.value("batch_size_override", -1);
            int64_t const maxGenerateLengthOverride = input.value("max_generate_length_override", -1);
            input.erase("batch_size_override");
            input.erase("max_generate_length_override");

            tempPath = writeTempInput(input, id);
            std::vector<rt::LLMGenerationRequest> batchedRequests;
            std::tie(loraWeightsMap, batchedRequests)
                = exampleUtils::parseRequestFile(tempPath, batchSizeOverride, maxGenerateLengthOverride);
            if (batchedRequests.empty())
            {
                throw std::runtime_error("No valid ASR requests found");
            }

            Json responses = Json::array();
            bool ok = true;
            for (size_t requestIdx = 0; requestIdx < batchedRequests.size(); ++requestIdx)
            {
                rt::LLMGenerationResponse llmResponse;
                bool const requestOk = runtime->handleRequest(batchedRequests[requestIdx], llmResponse, stream);
                ok = ok && requestOk;
                for (size_t batchIdx = 0; batchIdx < batchedRequests[requestIdx].requests.size(); ++batchIdx)
                {
                    bool const hasOutputText = requestOk && batchIdx < llmResponse.outputTexts.size();
                    std::string const text
                        = hasOutputText ? llmResponse.outputTexts[batchIdx] : "TensorRT Edge LLM cannot handle this request. Fails.";
                    responses.push_back(Json{{"request_idx", requestIdx},
                        {"batch_idx", batchIdx},
                        {"output_text", text}});
                }
            }
            double const totalMs
                = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - requestStart).count();
            response = Json{{"id", id}, {"event", ok ? "done" : "error"}, {"ok", ok}, {"responses", responses},
                {"total_ms", totalMs}};
        }
        catch (std::exception const& e)
        {
            response = Json{{"event", "error"}, {"ok", false}, {"error", e.what()}};
        }
        if (!tempPath.empty())
        {
            std::error_code ec;
            std::filesystem::remove(tempPath, ec);
        }
        std::cout << response.dump() << std::endl;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
