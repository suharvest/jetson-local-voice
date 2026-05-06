/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audioWriter.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/trtUtils.h"
#include "multimodal/code2WavRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/qwen3OmniTTSRuntime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;

namespace
{
struct Args
{
    std::string talkerEngineDir;
    std::string codePredictorEngineDir;
    std::string code2wavEngineDir;
    std::string tokenizerDir;
    bool debug{false};
};

struct ChunkJob
{
    std::vector<std::vector<int32_t>> windowCodes;
    int32_t skipContextFrames{0};
    int32_t totalFrames{0};
    int32_t chunkIndex{0};
    bool isFinal{false};
};

struct ScopeExit
{
    std::function<void()> fn;
    ~ScopeExit()
    {
        if (fn)
        {
            fn();
        }
    }
    void dismiss()
    {
        fn = nullptr;
    }
};

enum OptionId : int
{
    HELP = 1000,
    TALKER_ENGINE_DIR,
    CODE_PREDICTOR_ENGINE_DIR,
    CODE2WAV_ENGINE_DIR,
    TOKENIZER_DIR,
    DEBUG,
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " --talkerEngineDir=<path> --code2wavEngineDir=<path>"
              << " [--codePredictorEngineDir=<path>] [--tokenizerDir=<path>] [--debug]\n\n"
              << "Reads JSON lines from stdin:\n"
              << "  {\"id\":\"1\",\"text\":\"你好。\",\"output_file\":\"/tmp/out.wav\",\"max_audio_length\":80}\n"
              << "Writes JSON lines to stdout.\n";
}

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[] = {{"help", no_argument, 0, HELP},
        {"talkerEngineDir", required_argument, 0, TALKER_ENGINE_DIR},
        {"codePredictorEngineDir", required_argument, 0, CODE_PREDICTOR_ENGINE_DIR},
        {"code2wavEngineDir", required_argument, 0, CODE2WAV_ENGINE_DIR},
        {"tokenizerDir", required_argument, 0, TOKENIZER_DIR}, {"debug", no_argument, 0, DEBUG}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: printUsage(argv[0]); std::exit(EXIT_SUCCESS);
        case TALKER_ENGINE_DIR: args.talkerEngineDir = optarg; break;
        case CODE_PREDICTOR_ENGINE_DIR: args.codePredictorEngineDir = optarg; break;
        case CODE2WAV_ENGINE_DIR: args.code2wavEngineDir = optarg; break;
        case TOKENIZER_DIR: args.tokenizerDir = optarg; break;
        case DEBUG: args.debug = true; break;
        default: return false;
        }
    }

    return !args.talkerEngineDir.empty() && !args.code2wavEngineDir.empty();
}

std::vector<std::vector<int32_t>> transposeCodes(std::vector<std::vector<int32_t>> const& frames)
{
    if (frames.empty())
    {
        return {};
    }
    size_t const numFrames = frames.size();
    size_t const numLayers = frames[0].size();
    std::vector<std::vector<int32_t>> transposed(numLayers, std::vector<int32_t>(numFrames));
    for (size_t f = 0; f < numFrames; ++f)
    {
        for (size_t l = 0; l < numLayers; ++l)
        {
            transposed[l][f] = frames[f][l];
        }
    }
    return transposed;
}

std::vector<std::vector<int32_t>> transposeFrameWindow(
    std::vector<std::vector<int32_t>> const& frames, size_t begin, size_t end)
{
    if (begin >= end || end > frames.size())
    {
        return {};
    }
    size_t const numFrames = end - begin;
    size_t const numLayers = frames[begin].size();
    std::vector<std::vector<int32_t>> transposed(numLayers, std::vector<int32_t>(numFrames));
    for (size_t f = 0; f < numFrames; ++f)
    {
        for (size_t l = 0; l < numLayers; ++l)
        {
            transposed[l][f] = frames[begin + f][l];
        }
    }
    return transposed;
}

bool saveFloatSamplesToWav(std::string const& filepath, std::vector<float> const& samples, int32_t sampleRate)
{
    if (samples.empty())
    {
        return false;
    }

    rt::audioUtils::AudioData audio;
    audio.sampleRate = sampleRate;
    audio.numChannels = 1;
    audio.hasWaveform = true;
    audio.waveform = std::make_shared<rt::Tensor>(
        rt::Tensor({1, static_cast<int64_t>(samples.size())}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT));
    std::memcpy(audio.waveform->rawPointer(), samples.data(), samples.size() * sizeof(float));
    return saveAudioToWav(filepath, audio);
}

std::vector<int16_t> floatSamplesToPcm16(std::vector<float> const& samples)
{
    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); ++i)
    {
        float const clipped = std::clamp(samples[i], -1.0f, 1.0f);
        pcm[i] = static_cast<int16_t>(std::lrint(clipped * 32767.0f));
    }
    return pcm;
}

bool savePcm16(std::string const& filepath, std::vector<int16_t> const& samples)
{
    std::ofstream file(filepath, std::ios::binary);
    if (!file)
    {
        return false;
    }
    file.write(reinterpret_cast<char const*>(samples.data()),
        static_cast<std::streamsize>(samples.size() * sizeof(int16_t)));
    return static_cast<bool>(file);
}

std::string base64Encode(uint8_t const* data, size_t len)
{
    static constexpr char kTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3)
    {
        uint32_t const b0 = data[i];
        uint32_t const b1 = (i + 1 < len) ? data[i + 1] : 0;
        uint32_t const b2 = (i + 2 < len) ? data[i + 2] : 0;
        uint32_t const triple = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kTable[(triple >> 18) & 0x3F]);
        out.push_back(kTable[(triple >> 12) & 0x3F]);
        out.push_back((i + 1 < len) ? kTable[(triple >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < len) ? kTable[triple & 0x3F] : '=');
    }
    return out;
}

std::string base64EncodePcm16(std::vector<int16_t> const& samples)
{
    return base64Encode(reinterpret_cast<uint8_t const*>(samples.data()), samples.size() * sizeof(int16_t));
}

Qwen3OmniTTSRuntime::TalkerGenerationRequest buildRequest(Json const& item)
{
    Qwen3OmniTTSRuntime::TalkerGenerationRequest req;
    req.talkerTemperature = item.value("talker_temperature", 0.9f);
    req.talkerTopK = item.value("talker_top_k", 50);
    req.talkerTopP = item.value("talker_top_p", 1.0f);
    req.repetitionPenalty = item.value("repetition_penalty", 1.05f);
    req.codecEosLogitOffset = item.value("codec_eos_logit_offset", 0.0f);
    req.predictorTemperature = item.value("predictor_temperature", 0.0f);
    req.predictorTopK = item.value("predictor_top_k", 0);
    req.predictorTopP = item.value("predictor_top_p", 0.0f);
    req.maxAudioLength = item.value("max_audio_length", 4096);
    req.minAudioLength = item.value("min_audio_length", 2);
    req.language = item.value("language", "");
    req.speakerName = item.value("speaker", "");

    Message msg;
    msg.role = "user";
    Message::MessageContent content;
    content.type = "text";
    content.content = item.value("text", "");
    msg.contents.push_back(std::move(content));
    req.messages.push_back(std::move(msg));
    return req;
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
    cudaStream_t code2wavStream;
    CUDA_CHECK(cudaStreamCreate(&code2wavStream));

    std::filesystem::path const codePredictorDir = args.codePredictorEngineDir.empty()
        ? std::filesystem::path(args.talkerEngineDir).parent_path() / "code_predictor"
        : std::filesystem::path(args.codePredictorEngineDir);

    auto const initStart = std::chrono::steady_clock::now();
    auto ttsRuntime = std::make_unique<Qwen3OmniTTSRuntime>(
        args.talkerEngineDir, codePredictorDir.string(), args.tokenizerDir, stream);
    bool const lazyCode2Wav = std::getenv("EDGE_LLM_TTS_LAZY_CODE2WAV") != nullptr
        && std::string(std::getenv("EDGE_LLM_TTS_LAZY_CODE2WAV")) == "1";
    std::unique_ptr<Code2WavRunner> code2wavRunner;
    if (!lazyCode2Wav)
    {
        code2wavRunner = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, stream);
    }
    bool const enableGraph = std::getenv("EDGE_LLM_TTS_CUDA_GRAPH") == nullptr
        || std::string(std::getenv("EDGE_LLM_TTS_CUDA_GRAPH")) != "0";
    if (enableGraph && !ttsRuntime->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("CUDA graph capture failed for TTS worker, proceeding without.");
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
        auto const requestStart = std::chrono::steady_clock::now();
        try
        {
            Json item = Json::parse(line);
            std::string const id = item.value("id", "");
            bool const streamOutput = item.value("stream", false);
            bool const streamOnly = item.value("stream_only", false);
            bool const asyncCode2Wav = streamOnly
                && item.value("async_code2wav",
                    std::getenv("EDGE_LLM_TTS_ASYNC_CODE2WAV") != nullptr
                        && std::string(std::getenv("EDGE_LLM_TTS_ASYNC_CODE2WAV")) != "0");
            int32_t const firstChunkFrames = std::max(1, item.value("first_chunk_frames", 5));
            int32_t const chunkFrames = std::max(1, item.value("chunk_frames", 25));
            bool const adaptiveChunks = item.value("adaptive_chunks",
                std::getenv("EDGE_LLM_TTS_ADAPTIVE_CHUNKS") != nullptr
                    && std::string(std::getenv("EDGE_LLM_TTS_ADAPTIVE_CHUNKS")) != "0");
            int32_t const maxChunkFrames = std::max(chunkFrames, item.value("max_chunk_frames", chunkFrames * 6));
            int32_t const chunkGrowthFrames = std::max(0, item.value("chunk_growth_frames", chunkFrames * 2));
            std::string const chunkFormat = item.value("chunk_format", "wav");
            std::string const chunkTransport = item.value("chunk_transport", "file");
            std::string outputFile = item.value("output_file", "");
            if (outputFile.empty())
            {
                outputFile = "/tmp/qwen3_tts_worker_" + id + ".wav";
            }
            if (streamOnly && !streamOutput)
            {
                throw std::runtime_error("stream_only requires stream=true");
            }
            if (chunkFormat != "wav" && chunkFormat != "pcm_s16le")
            {
                throw std::runtime_error("Unsupported chunk_format: " + chunkFormat);
            }
            if (chunkTransport != "file" && chunkTransport != "base64")
            {
                throw std::runtime_error("Unsupported chunk_transport: " + chunkTransport);
            }
            if (chunkFormat == "wav" && chunkTransport != "file")
            {
                throw std::runtime_error("wav chunks currently support file transport only");
            }

            auto request = buildRequest(item);
            Qwen3OmniTTSRuntime::TalkerGenerationResponse talkerResponse;
            auto const genStart = std::chrono::steady_clock::now();

            std::vector<std::vector<int32_t>> streamedFrames;
            int32_t lastEmittedFrames = 0;
            int32_t nextChunkAt = firstChunkFrames;
            int32_t currentChunkFrames = chunkFrames;
            int32_t chunkIndex = 0;
            int64_t streamedSamples = 0;
            double streamedCode2WavMs = 0.0;
            std::chrono::steady_clock::time_point firstChunkAt{};

            std::deque<ChunkJob> chunkJobs;
            std::mutex chunkMutex;
            std::condition_variable chunkCv;
            bool chunkJobsClosed = false;
            std::exception_ptr chunkException;
            std::thread chunkThread;

            auto processChunk = [&](ChunkJob const& job) {
                if (!code2wavRunner)
                {
                    code2wavRunner = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, code2wavStream);
                }

                std::vector<float> samples;
                auto const chunkStart = std::chrono::steady_clock::now();
                cudaStream_t const chunkStream = asyncCode2Wav ? code2wavStream : stream;
                if (!code2wavRunner->generateWaveformChunk(job.windowCodes, job.skipContextFrames, samples, chunkStream))
                {
                    throw std::runtime_error("Code2Wav streaming chunk failed");
                }
                auto const chunkEnd = std::chrono::steady_clock::now();
                if (samples.empty())
                {
                    return;
                }

                if (job.chunkIndex == 0)
                {
                    firstChunkAt = chunkEnd;
                }
                double const code2wavMs = std::chrono::duration<double, std::milli>(chunkEnd - chunkStart).count();
                streamedCode2WavMs += code2wavMs;
                streamedSamples += static_cast<int64_t>(samples.size());
                Json chunk = Json{{"id", id},
                    {"event", "chunk"},
                    {"ok", true},
                    {"chunk_index", job.chunkIndex},
                    {"chunk_format", chunkFormat},
                    {"chunk_transport", chunkTransport},
                    {"frames", job.totalFrames},
                    {"samples", samples.size()},
                    {"sample_rate", code2wavRunner->getConfig().sampleRate},
                    {"is_final", job.isFinal},
                    {"adaptive_chunks", adaptiveChunks},
                    {"code2wav_ms", code2wavMs},
                    {"elapsed_ms", std::chrono::duration<double, std::milli>(chunkEnd - requestStart).count()}};
                if (chunkFormat == "wav")
                {
                    std::filesystem::path chunkPath(outputFile);
                    chunkPath.replace_filename(chunkPath.stem().string() + ".chunk" + std::to_string(job.chunkIndex)
                        + chunkPath.extension().string());
                    if (!saveFloatSamplesToWav(chunkPath.string(), samples, code2wavRunner->getConfig().sampleRate))
                    {
                        throw std::runtime_error("Failed to save chunk WAV: " + chunkPath.string());
                    }
                    chunk["chunk_file"] = chunkPath.string();
                }
                else
                {
                    auto pcm = floatSamplesToPcm16(samples);
                    chunk["bytes"] = pcm.size() * sizeof(int16_t);
                    if (chunkTransport == "base64")
                    {
                        chunk["audio_b64"] = base64EncodePcm16(pcm);
                    }
                    else
                    {
                        std::filesystem::path chunkPath(outputFile);
                        chunkPath.replace_filename(chunkPath.stem().string() + ".chunk"
                            + std::to_string(job.chunkIndex) + ".pcm");
                        if (!savePcm16(chunkPath.string(), pcm))
                        {
                            throw std::runtime_error("Failed to save chunk PCM: " + chunkPath.string());
                        }
                        chunk["chunk_file"] = chunkPath.string();
                    }
                }
                std::cout << chunk.dump() << std::endl;
            };

            if (asyncCode2Wav)
            {
                chunkThread = std::thread([&]() {
                    try
                    {
                        while (true)
                        {
                            ChunkJob job;
                            {
                                std::unique_lock<std::mutex> lock(chunkMutex);
                                chunkCv.wait(lock, [&]() { return chunkJobsClosed || !chunkJobs.empty(); });
                                if (chunkJobs.empty())
                                {
                                    if (chunkJobsClosed)
                                    {
                                        break;
                                    }
                                    continue;
                                }
                                job = std::move(chunkJobs.front());
                                chunkJobs.pop_front();
                            }
                            processChunk(job);
                        }
                    }
                    catch (...)
                    {
                        std::lock_guard<std::mutex> lock(chunkMutex);
                        chunkException = std::current_exception();
                    }
                });
            }
            ScopeExit chunkThreadGuard{[&]() {
                if (asyncCode2Wav)
                {
                    {
                        std::lock_guard<std::mutex> lock(chunkMutex);
                        chunkJobsClosed = true;
                    }
                    chunkCv.notify_one();
                    if (chunkThread.joinable())
                    {
                        chunkThread.join();
                    }
                }
            }};

            auto submitChunk = [&](bool isFinal) {
                int32_t const totalFrames = static_cast<int32_t>(streamedFrames.size());
                if (totalFrames <= lastEmittedFrames)
                {
                    return;
                }
                if (!code2wavRunner)
                {
                    code2wavRunner = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, stream);
                }

                int32_t const leftContext = code2wavRunner->getConfig().leftContextSize;
                int32_t const windowStart = std::max(0, lastEmittedFrames - leftContext);
                int32_t const skipContextFrames = lastEmittedFrames - windowStart;
                auto windowCodes = transposeFrameWindow(
                    streamedFrames, static_cast<size_t>(windowStart), static_cast<size_t>(totalFrames));

                ChunkJob job;
                job.windowCodes = std::move(windowCodes);
                job.skipContextFrames = skipContextFrames;
                job.totalFrames = totalFrames;
                job.chunkIndex = chunkIndex;
                job.isFinal = isFinal;

                lastEmittedFrames = totalFrames;
                ++chunkIndex;
                if (adaptiveChunks && chunkIndex > 1)
                {
                    currentChunkFrames = std::min(maxChunkFrames, currentChunkFrames + chunkGrowthFrames);
                }
                nextChunkAt = totalFrames + currentChunkFrames;

                if (asyncCode2Wav)
                {
                    {
                        std::lock_guard<std::mutex> lock(chunkMutex);
                        chunkJobs.push_back(std::move(job));
                    }
                    chunkCv.notify_one();
                    return;
                }
                processChunk(job);
            };

            auto frameCallback = [&](std::vector<int32_t> const& frameCodes, int32_t totalFrames) {
                streamedFrames.push_back(frameCodes);
                if (streamOutput && totalFrames >= nextChunkAt)
                {
                    submitChunk(false);
                }
            };

            bool ok = streamOutput ? ttsRuntime->handleAudioGeneration(request, talkerResponse, stream, frameCallback)
                                   : ttsRuntime->handleAudioGeneration(request, talkerResponse, stream);
            auto const genEnd = std::chrono::steady_clock::now();
            if (!ok || talkerResponse.rvqCodes.empty())
            {
                throw std::runtime_error("TTS generation failed");
            }
            if (streamOutput)
            {
                submitChunk(true);
            }
            if (asyncCode2Wav)
            {
                {
                    std::lock_guard<std::mutex> lock(chunkMutex);
                    chunkJobsClosed = true;
                }
                chunkCv.notify_one();
                if (chunkThread.joinable())
                {
                    chunkThread.join();
                }
                chunkThreadGuard.dismiss();
                if (chunkException)
                {
                    std::rethrow_exception(chunkException);
                }
            }

            if (streamOnly)
            {
                auto const doneAt = std::chrono::steady_clock::now();
                int32_t const streamSampleRate = code2wavRunner ? code2wavRunner->getConfig().sampleRate : 24000;
                double const audioSeconds = static_cast<double>(streamedSamples) / streamSampleRate;
                double const totalMs = std::chrono::duration<double, std::milli>(doneAt - requestStart).count();
                if (lazyCode2Wav)
                {
                    code2wavRunner.reset();
                }
                response = Json{{"id", id},
                    {"event", "done"},
                    {"ok", true},
                    {"stream_only", true},
                    {"frames", talkerResponse.numFrames},
                    {"samples", streamedSamples},
                    {"sample_rate", streamSampleRate},
                    {"audio_s", audioSeconds},
                    {"generation_ms", std::chrono::duration<double, std::milli>(genEnd - genStart).count()},
                    {"code2wav_ms", streamedCode2WavMs},
                    {"first_chunk_ms",
                        (firstChunkAt.time_since_epoch().count() != 0)
                            ? std::chrono::duration<double, std::milli>(firstChunkAt - requestStart).count()
                            : 0.0},
                    {"total_ms", totalMs},
                    {"rtf", audioSeconds > 0.0 ? totalMs / 1000.0 / audioSeconds : 0.0}};
                std::cout << response.dump() << std::endl;
                continue;
            }

            rt::audioUtils::AudioData audioOutput;
            auto transposed = transposeCodes(talkerResponse.rvqCodes);
            auto const wavStart = std::chrono::steady_clock::now();
            if (!code2wavRunner)
            {
                code2wavRunner = std::make_unique<Code2WavRunner>(args.code2wavEngineDir, stream);
            }
            if (!code2wavRunner->generateWaveform(transposed, audioOutput, stream))
            {
                throw std::runtime_error("Code2Wav failed");
            }
            auto const wavEnd = std::chrono::steady_clock::now();
            if (lazyCode2Wav)
            {
                code2wavRunner.reset();
            }
            if (!saveAudioToWav(outputFile, audioOutput))
            {
                throw std::runtime_error("Failed to save WAV: " + outputFile);
            }

            int64_t samples = (audioOutput.waveform && !audioOutput.waveform->isEmpty())
                ? audioOutput.waveform->getShape()[1]
                : 0;
            double const audioSeconds = static_cast<double>(samples) / audioOutput.sampleRate;
            double const totalMs
                = std::chrono::duration<double, std::milli>(wavEnd - requestStart).count();
            response = Json{{"id", id},
                {"event", "done"},
                {"ok", true},
                {"output_file", outputFile},
                {"frames", talkerResponse.numFrames},
                {"samples", samples},
                {"sample_rate", audioOutput.sampleRate},
                {"audio_s", audioSeconds},
                {"generation_ms", std::chrono::duration<double, std::milli>(genEnd - genStart).count()},
                {"code2wav_ms", std::chrono::duration<double, std::milli>(wavEnd - wavStart).count()},
                {"first_chunk_ms",
                    (streamOutput && firstChunkAt.time_since_epoch().count() != 0)
                        ? std::chrono::duration<double, std::milli>(firstChunkAt - requestStart).count()
                        : 0.0},
                {"total_ms", totalMs},
                {"rtf", audioSeconds > 0.0 ? totalMs / 1000.0 / audioSeconds : 0.0}};
        }
        catch (std::exception const& e)
        {
            response = Json{{"event", "error"}, {"ok", false}, {"error", e.what()}};
        }
        std::cout << response.dump() << std::endl;
    }

    CUDA_CHECK(cudaStreamDestroy(code2wavStream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
