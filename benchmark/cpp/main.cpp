// main.cpp — Qwen3-TTS C++ TRT inference CLI
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "tts_pipeline.h"

static void WriteWav(const std::string& path, const float* audio, int n_samples,
                     int sample_rate = 24000) {
  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "Cannot write: " << path << std::endl;
    return;
  }

  // Convert float32 to int16
  std::vector<int16_t> pcm(n_samples);
  for (int i = 0; i < n_samples; ++i) {
    float s = audio[i] * 32767.0f;
    if (s > 32767.0f) s = 32767.0f;
    if (s < -32768.0f) s = -32768.0f;
    pcm[i] = (int16_t)s;
  }

  // WAV header
  int data_size = n_samples * 2;
  int file_size = 44 + data_size - 8;
  int16_t channels = 1;
  int16_t bits_per_sample = 16;
  int byte_rate = sample_rate * channels * bits_per_sample / 8;
  int16_t block_align = channels * bits_per_sample / 8;

  f.write("RIFF", 4);
  f.write((char*)&file_size, 4);
  f.write("WAVE", 4);
  f.write("fmt ", 4);
  int fmt_size = 16;
  f.write((char*)&fmt_size, 4);
  int16_t audio_fmt = 1;  // PCM
  f.write((char*)&audio_fmt, 2);
  f.write((char*)&channels, 2);
  f.write((char*)&sample_rate, 4);
  f.write((char*)&byte_rate, 4);
  f.write((char*)&block_align, 2);
  f.write((char*)&bits_per_sample, 2);
  f.write("data", 4);
  f.write((char*)&data_size, 4);
  f.write((char*)pcm.data(), data_size);

  std::cout << "Saved: " << path << " (" << n_samples / sample_rate << "s)"
            << std::endl;
}

static void PrintUsage(const char* prog) {
  std::cerr << "Usage: " << prog << " [options]\n"
            << "  --model-dir DIR       Model directory (HF)\n"
            << "  --sherpa-dir DIR      Sherpa ONNX models\n"
            << "  --talker-engine PATH  Talker FP16 TRT engine\n"
            << "  --cp-engine PATH      Code predictor BF16 TRT engine\n"
            << "  --text TEXT           Text to synthesize\n"
            << "  --lang LANG           Language (english/chinese)\n"
            << "  --output PATH         Output WAV path\n"
            << "  --max-frames N        Max frames (default: 200)\n"
            << "  --seed N              Random seed (default: 42)\n"
            << "  --speaker-emb PATH    Speaker embedding .bin (voice clone)\n"
            << "  --token-ids IDS       Comma-separated token IDs (bypass tokenizer)\n"
            << "  --profile             Enable CUDA event profiling (H2D/kernel/D2H breakdown)\n"
            << std::endl;
}

int main(int argc, char* argv[]) {
  std::string model_dir = "/tmp/qwen3-tts-bench/model";
  std::string sherpa_dir = "/tmp/qwen3-sherpa";
  std::string talker_engine = "/tmp/talker_decode_fp16.engine";
  std::string cp_engine = "/tmp/cp_bf16.engine";
  std::string text = "Hello, welcome to the voice synthesis system.";
  std::string lang = "english";
  std::string output = "/tmp/tts_cpp.wav";
  std::string speaker_emb_path;
  std::string token_ids_str;
  int max_frames = 200;
  int seed = 42;
  bool profile = false;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--model-dir") && i + 1 < argc)
      model_dir = argv[++i];
    else if (!strcmp(argv[i], "--sherpa-dir") && i + 1 < argc)
      sherpa_dir = argv[++i];
    else if (!strcmp(argv[i], "--talker-engine") && i + 1 < argc)
      talker_engine = argv[++i];
    else if (!strcmp(argv[i], "--cp-engine") && i + 1 < argc)
      cp_engine = argv[++i];
    else if (!strcmp(argv[i], "--text") && i + 1 < argc)
      text = argv[++i];
    else if (!strcmp(argv[i], "--lang") && i + 1 < argc)
      lang = argv[++i];
    else if (!strcmp(argv[i], "--output") && i + 1 < argc)
      output = argv[++i];
    else if (!strcmp(argv[i], "--max-frames") && i + 1 < argc)
      max_frames = std::atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
      seed = std::atoi(argv[++i]);
    else if (!strcmp(argv[i], "--speaker-emb") && i + 1 < argc)
      speaker_emb_path = argv[++i];
    else if (!strcmp(argv[i], "--token-ids") && i + 1 < argc)
      token_ids_str = argv[++i];
    else if (!strcmp(argv[i], "--profile"))
      profile = true;
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      PrintUsage(argv[0]);
      return 0;
    }
  }

  // Parse token IDs if provided
  std::vector<int64_t> token_ids;
  if (!token_ids_str.empty()) {
    std::istringstream iss(token_ids_str);
    std::string tok;
    while (std::getline(iss, tok, ',')) {
      token_ids.push_back(std::stoll(tok));
    }
    std::cout << "Using " << token_ids.size() << " pre-tokenized IDs" << std::endl;
  }

  // Build pipeline
  TTSPipeline pipeline(model_dir, sherpa_dir, talker_engine, cp_engine);

  if (profile) {
    pipeline.EnableProfiling(true);
    std::cout << "CUDA event profiling enabled" << std::endl;
  }

  SynthResult result;
  if (!token_ids.empty()) {
    // Use pre-tokenized IDs
    if (!speaker_emb_path.empty()) {
      std::ifstream sf(speaker_emb_path, std::ios::binary | std::ios::ate);
      size_t sz = sf.tellg();
      sf.seekg(0);
      std::vector<float> spk(sz / sizeof(float));
      sf.read((char*)spk.data(), sz);
      result = pipeline.SynthesizeWithTokenIds(text, lang, token_ids, &spk, max_frames, seed);
    } else {
      result = pipeline.SynthesizeWithTokenIds(text, lang, token_ids, nullptr, max_frames, seed);
    }
  } else if (!speaker_emb_path.empty()) {
    // Load speaker embedding from binary file
    std::ifstream sf(speaker_emb_path, std::ios::binary | std::ios::ate);
    if (!sf.is_open()) {
      std::cerr << "Cannot open speaker embedding: " << speaker_emb_path
                << std::endl;
      return 1;
    }
    size_t sz = sf.tellg();
    sf.seekg(0);
    std::vector<float> spk(sz / sizeof(float));
    sf.read((char*)spk.data(), sz);
    result = pipeline.SynthesizeWithSpeaker(text, lang, spk, max_frames, seed);
  } else {
    result = pipeline.Synthesize(text, lang, max_frames, seed);
  }

  // Print profiling breakdown
  if (profile) {
    pipeline.PrintProfilingStats();
  }

  // Write WAV
  if (!result.audio.empty()) {
    WriteWav(output, result.audio.data(), (int)result.audio.size());
  }

  return 0;
}
