// tts_binding.cpp — pybind11 binding for Qwen3-TTS/ASR C++ TRT engine
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cstring>
#include <vector>

#include "tts_pipeline.h"
#include "asr_pipeline.h"

namespace py = pybind11;

// WAV writer (returns bytes)
static py::bytes MakeWav(const std::vector<float>& audio, int sample_rate) {
  int n = (int)audio.size();
  int data_size = n * 2;
  int file_size = 36 + data_size;

  std::vector<char> buf(44 + data_size);
  char* p = buf.data();

  std::memcpy(p, "RIFF", 4); p += 4;
  std::memcpy(p, &file_size, 4); p += 4;
  std::memcpy(p, "WAVE", 4); p += 4;
  std::memcpy(p, "fmt ", 4); p += 4;
  int fmt_size = 16;
  std::memcpy(p, &fmt_size, 4); p += 4;
  short audio_fmt = 1, channels = 1, bits = 16;
  std::memcpy(p, &audio_fmt, 2); p += 2;
  std::memcpy(p, &channels, 2); p += 2;
  std::memcpy(p, &sample_rate, 4); p += 4;
  int byte_rate = sample_rate * 2;
  std::memcpy(p, &byte_rate, 4); p += 4;
  short block_align = 2;
  std::memcpy(p, &block_align, 2); p += 2;
  std::memcpy(p, &bits, 2); p += 2;
  std::memcpy(p, "data", 4); p += 4;
  std::memcpy(p, &data_size, 4); p += 4;

  for (int i = 0; i < n; ++i) {
    float s = audio[i] * 32767.0f;
    if (s > 32767.0f) s = 32767.0f;
    if (s < -32768.0f) s = -32768.0f;
    short v = (short)s;
    std::memcpy(p, &v, 2); p += 2;
  }

  return py::bytes(buf.data(), buf.size());
}

PYBIND11_MODULE(qwen3_tts_engine, m) {
  m.doc() = "Qwen3-TTS/ASR C++ TRT native inference engine";

  // ── ASR Decoder (reuses TRTTalkerEngine with different vocab_size) ──
  py::class_<TRTTalkerEngine>(m, "ASRDecoder")
      .def(py::init<const std::string&, int, int, int, int, int, int>(),
           py::arg("engine_path"),
           py::arg("n_layers") = 28,
           py::arg("hidden_dim") = 1024,
           py::arg("n_heads") = 8,
           py::arg("head_dim") = 128,
           py::arg("vocab_size") = 151936,
           py::arg("max_seq") = 500)

      .def("seed_kv",
           [](TRTTalkerEngine& self, py::dict kv_dict, int seq_len) {
             // Collect KV pointers from dict of numpy arrays
             std::vector<const float*> ptrs;
             for (int i = 0; i < 28; ++i) {
               for (auto prefix : {"past_key_", "past_value_"}) {
                 std::string name = std::string(prefix) + std::to_string(i);
                 if (kv_dict.contains(name)) {
                   auto arr = kv_dict[name.c_str()].cast<py::array_t<float>>();
                   ptrs.push_back(static_cast<const float*>(arr.request().ptr));
                 }
               }
             }
             self.SeedKV(ptrs.data(), (int)ptrs.size(), seq_len);
           },
           py::arg("kv_dict"), py::arg("seq_len"))

      .def("decode_step",
           [](TRTTalkerEngine& self, py::array_t<float> embeds,
              int vocab_size) -> py::array_t<float> {
             auto buf = embeds.request();
             std::vector<float> logits(vocab_size);
             std::vector<float> hidden(1024);
             self.DecodeStep(static_cast<const float*>(buf.ptr),
                             logits.data(), hidden.data());
             auto result = py::array_t<float>({1, 1, vocab_size});
             std::memcpy(result.mutable_data(), logits.data(),
                         vocab_size * sizeof(float));
             return result;
           },
           py::arg("input_embeds"),
           py::arg("vocab_size") = 151936)

      .def("reset", &TRTTalkerEngine::Reset);


  py::class_<SynthResult>(m, "SynthResult")
      .def_readonly("n_frames", &SynthResult::n_frames)
      .def_readonly("prefill_ms", &SynthResult::prefill_ms)
      .def_readonly("decode_ms_avg", &SynthResult::decode_ms_avg)
      .def_readonly("cp_ms_avg", &SynthResult::cp_ms_avg)
      .def_readonly("vocoder_ms", &SynthResult::vocoder_ms)
      .def_readonly("total_ms", &SynthResult::total_ms)
      .def_readonly("rtf", &SynthResult::rtf)
      .def_readonly("sample_rate", &SynthResult::sample_rate)
      .def_readonly("audio", &SynthResult::audio);

  py::class_<TTSPipeline>(m, "Pipeline")
      .def(py::init<const std::string&, const std::string&,
                     const std::string&, const std::string&, int>(),
           py::arg("model_dir"),
           py::arg("sherpa_dir"),
           py::arg("talker_engine"),
           py::arg("cp_engine"),
           py::arg("device_id") = 0)

      .def("synthesize",
           [](TTSPipeline& self, const std::string& text,
              const std::string& lang,
              const std::vector<int64_t>& token_ids,
              int max_frames, int seed) -> py::dict {
             SynthResult r;
             if (!token_ids.empty()) {
               r = self.SynthesizeWithTokenIds(text, lang, token_ids,
                                                nullptr, max_frames, seed);
             } else {
               r = self.Synthesize(text, lang, max_frames, seed);
             }

             py::dict result;
             result["wav_bytes"] = MakeWav(r.audio, r.sample_rate);
             result["n_frames"] = r.n_frames;
             result["duration"] = r.n_frames / 12.5;
             result["prefill_ms"] = r.prefill_ms;
             result["per_step_ms"] = r.decode_ms_avg + r.cp_ms_avg;
             result["rtf"] = r.rtf;
             result["total_ms"] = r.total_ms;
             return result;
           },
           py::arg("text") = "",
           py::arg("lang") = "english",
           py::arg("token_ids") = std::vector<int64_t>{},
           py::arg("max_frames") = 200,
           py::arg("seed") = 42)

      .def("synthesize_clone",
           [](TTSPipeline& self, const std::string& text,
              const std::string& lang,
              const std::vector<int64_t>& token_ids,
              py::bytes speaker_emb_bytes,
              int max_frames, int seed) -> py::dict {
             std::string emb_str = speaker_emb_bytes;
             std::vector<float> spk(emb_str.size() / sizeof(float));
             std::memcpy(spk.data(), emb_str.data(), emb_str.size());

             auto r = self.SynthesizeWithTokenIds(text, lang, token_ids,
                                                   &spk, max_frames, seed);
             py::dict result;
             result["wav_bytes"] = MakeWav(r.audio, r.sample_rate);
             result["n_frames"] = r.n_frames;
             result["duration"] = r.n_frames / 12.5;
             result["rtf"] = r.rtf;
             result["total_ms"] = r.total_ms;
             return result;
           },
           py::arg("text"),
           py::arg("lang"),
           py::arg("token_ids"),
           py::arg("speaker_emb_bytes"),
           py::arg("max_frames") = 200,
           py::arg("seed") = 42)

      .def("extract_speaker_embedding",
           [](TTSPipeline& self, py::array_t<float> mel) -> py::bytes {
             auto buf = mel.request();
             int mel_frames = buf.shape[0];
             auto emb = self.ExtractSpeakerEmbedding(
                 static_cast<float*>(buf.ptr), mel_frames);
             return py::bytes(reinterpret_cast<char*>(emb.data()),
                              emb.size() * sizeof(float));
           },
           py::arg("mel"))

      .def("synthesize_streaming",
           [](TTSPipeline& self, const std::string& text,
              const std::string& lang,
              const std::vector<int64_t>& token_ids,
              int first_chunk_frames, int chunk_frames,
              int max_frames, int seed) -> py::list {
             StreamConfig config;
             config.first_chunk_frames = first_chunk_frames;
             config.chunk_frames = chunk_frames;
             config.max_frames = max_frames;
             config.seed = seed;

             py::list chunks;

             // Release GIL during C++ generation, reacquire for callback
             {
               py::gil_scoped_release release;

               std::vector<StreamChunk> collected;
               self.SynthesizeStreaming(text, lang, token_ids, config,
                   [&collected](const StreamChunk& chunk) {
                     collected.push_back(chunk);
                   });

               py::gil_scoped_acquire acquire;
               for (auto& c : collected) {
                 py::dict d;
                 d["wav_bytes"] = MakeWav(c.audio, 24000);
                 d["pcm_samples"] = (int)c.audio.size();
                 d["total_frames"] = c.total_frames;
                 d["is_final"] = c.is_final;
                 chunks.append(d);
               }
             }

             return chunks;
           },
           py::arg("text") = "",
           py::arg("lang") = "english",
           py::arg("token_ids") = std::vector<int64_t>{},
           py::arg("first_chunk_frames") = 10,
           py::arg("chunk_frames") = 25,
           py::arg("max_frames") = 200,
           py::arg("seed") = 42)

      .def("enable_profiling", &TTSPipeline::EnableProfiling,
           py::arg("enable") = true,
           "Enable CUDA event profiling for per-step timing breakdown")
      .def("print_profiling_stats", &TTSPipeline::PrintProfilingStats,
           "Print and reset CUDA event profiling stats")

      .def("synthesize_streaming_callback",
           [](TTSPipeline& self, const std::string& text,
              const std::string& lang,
              const std::vector<int64_t>& token_ids,
              py::object callback,
              int first_chunk_frames, int chunk_frames,
              int max_frames, int seed) {
             StreamConfig config;
             config.first_chunk_frames = first_chunk_frames;
             config.chunk_frames = chunk_frames;
             config.max_frames = max_frames;
             config.seed = seed;

             // Release GIL for C++ work, reacquire in callback for Python
             py::gil_scoped_release release;
             self.SynthesizeStreaming(text, lang, token_ids, config,
                 [&callback](const StreamChunk& chunk) {
                   py::gil_scoped_acquire acquire;
                   py::dict d;
                   d["wav_bytes"] = MakeWav(chunk.audio, 24000);
                   d["pcm_samples"] = (int)chunk.audio.size();
                   d["total_frames"] = chunk.total_frames;
                   d["is_final"] = chunk.is_final;
                   callback(d);
                 });
           },
           py::arg("text"),
           py::arg("lang"),
           py::arg("token_ids"),
           py::arg("callback"),
           py::arg("first_chunk_frames") = 10,
           py::arg("chunk_frames") = 25,
           py::arg("max_frames") = 200,
           py::arg("seed") = 42);

  // ── ASR Pipeline (encoder + prefill + TRT decode, all in C++) ──
  py::class_<ASRPipeline>(m, "ASRPipeline")
      .def(py::init<const std::string&, const std::string&, int>(),
           py::arg("model_dir"),
           py::arg("engine_path"),
           py::arg("device_id") = 0)

      .def("transcribe",
           [](ASRPipeline& self, py::array_t<float> mel,
              const std::vector<int64_t>& prompt_ids, int audio_offset,
              int max_tokens) -> py::dict {
             auto buf = mel.request();
             // Accept [128, T] or [1, 128, T]
             int mel_len;
             const float* mel_ptr;
             if (buf.ndim == 2) {
               mel_len = (int)buf.shape[1];
               mel_ptr = static_cast<const float*>(buf.ptr);
             } else if (buf.ndim == 3) {
               mel_len = (int)buf.shape[2];
               mel_ptr = static_cast<const float*>(buf.ptr);
             } else {
               throw std::runtime_error(
                   "mel must be [128, T] or [1, 128, T]");
             }

             auto r = self.Transcribe(mel_ptr, mel_len, prompt_ids,
                                       audio_offset, max_tokens);

             py::dict result;
             result["text_ids"] = r.text_ids;
             result["n_tokens"] = r.n_tokens;
             result["encoder_ms"] = r.encoder_ms;
             result["prefill_ms"] = r.prefill_ms;
             result["decode_ms"] = r.decode_ms;
             result["per_token_ms"] = r.per_token_ms;
             result["total_ms"] = r.total_ms;
             return result;
           },
           py::arg("mel"),
           py::arg("prompt_ids"),
           py::arg("audio_offset"),
           py::arg("max_tokens") = 200)

      .def("run_encoder",
           [](ASRPipeline& self, py::array_t<float> mel) -> int {
             auto buf = mel.request();
             int mel_len;
             const float* mel_ptr;
             if (buf.ndim == 2) {
               mel_len = (int)buf.shape[1];
               mel_ptr = static_cast<const float*>(buf.ptr);
             } else if (buf.ndim == 3) {
               mel_len = (int)buf.shape[2];
               mel_ptr = static_cast<const float*>(buf.ptr);
             } else {
               throw std::runtime_error(
                   "mel must be [128, T] or [1, 128, T]");
             }
             auto enc = self.RunEncoder(mel_ptr, mel_len);
             return enc.audio_len;
           },
           py::arg("mel"),
           "Run encoder only and return audio feature length (T').")

      .def("enable_profiling", &ASRPipeline::EnableProfiling,
           py::arg("enable") = true,
           "Enable CUDA event profiling for per-step timing breakdown")
      .def("print_profiling_stats", &ASRPipeline::PrintProfilingStats,
           "Print and reset CUDA event profiling stats")

      .def_property_readonly("has_prefill", &ASRPipeline::has_prefill,
           "True if ORT prefill session is loaded (needed for correct results)")
      .def_property_readonly("hidden_dim", &ASRPipeline::hidden_dim)
      .def_property_readonly("vocab_size", &ASRPipeline::vocab_size)
      .def_property_readonly("n_layers", &ASRPipeline::n_layers);
}
