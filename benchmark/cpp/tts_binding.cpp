// tts_binding.cpp — pybind11 binding for Qwen3-TTS C++ TRT engine
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cstring>
#include <vector>

#include "tts_pipeline.h"

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
  m.doc() = "Qwen3-TTS C++ TRT native inference engine";

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
           py::arg("mel"));
}
