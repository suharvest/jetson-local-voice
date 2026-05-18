// json_minimal.h — Minimal JSON parser for config.json (no external deps)
// Only supports flat objects with string/number values and one level of nesting.
#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

class JsonMinimal {
 public:
  static JsonMinimal Parse(const std::string& path) {
    JsonMinimal j;
    std::ifstream f(path);
    if (!f.is_open()) {
      std::cerr << "Failed to open: " << path << std::endl;
      return j;
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    j.ParseString(content);
    return j;
  }

  int GetInt(const std::string& key, int def = 0) const {
    auto it = values_.find(key);
    if (it == values_.end()) return def;
    try {
      return std::stoi(it->second);
    } catch (...) {
      return def;
    }
  }

  std::string GetStr(const std::string& key,
                     const std::string& def = "") const {
    auto it = values_.find(key);
    return (it != values_.end()) ? it->second : def;
  }

  // Get nested object as key-value pairs
  std::vector<std::pair<std::string, std::string>> GetObject(
      const std::string& key) const {
    auto it = objects_.find(key);
    if (it == objects_.end()) return {};
    return it->second;
  }

 private:
  void ParseString(const std::string& s) {
    size_t i = 0;
    SkipWs(s, i);
    if (i < s.size() && s[i] == '{') ++i;
    ParseObject(s, i, values_, objects_);
  }

  static void SkipWs(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' ||
                            s[i] == '\t'))
      ++i;
  }

  static std::string ReadString(const std::string& s, size_t& i) {
    if (i >= s.size() || s[i] != '"') return "";
    ++i;
    std::string result;
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\' && i + 1 < s.size()) {
        ++i;
      }
      result += s[i++];
    }
    if (i < s.size()) ++i;  // skip closing "
    return result;
  }

  static std::string ReadValue(const std::string& s, size_t& i) {
    SkipWs(s, i);
    if (i < s.size() && s[i] == '"') return ReadString(s, i);
    // Number or literal
    std::string result;
    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' &&
           s[i] != ' ' && s[i] != '\n') {
      result += s[i++];
    }
    return result;
  }

  static void ParseObject(
      const std::string& s, size_t& i,
      std::map<std::string, std::string>& flat,
      std::map<std::string, std::vector<std::pair<std::string, std::string>>>&
          nested) {
    while (i < s.size()) {
      SkipWs(s, i);
      if (i >= s.size() || s[i] == '}') {
        ++i;
        return;
      }
      if (s[i] == ',') {
        ++i;
        continue;
      }

      std::string key = ReadString(s, i);
      SkipWs(s, i);
      if (i < s.size() && s[i] == ':') ++i;
      SkipWs(s, i);

      if (i < s.size() && s[i] == '{') {
        // Nested object
        ++i;
        std::vector<std::pair<std::string, std::string>> pairs;
        while (i < s.size()) {
          SkipWs(s, i);
          if (i >= s.size() || s[i] == '}') {
            ++i;
            break;
          }
          if (s[i] == ',') {
            ++i;
            continue;
          }
          std::string nk = ReadString(s, i);
          SkipWs(s, i);
          if (i < s.size() && s[i] == ':') ++i;
          std::string nv = ReadValue(s, i);
          pairs.push_back({nk, nv});
        }
        nested[key] = pairs;
      } else if (i < s.size() && s[i] == '[') {
        // Skip arrays for now
        int depth = 1;
        ++i;
        while (i < s.size() && depth > 0) {
          if (s[i] == '[') ++depth;
          if (s[i] == ']') --depth;
          ++i;
        }
      } else {
        flat[key] = ReadValue(s, i);
      }
    }
  }

  std::map<std::string, std::string> values_;
  std::map<std::string, std::vector<std::pair<std::string, std::string>>>
      objects_;
};
