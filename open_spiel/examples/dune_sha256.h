#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SHA256_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SHA256_H_

#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <stdexcept>

namespace open_spiel {

class SHA256 {
 public:
  SHA256() { Reset(); }

  void Reset() {
    state_[0] = 0x6a09e667;
    state_[1] = 0xbb67ae85;
    state_[2] = 0x3c6ef372;
    state_[3] = 0xa54ff53a;
    state_[4] = 0x510e527f;
    state_[5] = 0x9b05688c;
    state_[6] = 0x1f83d9ab;
    state_[7] = 0x5be0cd19;
    bit_count_ = 0;
    buffer_len_ = 0;
  }

  void Update(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      buffer_[buffer_len_++] = data[i];
      bit_count_ += 8;
      if (buffer_len_ == 64) {
        Transform(state_, buffer_);
        buffer_len_ = 0;
      }
    }
  }

  void Update(const std::string& str) {
    Update(reinterpret_cast<const uint8_t*>(str.data()), str.size());
  }

  std::string Final() {
    uint64_t bits = bit_count_;
    // Pad with 0x80
    uint8_t pad = 0x80;
    Update(&pad, 1);
    // Pad with 0x00 until we have 8 bytes remaining (56 bytes processed in block)
    while (buffer_len_ != 56) {
      uint8_t zero = 0x00;
      Update(&zero, 1);
    }
    // Append bit count in big-endian
    uint8_t bits_be[8];
    for (int i = 0; i < 8; ++i) {
      bits_be[i] = static_cast<uint8_t>(bits >> (56 - i * 8));
    }
    Update(bits_be, 8);

    // Format hex string
    std::ostringstream oss;
    for (int i = 0; i < 8; ++i) {
      oss << std::hex << std::setw(8) << std::setfill('0') << state_[i];
    }
    Reset();
    return oss.str();
  }

 private:
  static inline uint32_t ROTR(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
  }
  static inline uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
  }
  static inline uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
  }
  static inline uint32_t Sigma0(uint32_t x) {
    return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22);
  }
  static inline uint32_t Sigma1(uint32_t x) {
    return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25);
  }
  static inline uint32_t sigma0(uint32_t x) {
    return ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3);
  }
  static inline uint32_t sigma1(uint32_t x) {
    return ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10);
  }

  void Transform(uint32_t state[8], const uint8_t buffer[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint32_t>(buffer[i * 4]) << 24) |
             (static_cast<uint32_t>(buffer[i * 4 + 1]) << 16) |
             (static_cast<uint32_t>(buffer[i * 4 + 2]) << 8) |
             (static_cast<uint32_t>(buffer[i * 4 + 3]));
    }
    for (int i = 16; i < 64; ++i) {
      w[i] = sigma1(w[i - 2]) + w[i - 7] + sigma0(w[i - 15]) + w[i - 16];
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    for (int i = 0; i < 64; ++i) {
      uint32_t t1 = h + Sigma1(e) + Ch(e, f, g) + K[i] + w[i];
      uint32_t t2 = Sigma0(a) + Maj(a, b, c);
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  uint32_t state_[8];
  uint64_t bit_count_;
  uint8_t buffer_[64];
  size_t buffer_len_;
};

inline std::string ComputeFileSHA256(const std::string& filepath, size_t* file_size_out = nullptr) {
  std::ifstream ifs(filepath, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Could not open file to compute SHA-256: " + filepath);
  }
  SHA256 hasher;
  char chunk[65536];
  size_t total_size = 0;
  while (ifs.read(chunk, sizeof(chunk))) {
    size_t count = ifs.gcount();
    hasher.Update(reinterpret_cast<const uint8_t*>(chunk), count);
    total_size += count;
  }
  size_t remainder = ifs.gcount();
  if (remainder > 0) {
    hasher.Update(reinterpret_cast<const uint8_t*>(chunk), remainder);
    total_size += remainder;
  }
  if (file_size_out) {
    *file_size_out = total_size;
  }
  return hasher.Final();
}

inline std::string ComputeStringSHA256(const std::string& input) {
  SHA256 hasher;
  hasher.Update(input);
  return hasher.Final();
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_SHA256_H_
