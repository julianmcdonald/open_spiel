// Tests for dune_sha256.h
//
// Verifies OpenSSL EVP SHA-256 computation against standard NIST test vectors,
// tests incremental chunk fragmentation, validates copy/move semantics,
// and verifies exact equivalence against the legacy ScalarSHA256 reference.

#include "dune_sha256.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace open_spiel;

static int test_count = 0;
static int pass_count = 0;

#define TEST_BEGIN(name)                                              \
  do {                                                                \
    ++test_count;                                                     \
    const char* test_name_ = (name);                                  \
    std::cout << "Test " << test_count << ": " << test_name_ << "... ";

#define TEST_END()                                                    \
    ++pass_count;                                                     \
    std::cout << "PASSED\n";                                          \
  } while (0)

#define CHECK_EQ(a, b)                                                \
  do {                                                                \
    auto a_ = (a); auto b_ = (b);                                    \
    if (a_ != b_) {                                                   \
      std::cerr << "FAILED\n  Expected " #a " == " #b               \
                << "\n  Got " << a_ << " vs " << b_                  \
                << "\n  at " << __FILE__ << ":" << __LINE__ << "\n";  \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

int main() {
  std::cout << "=== dune_sha256_test ===\n\n";

  // Test 1: Empty string
  TEST_BEGIN("Empty string hash") {
    std::string hash = ComputeStringSHA256("");
    CHECK_EQ(hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  } TEST_END();

  // Test 2: "abc"
  TEST_BEGIN("abc hash") {
    std::string hash = ComputeStringSHA256("abc");
    CHECK_EQ(hash, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  } TEST_END();

  // Test 3: Long string test vector
  TEST_BEGIN("Long string hash") {
    std::string hash = ComputeStringSHA256(
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
    CHECK_EQ(hash, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  } TEST_END();

  // Test 4: File hashing
  TEST_BEGIN("File hashing") {
    std::string filename = "dune_sha256_temp_test.txt";
    {
      std::ofstream ofs(filename, std::ios::binary);
      ofs << "abc";
    }
    size_t size = 0;
    std::string hash = ComputeFileSHA256(filename, &size);
    CHECK_EQ(size, 3);
    CHECK_EQ(hash, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    std::remove(filename.c_str());
  } TEST_END();

  // Test 5: Fragmented incremental input
  TEST_BEGIN("Fragmented incremental input (variable chunk sizes)") {
    std::string text = "The quick brown fox jumps over the lazy dog. 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::string expected = ComputeStringSHA256(text);

    // Feed 1 byte at a time
    {
      SHA256 h;
      for (char c : text) {
        h.Update(&c, 1);
      }
      CHECK_EQ(h.Final(), expected);
    }

    // Feed in chunks of 3 bytes
    {
      SHA256 h;
      size_t pos = 0;
      while (pos < text.size()) {
        size_t len = std::min<size_t>(3, text.size() - pos);
        h.Update(text.data() + pos, len);
        pos += len;
      }
      CHECK_EQ(h.Final(), expected);
    }

    // Feed in chunks of 7 bytes
    {
      SHA256 h;
      size_t pos = 0;
      while (pos < text.size()) {
        size_t len = std::min<size_t>(7, text.size() - pos);
        h.Update(text.data() + pos, len);
        pos += len;
      }
      CHECK_EQ(h.Final(), expected);
    }
  } TEST_END();

  // Test 6: Equivalence between OpenSSL SHA256 and ScalarSHA256 reference
  TEST_BEGIN("Equivalence between OpenSSL SHA256 and ScalarSHA256 across boundary lengths") {
    std::vector<size_t> lengths = {
      0, 1, 2, 3, 54, 55, 56, 57, 63, 64, 65,
      119, 120, 127, 128, 129, 255, 256, 257,
      1000, 4096, 65536, 131072
    };

    std::mt19937 rng(42);
    for (size_t len : lengths) {
      std::vector<uint8_t> data(len);
      for (size_t i = 0; i < len; ++i) {
        data[i] = static_cast<uint8_t>(rng() & 0xFF);
      }

      SHA256 evp_hasher;
      evp_hasher.Update(data.data(), data.size());
      std::string evp_hash = evp_hasher.Final();

      ScalarSHA256 scalar_hasher;
      scalar_hasher.Update(data.data(), data.size());
      std::string scalar_hash = scalar_hasher.Final();

      CHECK_EQ(evp_hash, scalar_hash);
    }
  } TEST_END();

  // Test 7: Copy/Move semantics and Final() reset behavior
  TEST_BEGIN("Copy/Move semantics and Final() reset contract") {
    // Final() resets the context so it can be reused immediately
    SHA256 h1;
    h1.Update("abc");
    std::string hash_abc = h1.Final();
    CHECK_EQ(hash_abc, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    // Reusing h1 immediately after Final() should hash from clean initial state
    h1.Update("");
    CHECK_EQ(h1.Final(), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    // Copy constructor test
    SHA256 h2;
    h2.Update("abc");
    SHA256 h2_copy(h2);
    // Both continue independently
    h2.Update("def");
    h2_copy.Update("def");
    CHECK_EQ(h2.Final(), h2_copy.Final());

    // Move constructor test
    SHA256 h3;
    h3.Update("abcdef");
    SHA256 h3_moved(std::move(h3));
    CHECK_EQ(h3_moved.Final(), ComputeStringSHA256("abcdef"));

    // Copy assignment test
    SHA256 h4;
    h4.Update("hello");
    SHA256 h5;
    h5 = h4;
    CHECK_EQ(h4.Final(), h5.Final());

    // Move assignment test
    SHA256 h6;
    h6.Update("world");
    SHA256 h7;
    h7 = std::move(h6);
    CHECK_EQ(h7.Final(), ComputeStringSHA256("world"));
  } TEST_END();

  std::cout << "\nAll " << pass_count << "/" << test_count << " tests PASSED\n";
  return 0;
}
