// Tests for dune_sha256.h
//
// Verifies SHA-256 computation against standard test vectors.

#include "dune_sha256.h"
#include <iostream>
#include <fstream>
#include <cstdlib>

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

  std::cout << "\nAll " << pass_count << "/" << test_count << " tests PASSED\n";
  return 0;
}
