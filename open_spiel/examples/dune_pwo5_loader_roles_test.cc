// PWO-5 gate 3: the auxiliary-label loader's manifest FILE ROLES.
//
// docs/PWO5_AMENDMENT_1_TARGET_EXPOSURE_TELEMETRY_2026_07_31.md section 6
// (RULING 5), against docs/PWO5_PILOT_REGISTRATION.md sections 6.1, 6.2, 8.5.
//
// ---------------------------------------------------------------------------
// WHAT RULING 5 REGISTERS, AND WHAT THIS FILE PROVES
// ---------------------------------------------------------------------------
//
// The legacy loader's built-in split is IsValidationLabel -- a PER-ROW FNV-1a
// hash mod 11 applied to every row of every file. On the PWO-5 pack it diverts
// 1,890 of 20,582 rows (9.18%) into a bucket the distillation loss never trains
// on, so a registered exposure of 20,582 rows and 14.93 expected passes
// silently became 18,692 and 16.43.
//
// Ruling 5 registers four rules, and each has a test below:
//
//   1. a file carrying an explicit `role` assigns EVERY one of its rows to that
//      role, and IsValidationLabel is not consulted for it;
//   2. a file carrying NO role keeps the legacy %11 behaviour, so no committed
//      result changes and no other consumer of the loader is affected;
//   3. the realized per-role counts MUST equal the manifest's declared counts,
//      asserted fatally after load -- the previous code checked the declared
//      counts against the FLOORS and never against reality;
//   4. NO HELD-OUT ROW MAY RECEIVE A TRAINING GRADIENT.
//
// Rule 4 is the load-bearing one and it is proved two ways: exhaustively (every
// row in the training vector carries the train pack's signature) and
// behaviourally (a large draw from the sampler never returns a validation row).
//
// The tests link the SHIPPED SearchLabelBuffer out of
// dune_search_label_buffer.h. A test that restated the loader would prove
// nothing about the loader the trainer links -- that is exactly how the
// Memocorders defect came to have three drifting copies.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"

#include "open_spiel/abseil-cpp/absl/flags/flag.h"

// dune_ppo_training_utils.cc ABSL_DECLARE_FLAGs these; they are DEFINED in
// dune_ppo_train.cc, which this test does not link. Defined here at their
// compiled defaults purely to satisfy the linker -- this test exercises the
// loader, and touches no code path that reads any of them. Same pattern as
// dune_pwo5_schema_test.cc.
ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, logit_cap, 10.0, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, true, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543, "");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

#include "dune_search_label_buffer.h"
#include "dune_sha256.h"

namespace {

using open_spiel::SearchLabelBuffer;
using open_spiel::SearchLabelFileEntry;
using open_spiel::SearchLabelRole;

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  if (ok) {
    std::cout << "  PASS  " << what << "\n";
  } else {
    std::cout << "  FAIL  " << what << "\n";
    ++g_failures;
  }
}

// A throwing error handler, so the loader's fatal paths are testable in
// process. SpielFatalError otherwise exits(1) and a failure case would end the
// test binary instead of being observed.
struct FatalError {
  std::string msg;
};
[[noreturn]] void ThrowingHandler(const std::string& msg) {
  // Echoed before throwing so an UNEXPECTED fatal (one raised outside an
  // ExpectFatal block) is diagnosable rather than an anonymous terminate().
  std::cout << std::flush;
  std::cerr << "[fatal] " << msg << std::endl;
  throw FatalError{msg};
}

// Runs `fn` and reports whether it raised a fatal error whose message contains
// `expect_substr`. A fatal error with the WRONG message is a failure: a guard
// that fires for the wrong reason is a guard that will stop firing when the
// wrong reason goes away.
void ExpectFatal(const std::string& what, const std::string& expect_substr,
                 const std::function<void()>& fn) {
  try {
    fn();
    std::cout << "  FAIL  " << what << "  (no fatal error raised)\n";
    ++g_failures;
  } catch (const FatalError& e) {
    if (e.msg.find(expect_substr) != std::string::npos) {
      std::cout << "  PASS  " << what << "\n";
    } else {
      std::cout << "  FAIL  " << what << "\n        expected substring: "
                << expect_substr << "\n        actual message:     " << e.msg
                << "\n";
      ++g_failures;
    }
  }
}

// ---------------------------------------------------------------------------
// Synthetic packs, byte-identical in layout to what dune_pwo5_prepare writes
// and to what SearchLabelBuffer::LoadFile reads.
// ---------------------------------------------------------------------------
constexpr int32_t kObsSize = 8;
constexpr int32_t kActionDim = 16;
constexpr int32_t kNumLegal = 2;

// The loader's own floors: training >= 8192, validation >= 1024. The synthetic
// packs must clear them or every case below would fail for the wrong reason.
constexpr int kTrainRows = 8200;
constexpr int kValRows = 1100;
// The LEGACY fixture must be big enough that the %11 rule's own validation
// share clears the loader's 1,024 floor: at 9,300 rows it yields only 876 and
// the case would fail on the floor rather than on the behaviour under test.
constexpr int kLegacyRows = 13000;

// state[0] tags which pack a row came from, so "no held-out row received a
// training gradient" is checkable by inspection of the loaded vectors.
constexpr float kTrainSignature = 1.0f;
constexpr float kValSignature = 999.0f;

template <typename T>
void Put(std::ofstream& o, T v) {
  o.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

void WritePack(const std::string& path, int rows, float signature) {
  std::ofstream o(path, std::ios::binary);
  Put<uint32_t>(o, 0x4c545344u);        // magic "DSTL"
  Put<uint32_t>(o, 2u);                 // schema
  Put<int32_t>(o, kObsSize);
  Put<int32_t>(o, kActionDim);
  Put<int32_t>(o, 200);                 // max_simulations
  Put<float>(o, 4.0f);                  // utility_divisor -- must be 4.0
  Put<float>(o, 0.3f);                  // puct_c
  Put<float>(o, 0.0f);                  // target_teacher_kl
  Put<int32_t>(o, 2);                   // min_visits
  Put<int32_t>(o, 0);                   // min_coverage
  Put<float>(o, 1.0f);                  // blueprint_temp
  Put<uint64_t>(o, 0xabcdef0123456789ull);
  Put<uint32_t>(o, 0u);                 // reserved
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < kObsSize; ++i) {
      Put<float>(o, i == 0 ? signature : static_cast<float>(r * 7 + i));
    }
    Put<int32_t>(o, kNumLegal);
    for (int k = 0; k < kNumLegal; ++k) {
      Put<int32_t>(o, k);
      Put<float>(o, 0.5f);
      Put<float>(o, 0.5f);
    }
    Put<float>(o, 0.0f);   // teacher_kl
    Put<int32_t>(o, 2);    // num_covered_actions
    Put<float>(o, 0.0f);   // eta
    Put<uint8_t>(o, 0);    // eta_capped
    Put<uint8_t>(o, static_cast<uint8_t>(r % 4));  // player_id
    Put<uint8_t>(o, 0);
    Put<uint8_t>(o, 0);
  }
}

struct FileSpec {
  std::string filename;
  std::string role;  // "" = no role declared
  bool omit_from_disk = false;
  std::string forced_sha;  // "" = the real digest
};

// Builds the manifest the loader will verify, reproducing its semantic object
// EXACTLY -- including ruling 5's `file_roles` key when any file declares a
// role. If this and the loader ever disagree, the fingerprint check fails and
// the test says so, which is the point.
void WriteManifest(const std::filesystem::path& dir,
                   const std::vector<FileSpec>& specs, int64_t declared_train,
                   int64_t declared_validation,
                   bool force_legacy_semantic = false) {
  namespace json = open_spiel::json;
  json::Object semantic;
  semantic["schema_version"] = json::Value(static_cast<int64_t>(2));
  semantic["base_seed"] = json::Value(static_cast<int64_t>(20260729));
  semantic["model_checkpoint_sha256"] = json::Value(std::string("deadbeef"));
  semantic["effective_search_config"] = json::Value(std::string("synthetic"));
  semantic["architecture"] = json::Value(std::string("synthetic"));
  semantic["training_label_count"] = json::Value(declared_train);
  semantic["validation_label_count"] = json::Value(declared_validation);

  std::vector<SearchLabelFileEntry> entries;
  bool any_role = false;
  json::Array files;
  for (const auto& s : specs) {
    SearchLabelFileEntry e;
    e.filename = s.filename;
    const std::filesystem::path p = dir / s.filename;
    e.sha256 = !s.forced_sha.empty()
                   ? s.forced_sha
                   : (std::filesystem::exists(p)
                          ? open_spiel::ComputeFileSHA256(p.string())
                          : std::string("0"));
    if (s.role == "train") {
      e.role = SearchLabelRole::kTrain;
      e.role_declared = true;
      any_role = true;
    } else if (s.role == "validation") {
      e.role = SearchLabelRole::kValidation;
      e.role_declared = true;
      any_role = true;
    } else if (!s.role.empty()) {
      // An intentionally invalid role, carried through verbatim so the loader
      // is the thing that rejects it.
      any_role = true;
    }
    entries.push_back(e);

    json::Object fo;
    fo["filename"] = json::Value(s.filename);
    fo["sha256"] = json::Value(e.sha256);
    if (!s.role.empty()) fo["role"] = json::Value(s.role);
    files.push_back(json::Value(fo));
  }
  if (any_role && !force_legacy_semantic) {
    semantic["file_roles"] =
        json::Value(SearchLabelBuffer::CanonicalFileRoles(entries));
  }

  const std::string fingerprint =
      open_spiel::ComputeStringSHA256(json::ToString(semantic));
  json::Object manifest = semantic;
  manifest["files"] = json::Value(files);
  manifest["search_label_fingerprint"] = json::Value(fingerprint);

  std::ofstream o((dir / "manifest.json").string());
  o << json::ToString(manifest, /*wrap=*/true) << "\n";
}

std::filesystem::path FreshDir(const std::string& name) {
  const std::filesystem::path d =
      std::filesystem::temp_directory_path() / ("pwo5_roles_" + name);
  std::filesystem::remove_all(d);
  std::filesystem::create_directories(d);
  return d;
}

// The standard role-aware fixture: a train pack and a validation pack.
std::filesystem::path RoleAwareDir(const std::string& name) {
  const auto d = FreshDir(name);
  WritePack((d / "labels_train_000.bin").string(), kTrainRows, kTrainSignature);
  WritePack((d / "labels_validation_000.bin").string(), kValRows,
            kValSignature);
  WriteManifest(d,
                {{"labels_train_000.bin", "train", false, ""},
                 {"labels_validation_000.bin", "validation", false, ""}},
                kTrainRows, kValRows);
  return d;
}

// SearchLabelBuffer owns a std::mutex and is therefore neither copyable nor
// movable, so it is configured in place rather than returned by value.
std::unique_ptr<SearchLabelBuffer> MakeBuffer() {
  auto b = std::make_unique<SearchLabelBuffer>();
  b->SetExpectedDimensions(kObsSize, kActionDim);
  return b;
}

}  // namespace

int main() {
  open_spiel::SetErrorHandler(ThrowingHandler);

  std::cout << "\n=== PWO-5 ruling 5: manifest file roles in the auxiliary "
               "label loader ===\n\n";

  // -------------------------------------------------------------------------
  std::cout << "[rule 1] a declared role assigns EVERY row of its file\n";
  {
    const auto d = RoleAwareDir("happy");
    auto b = MakeBuffer();
    b->LoadFromDirectory(d.string());
    Check(b->role_aware(), "the manifest is recognized as role-aware");
    Check(b->Size() == static_cast<size_t>(kTrainRows),
          "train-role rows land in the TRAINING bucket: " +
              std::to_string(b->Size()) + " == " + std::to_string(kTrainRows));
    Check(b->ValidationSize() == static_cast<size_t>(kValRows),
          "validation-role rows land in the VALIDATION bucket: " +
              std::to_string(b->ValidationSize()) + " == " +
              std::to_string(kValRows));
    // The whole point of the ruling: the %11 rule is BYPASSED, not tuned
    // around. Under the legacy rule roughly 1/11 of the 8,200 train rows would
    // have been diverted, so an exact 8,200 is the proof it was not consulted.
    Check(b->Size() == static_cast<size_t>(kTrainRows),
          "IsValidationLabel was NOT consulted for the train-role file "
          "(no ~1/11 diversion)");
  }

  // -------------------------------------------------------------------------
  std::cout << "\n[rule 4] NO HELD-OUT ROW CAN RECEIVE A TRAINING GRADIENT\n";
  {
    const auto d = RoleAwareDir("isolation");
    auto b = MakeBuffer();
    b->LoadFromDirectory(d.string());

    // (a) exhaustive: inspect every row the sampler can reach.
    bool any_val_in_train = false;
    for (size_t i = 0; i < b->Size(); ++i) {
      if (b->TrainingLabelForTest(i).state[0] == kValSignature) {
        any_val_in_train = true;
        break;
      }
    }
    Check(!any_val_in_train,
          "EXHAUSTIVE: no validation-signature row exists anywhere in the "
          "training vector (all " + std::to_string(b->Size()) + " rows checked)");

    bool all_val_are_val = true;
    for (size_t i = 0; i < b->ValidationSize(); ++i) {
      if (b->ValidationLabelForTest(i).state[0] != kValSignature) {
        all_val_are_val = false;
        break;
      }
    }
    Check(all_val_are_val,
          "EXHAUSTIVE: every row of the validation bucket came from the "
          "validation-role pack");

    // (b) behavioural: draw hard from the sampler the distillation loss uses.
    std::mt19937 rng(12345);
    const int kDraws = 400;
    const int kPerDraw = 512;
    int64_t drawn = 0, leaked = 0;
    for (int i = 0; i < kDraws; ++i) {
      auto batch = b->Sample(kPerDraw, &rng);
      for (const auto& lbl : batch) {
        ++drawn;
        if (lbl.state[0] == kValSignature) ++leaked;
      }
    }
    Check(leaked == 0,
          "BEHAVIOURAL: " + std::to_string(drawn) +
              " sampled rows, 0 from the held-out pack (leaked=" +
              std::to_string(leaked) + ")");
  }

  // -------------------------------------------------------------------------
  std::cout << "\n[rule 2] a ROLELESS legacy manifest keeps the %11 behaviour\n";
  {
    const auto d = FreshDir("legacy");
    WritePack((d / "labels_000.bin").string(), kLegacyRows, kTrainSignature);
    // Count what IsValidationLabel will do, so the declared counts are the
    // realized ones. This mirrors what the legacy generator did.
    int64_t expect_val = 0;
    {
      std::vector<int32_t> legal{0, 1};
      for (int r = 0; r < kLegacyRows; ++r) {
        std::vector<float> state(kObsSize);
        state[0] = kTrainSignature;
        for (int i = 1; i < kObsSize; ++i) state[i] = static_cast<float>(r * 7 + i);
        if (open_spiel::IsValidationLabel(state, legal, r % 4)) ++expect_val;
      }
    }
    const int64_t expect_train = kLegacyRows - expect_val;
    WriteManifest(d, {{"labels_000.bin", "", false, ""}}, expect_train,
                  expect_val);

    auto b = MakeBuffer();
    b->LoadFromDirectory(d.string());
    Check(!b->role_aware(), "a roleless manifest is NOT role-aware");
    Check(b->Size() == static_cast<size_t>(expect_train) &&
              b->ValidationSize() == static_cast<size_t>(expect_val),
          "legacy %11 split reproduced exactly: " + std::to_string(b->Size()) +
              " train / " + std::to_string(b->ValidationSize()) +
              " validation (expected " + std::to_string(expect_train) + " / " +
              std::to_string(expect_val) + ")");
    Check(expect_val > 0,
          "the legacy path really did divert rows (" +
              std::to_string(expect_val) + "), so the comparison is not vacuous");
  }

  // -------------------------------------------------------------------------
  std::cout << "\n[legacy fingerprint preservation]\n";
  {
    // A roleless manifest's semantic object must be BYTE-IDENTICAL to what it
    // has always been. If ruling 5 had added `file_roles` unconditionally,
    // every committed legacy fingerprint would stop verifying.
    const auto d = FreshDir("legacyfp");
    WritePack((d / "labels_000.bin").string(), kLegacyRows, kTrainSignature);
    namespace json = open_spiel::json;
    json::Object semantic;
    semantic["schema_version"] = json::Value(static_cast<int64_t>(2));
    semantic["base_seed"] = json::Value(static_cast<int64_t>(20260729));
    semantic["model_checkpoint_sha256"] = json::Value(std::string("deadbeef"));
    semantic["effective_search_config"] = json::Value(std::string("synthetic"));
    semantic["architecture"] = json::Value(std::string("synthetic"));
    semantic["training_label_count"] = json::Value(static_cast<int64_t>(8500));
    semantic["validation_label_count"] = json::Value(static_cast<int64_t>(1200));
    const std::string legacy_fp =
        open_spiel::ComputeStringSHA256(json::ToString(semantic));
    // The same object with a file_roles key is a DIFFERENT fingerprint --
    // which is the property that makes role tampering detectable.
    json::Object with_roles = semantic;
    with_roles["file_roles"] = json::Value(std::string("x:y:train;"));
    const std::string role_fp =
        open_spiel::ComputeStringSHA256(json::ToString(with_roles));
    Check(legacy_fp != role_fp,
          "adding file_roles CHANGES the fingerprint, so a role edit cannot "
          "pass unnoticed");
  }

  // -------------------------------------------------------------------------
  std::cout << "\n[rule 3 and the rejection set]\n";

  ExpectFatal("unknown role is rejected", "unknown role", [] {
    const auto d = FreshDir("badrole");
    WritePack((d / "labels_train_000.bin").string(), kTrainRows, kTrainSignature);
    WritePack((d / "labels_validation_000.bin").string(), kValRows, kValSignature);
    WriteManifest(d,
                  {{"labels_train_000.bin", "training", false, ""},
                   {"labels_validation_000.bin", "validation", false, ""}},
                  kTrainRows, kValRows);
    auto b = MakeBuffer();
    b->LoadFromDirectory(d.string());
  });

  ExpectFatal("duplicate manifest entry is rejected", "duplicate manifest", [] {
    const auto d = FreshDir("dup");
    WritePack((d / "labels_train_000.bin").string(), kTrainRows, kTrainSignature);
    WritePack((d / "labels_validation_000.bin").string(), kValRows, kValSignature);
    WriteManifest(d,
                  {{"labels_train_000.bin", "train", false, ""},
                   {"labels_train_000.bin", "train", false, ""},
                   {"labels_validation_000.bin", "validation", false, ""}},
                  kTrainRows, kValRows);
    auto b = MakeBuffer();
    b->LoadFromDirectory(d.string());
  });

  ExpectFatal("a listed file that does not exist is rejected",
              "does not exist", [] {
    const auto d = FreshDir("missing");
    WritePack((d / "labels_validation_000.bin").string(), kValRows, kValSignature);
    WriteManifest(d,
                  {{"labels_train_000.bin", "train", true, "aa"},
                   {"labels_validation_000.bin", "validation", false, ""}},
                  kTrainRows, kValRows);
    auto b = MakeBuffer();
    b->LoadFromDirectory(d.string());
  });

  ExpectFatal("a wrong sha256 is rejected", "Hash mismatch", [] {
    const auto d = FreshDir("badhash");
    WritePack((d / "labels_train_000.bin").string(), kTrainRows, kTrainSignature);
    WritePack((d / "labels_validation_000.bin").string(), kValRows, kValSignature);
    WriteManifest(d,
                  {{"labels_train_000.bin", "train", false,
                    std::string(64, 'a')},
                   {"labels_validation_000.bin", "validation", false, ""}},
                  kTrainRows, kValRows);
    auto b = MakeBuffer();
    b->LoadFromDirectory(d.string());
  });

  ExpectFatal("a non-.bin file type is rejected", "is not a .bin label pack",
              [] {
    const auto d = FreshDir("badtype");
    WritePack((d / "labels_train_000.dat").string(), kTrainRows, kTrainSignature);
    WritePack((d / "labels_validation_000.bin").string(), kValRows, kValSignature);
    WriteManifest(d,
                  {{"labels_train_000.dat", "train", false, ""},
                   {"labels_validation_000.bin", "validation", false, ""}},
                  kTrainRows, kValRows);
    auto b = MakeBuffer();
    b->LoadFromDirectory(d.string());
  });

  ExpectFatal("a path-traversing filename is rejected", "invalid filename", [] {
    const auto d = FreshDir("traversal");
    WritePack((d / "labels_validation_000.bin").string(), kValRows, kValSignature);
    WriteManifest(d,
                  {{"../labels_train_000.bin", "train", true, "aa"},
                   {"labels_validation_000.bin", "validation", false, ""}},
                  kTrainRows, kValRows);
    auto b = MakeBuffer();
    b->LoadFromDirectory(d.string());
  });

  ExpectFatal("declared/realized count mismatch is rejected",
              "realized role counts do not match the manifest", [] {
    const auto d = FreshDir("countmismatch");
    WritePack((d / "labels_train_000.bin").string(), kTrainRows, kTrainSignature);
    WritePack((d / "labels_validation_000.bin").string(), kValRows, kValSignature);
    // Declares 20,582 -- the registered figure -- while the pack holds 8,200.
    // This is EXACTLY the disagreement ruling 5 item 3 was written about.
    WriteManifest(d,
                  {{"labels_train_000.bin", "train", false, ""},
                   {"labels_validation_000.bin", "validation", false, ""}},
                  20582, kValRows);
    auto b = MakeBuffer();
    b->LoadFromDirectory(d.string());
  });

  ExpectFatal("an UNLISTED .bin in a role-aware directory is rejected",
              "is not listed in manifest.json", [] {
    const auto d = RoleAwareDir("unlisted");
    // A pack nobody hashed and nobody gave a role, sitting in a directory the
    // loader used to glob wholesale.
    WritePack((d / "labels_sneaky_000.bin").string(), 50, kValSignature);
    auto b = MakeBuffer();
    b->LoadFromDirectory(d.string());
  });

  ExpectFatal("swapping the roles without re-fingerprinting is rejected",
              "search_label_fingerprint mismatch", [] {
    const auto d = RoleAwareDir("tamper");
    // Rewrite manifest.json with the roles exchanged, keeping the ORIGINAL
    // fingerprint -- the attack ruling 5's file_roles binding exists to stop.
    std::ifstream in((d / "manifest.json").string());
    std::string text((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    in.close();
    const std::size_t a = text.find("\"role\": \"train\"");
    if (a != std::string::npos) {
      text.replace(a, std::string("\"role\": \"train\"").size(),
                   "\"role\": \"validation\"");
    }
    std::ofstream out((d / "manifest.json").string());
    out << text;
    out.close();
    auto b = MakeBuffer();
    b->LoadFromDirectory(d.string());
  });

  ExpectFatal("a rescan of a role-aware directory is refused",
              "Role-aware datasets are closed at load time", [] {
    const auto d = RoleAwareDir("rescan");
    auto b = MakeBuffer();
    b->LoadFromDirectory(d.string());
    // LoadNewFiles is the online-collector path. On a role-aware dataset it
    // would reload the manifest's files WITHOUT their roles -- the precise
    // defect ruling 5 removes -- so it must refuse rather than silently
    // succeed.
    b->LoadNewFiles(d.string());
  });

  std::cout << "\n";
  if (g_failures == 0) {
    std::cout << "ALL RULING-5 LOADER ROLE TESTS PASSED\n";
    return 0;
  }
  std::cout << g_failures << " CHECK(S) FAILED\n";
  return 1;
}
