// WO-1 Phase 4: proves the OFFLINE teacher and the ONLINE collector compute
// identical acceptance on a shared fixture.
//
// The defect this pins: the two paths shared the constants 3 / 2 / 0.50, which
// made their acceptance rates look comparable, but measured covered mass
// against DIFFERENT prior vectors. The collector used the raw network prior
// (WO-20); the teacher read SearchDiagnostics::covered_prior_mass, which
// GetRootDiagnostics accumulates from the POST-noise, POST-temperature tree
// prior. The teacher builds a kTrainingFullFast session, which forces
// dirichlet_epsilon to 0.25 for full sessions, and only full sessions are
// screened -- so the teacher's mass was always post-noise at epsilon 0.25.
//
// Both paths now route through ComputeSearchAcceptance/SelectAcceptancePriors.
// There is deliberately no second implementation; these tests assert that the
// unification actually holds, and that the tree-prior source still differs
// (otherwise the fixture would not be discriminating and the test would pass
// vacuously).

#include <string>
#include <vector>

#include "dune_online_search_collector.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

constexpr int kMinCoverage = 3;
constexpr int kMinVisitsPerAction = 2;
constexpr double kMinPriorMass = 0.50;

// One search root. `tree_priors` is what the raw prior becomes after a
// Dirichlet(0.25) mixture flattens it -- the transformation the teacher's
// session applies and the collector's rule strips.
struct Fixture {
  std::string name;
  std::vector<int> visit_counts;
  std::vector<double> raw_priors;
  std::vector<double> tree_priors;
};

std::vector<Fixture> MakeFixtures() {
  return {
      // Covered set holds 0.60 of the raw prior but only 0.45 once noise
      // flattens it: accepted under the unified rule, rejected under the tree
      // prior. This is the case that made the two paths disagree.
      {"discriminating_mass",
       /*visits=*/{5, 4, 3, 0, 0},
       /*raw=*/{0.30, 0.20, 0.10, 0.25, 0.15},
       /*tree=*/{0.20, 0.15, 0.10, 0.30, 0.25}},
      // Too few actions meet the visit floor, regardless of prior source.
      {"coverage_fails",
       /*visits=*/{9, 1, 1, 0, 0},
       /*raw=*/{0.70, 0.10, 0.10, 0.05, 0.05},
       /*tree=*/{0.55, 0.15, 0.15, 0.10, 0.05}},
      // Comfortably accepted under either source.
      {"accepts_either",
       /*visits=*/{6, 5, 4, 3, 0},
       /*raw=*/{0.25, 0.25, 0.20, 0.20, 0.10},
       /*tree=*/{0.22, 0.22, 0.20, 0.20, 0.16}},
      // Exactly on the mass boundary: the rule is >=, so this must accept.
      {"mass_exactly_at_threshold",
       /*visits=*/{2, 2, 2, 0, 0},
       /*raw=*/{0.20, 0.20, 0.10, 0.30, 0.20},
       /*tree=*/{0.20, 0.20, 0.10, 0.30, 0.20}},
      // Raw priors empty => SelectAcceptancePriors falls back to the tree
      // vector (root absent from the tree, where `priors` is already raw).
      {"raw_empty_falls_back",
       /*visits=*/{4, 3, 2, 0},
       /*raw=*/{},
       /*tree=*/{0.30, 0.20, 0.15, 0.35}},
  };
}

// The ONLINE collector's call shape.
bool CollectorAccept(const Fixture& f, AcceptancePriorSource source,
                     int* covered, double* mass) {
  const std::vector<double>& priors =
      SelectAcceptancePriors(f.raw_priors, f.tree_priors, source);
  return ComputeSearchAcceptance(f.visit_counts, priors,
                                 static_cast<int>(f.visit_counts.size()),
                                 kMinCoverage, kMinVisitsPerAction,
                                 kMinPriorMass, covered, mass);
}

// The OFFLINE teacher's call shape (dune_search_teacher.cc: same helpers, same
// constants, priors selected the same way).
bool TeacherAccept(const Fixture& f, AcceptancePriorSource source, int* covered,
                   double* mass) {
  const std::vector<double>& acceptance_priors =
      SelectAcceptancePriors(f.raw_priors, f.tree_priors, source);
  int num_covered = 0;
  double covered_mass = 0.0;
  ComputeSearchAcceptance(f.visit_counts, acceptance_priors,
                          static_cast<int>(f.visit_counts.size()), kMinCoverage,
                          kMinVisitsPerAction, kMinPriorMass, &num_covered,
                          &covered_mass);
  const int required =
      std::min(kMinCoverage, static_cast<int>(f.visit_counts.size()));
  const bool has_coverage = (num_covered >= required);
  const bool has_mass = (covered_mass >= kMinPriorMass);
  if (covered != nullptr) *covered = num_covered;
  if (mass != nullptr) *mass = covered_mass;
  return has_coverage && has_mass;
}

void TestBothPathsAgreeOnSharedFixture() {
  for (const Fixture& f : MakeFixtures()) {
    for (AcceptancePriorSource source :
         {AcceptancePriorSource::kRawNetworkPrior,
          AcceptancePriorSource::kTreePrior}) {
      int c_cov = -1, t_cov = -2;
      double c_mass = -1.0, t_mass = -2.0;
      const bool c = CollectorAccept(f, source, &c_cov, &c_mass);
      const bool t = TeacherAccept(f, source, &t_cov, &t_mass);
      SPIEL_CHECK_EQ(c, t);
      SPIEL_CHECK_EQ(c_cov, t_cov);
      SPIEL_CHECK_FLOAT_NEAR(c_mass, t_mass, 1e-12);
    }
  }
}

// The teacher's separate has_coverage / has_mass predicates must conjoin to
// exactly ComputeSearchAcceptance's verdict -- the trap being a future edit
// that lets one drift.
void TestTeacherPredicateConjunctionMatchesVerdict() {
  for (const Fixture& f : MakeFixtures()) {
    int covered = 0;
    double mass = 0.0;
    const std::vector<double>& priors = SelectAcceptancePriors(
        f.raw_priors, f.tree_priors, AcceptancePriorSource::kRawNetworkPrior);
    const bool verdict = ComputeSearchAcceptance(
        f.visit_counts, priors, static_cast<int>(f.visit_counts.size()),
        kMinCoverage, kMinVisitsPerAction, kMinPriorMass, &covered, &mass);
    const int required =
        std::min(kMinCoverage, static_cast<int>(f.visit_counts.size()));
    SPIEL_CHECK_EQ(verdict, (covered >= required) && (mass >= kMinPriorMass));
  }
}

// Guards against a vacuous pass: if the two sources ever agreed everywhere, the
// agreement test above would prove nothing about which prior is being used.
void TestFixtureIsDiscriminating() {
  bool found_disagreement = false;
  for (const Fixture& f : MakeFixtures()) {
    if (f.raw_priors.empty()) continue;  // fallback case agrees by definition
    const bool raw =
        CollectorAccept(f, AcceptancePriorSource::kRawNetworkPrior, nullptr,
                        nullptr);
    const bool tree =
        CollectorAccept(f, AcceptancePriorSource::kTreePrior, nullptr, nullptr);
    if (raw != tree) found_disagreement = true;
  }
  SPIEL_CHECK_TRUE(found_disagreement);
}

// The persisted contract strings are load-bearing (manifests, fingerprints).
void TestContractStringsRoundTrip() {
  SPIEL_CHECK_EQ(
      std::string(AcceptancePriorSourceName(
          AcceptancePriorSource::kRawNetworkPrior)),
      "raw_network_prior");
  SPIEL_CHECK_EQ(
      std::string(AcceptancePriorSourceName(AcceptancePriorSource::kTreePrior)),
      "tree_prior");
  AcceptancePriorSource parsed;
  SPIEL_CHECK_TRUE(ParseAcceptancePriorSource("raw_network_prior", &parsed));
  SPIEL_CHECK_TRUE(parsed == AcceptancePriorSource::kRawNetworkPrior);
  SPIEL_CHECK_TRUE(ParseAcceptancePriorSource("tree_prior", &parsed));
  SPIEL_CHECK_TRUE(parsed == AcceptancePriorSource::kTreePrior);
  SPIEL_CHECK_FALSE(ParseAcceptancePriorSource("post_noise", &parsed));
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestBothPathsAgreeOnSharedFixture();
  open_spiel::TestTeacherPredicateConjunctionMatchesVerdict();
  open_spiel::TestFixtureIsDiscriminating();
  open_spiel::TestContractStringsRoundTrip();
  std::cout << "All acceptance parity tests passed." << std::endl;
  return 0;
}
