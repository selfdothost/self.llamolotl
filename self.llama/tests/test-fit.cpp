// Unit tests for the fit-to-VRAM ceiling helper (T-002, self.llamolotl#27).
//
// Covers cavekit-inference.md R6 AC1/AC2: a config/preset-set n_gpu_layers is treated as a
// *ceiling* the fit-to-VRAM fill loop may reduce from, not a hard pin that aborts the fallback.
// The step-3 back-to-front fill loop clamps its per-device high bound (hp_ngl + 1) through
// common_fit_clamp_ngl_ceiling(), so these tests exercise that clamp decision in isolation --
// no GPU, no model load, no device-memory query required. The helper is a pure function
// exposed via fit.h precisely so this can be tested GPU-free.

#include "testing.h"

#include "fit.h"

#include <cstdint>
#include <limits>

// AC1: a ceiling below the available high bound reduces the result rather than aborting.
// (The pre-change code threw "n_gpu_layers already set by user, abort" here; the ceiling path
// instead returns a smaller, non-zero bound so a partial offload can be selected.)
static void test_ceiling_reduces(testing & t) {
    // model has 32 real layers -> high bound is hp_ngl + 1 == 33; config ceiling is 20.
    const uint32_t high    = 33;
    const int64_t  ceiling = 20;

    const uint32_t got = common_fit_clamp_ngl_ceiling(high, ceiling);

    t.assert_true("ceiling reduces the high bound below the full-offload bound", got < high);
    t.assert_true("reduced bound is still a real (non-zero) offload, not an abort", got > 0);
}

// AC2: the reduced value is the min of the two -> "largest that fits under the ceiling".
static void test_ceiling_is_min(testing & t) {
    // ceiling below the high bound -> the ceiling wins (largest offload that fits under it).
    t.assert_equal("ceiling below high bound -> clamp to ceiling",
                   uint32_t(20), common_fit_clamp_ngl_ceiling(33, 20));

    // ceiling above the high bound -> the high bound wins (can't offload more layers than exist).
    t.assert_equal("ceiling above high bound -> clamp to high bound",
                   uint32_t(33), common_fit_clamp_ngl_ceiling(33, 40));

    // ceiling exactly equal -> either interpretation gives the same value.
    t.assert_equal("ceiling == high bound -> unchanged",
                   uint32_t(33), common_fit_clamp_ngl_ceiling(33, 33));

    // ceiling of 0 -> a valid "offload nothing" request, distinct from the unset sentinel.
    t.assert_equal("ceiling of 0 clamps to 0 (offload nothing)",
                   uint32_t(0), common_fit_clamp_ngl_ceiling(33, 0));
}

// (c): an unset (auto) ceiling of -1 leaves the high bound untouched, so the fill loop keeps
// assigning as many layers as fit in VRAM (today's behaviour on the auto/multi-device paths).
static void test_unset_ceiling_passthrough(testing & t) {
    t.assert_equal("ngl_ceiling == -1 (auto/unset) leaves the high bound unchanged",
                   uint32_t(33), common_fit_clamp_ngl_ceiling(33, -1));

    // any negative value is the "no ceiling" sentinel (e.g. -2 == "all"), so it must pass through too.
    t.assert_equal("any negative ceiling passes the high bound through unchanged",
                   uint32_t(33), common_fit_clamp_ngl_ceiling(33, -2));

    // passthrough must not clamp even for a large high bound.
    const uint32_t big = std::numeric_limits<uint32_t>::max();
    t.assert_equal("unset ceiling does not clamp a large high bound",
                   big, common_fit_clamp_ngl_ceiling(big, -1));
}

int main(int argc, char ** argv) {
    testing t;

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("ceiling_reduces",           test_ceiling_reduces);
    t.test("ceiling_is_min",            test_ceiling_is_min);
    t.test("unset_ceiling_passthrough", test_unset_ceiling_passthrough);

    return t.summary();
}
