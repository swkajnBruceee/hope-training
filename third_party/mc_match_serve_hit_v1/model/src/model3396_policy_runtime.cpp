#include "model3396_policy_runtime.hpp"

// The profile deliberately reuses the common A3PolicyRuntime implementation.
// Its only model-specific runtime contract is validated by the profile main:
// obs_raw[126] -> actions[14].
