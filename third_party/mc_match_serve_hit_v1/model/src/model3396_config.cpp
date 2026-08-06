#include "model3396_config.hpp"

namespace a3_deploy::model3396 {
static_assert(kObsDim == 126 && kActionDim == 14);
static_assert(kActionMask[12] == 0.0f && kActionMask[13] == 0.0f);
}
