#include <gtest/gtest.h>

#include "target_selector.h"

TEST(TargetSelectorTest, SelectDefaultReturnsFixedCenter) {
  decision::TargetSelector selector;
  const auto target = selector.selectDefault();

  EXPECT_TRUE(target.valid);
  EXPECT_EQ(target.mode, "fixed_center");
  EXPECT_NEAR(target.target_land.x(), 2.055, 1e-12);
  EXPECT_NEAR(target.target_land.y(), -0.7625, 1e-12);
  EXPECT_NEAR(target.target_land.z(), 0.0, 1e-12);
  EXPECT_NEAR(target.delta_t_flight, 0.5, 1e-12);
  EXPECT_NEAR(target.desired_ball_speed, -1.0, 1e-12);
  EXPECT_NEAR(target.max_ball_out_speed, -1.0, 1e-12);
  EXPECT_NEAR(target.max_racket_speed, 6.0, 1e-12);
  EXPECT_NEAR(target.net_clearance_margin, 0.03, 1e-12);
}
