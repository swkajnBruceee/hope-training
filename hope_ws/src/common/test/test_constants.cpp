#include <gtest/gtest.h>

#include "constants.h"

TEST(ConstantsTest, TableParamsDefaults) {
  common::TableParams t;
  EXPECT_DOUBLE_EQ(t.length, 2.74);
  EXPECT_DOUBLE_EQ(t.width, 1.525);
  EXPECT_DOUBLE_EQ(t.height, 0.76);
  EXPECT_DOUBLE_EQ(t.net_x, 1.37);
  EXPECT_DOUBLE_EQ(t.net_height, 0.1525);
  EXPECT_DOUBLE_EQ(t.net_overhang, 0.15);
}

TEST(ConstantsTest, BallPhysicsDefaults) {
  common::BallPhysics b;
  EXPECT_DOUBLE_EQ(b.k, 0.09375);
  EXPECT_DOUBLE_EQ(b.C_h, 0.649);
  EXPECT_DOUBLE_EQ(b.C_v, 0.906);
  EXPECT_DOUBLE_EQ(b.g.x(), 0.0);
  EXPECT_DOUBLE_EQ(b.g.y(), 0.0);
  EXPECT_DOUBLE_EQ(b.g.z(), -9.81);
  EXPECT_DOUBLE_EQ(b.radius, 0.02);
  EXPECT_DOUBLE_EQ(b.mass, 0.0027);
}

TEST(ConstantsTest, PlannerConfigDefaults) {
  common::PlannerConfig c;
  EXPECT_EQ(c.poly_order, 2);
  EXPECT_EQ(c.fit_window, 31);
  EXPECT_DOUBLE_EQ(c.fit_window_s, 31.0 / 360.0);
  EXPECT_DOUBLE_EQ(c.mocap_hz, 360.0);
  EXPECT_DOUBLE_EQ(c.dt_integrate, 0.001);
  EXPECT_DOUBLE_EQ(c.max_predict_time, 2.0);
  EXPECT_DOUBLE_EQ(c.bounce_z_tol, 0.005);
  EXPECT_DOUBLE_EQ(c.x_hit, 0.0);
  EXPECT_DOUBLE_EQ(c.target_land.x(), 2.055);
  EXPECT_DOUBLE_EQ(c.target_land.y(), -0.7625);
  EXPECT_DOUBLE_EQ(c.target_land.z(), 0.0);
  EXPECT_DOUBLE_EQ(c.delta_t_flight, 0.5);
  EXPECT_DOUBLE_EQ(c.C_r, 0.842);
  EXPECT_DOUBLE_EQ(c.racket_radius, 0.075);
  EXPECT_DOUBLE_EQ(c.racket_marker_plane_gap, 0.0365);
}
