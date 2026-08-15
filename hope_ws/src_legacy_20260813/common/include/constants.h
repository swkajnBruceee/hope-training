#ifndef COMMON_CONSTANTS_H
#define COMMON_CONSTANTS_H

#include <Eigen/Dense>

namespace common {

/**
 * ITTF regulation table dimensions in the HOPE canonical frame.
 * Source: HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 2.
 */
struct TableParams {
  double length = 2.74;          // m, along X
  double width = 1.525;          // m, along -Y
  double height = 0.76;          // m, table surface above floor
  double net_x = 1.37;           // m, net position along X
  double net_height = 0.1525;    // m, net height above table surface
  double net_overhang = 0.15;    // m, net extends past each table edge in Y
};

/**
 * Aerodynamic and restitution parameters.
 * Calibrated from recorded ball trajectories by fitting observed
 * accelerations and bounce velocity ratios (see calibrate_ball_physics).
 */
struct BallPhysics {
  double k = 0.09375;           // quadratic drag coefficient (1/m): a_drag = -k|v|v
  double C_h = 0.649;           // table tangential velocity retention beta_table
  double C_v = 0.906;           // table normal restitution e_table
  Eigen::Vector3d g = (Eigen::Vector3d() << 0.0, 0.0, -9.81).finished();
  double radius = 0.02;         // ball radius, 40 mm diameter
  double mass = 0.0027;         // 2.7 g
};

/**
 * Planner tuning parameters.
 */
struct PlannerConfig {
  // State estimation
  int poly_order = 2;            // polynomial fit order
  int fit_window = 31;           // number of position samples for velocity fit
  double fit_window_s = 31.0 / 360.0;  // target time span for the fit window (s)
  double mocap_hz = 360.0;       // motion capture frame rate

  // Trajectory prediction
  double dt_integrate = 0.001;   // integration time step (s)
  double max_predict_time = 2.0; // max forward prediction horizon (s)
  double bounce_z_tol = 0.005;   // z threshold for bounce detection (m)

  // Racket planning
  double x_hit = 0.0;            // virtual hitting plane X coordinate (m)
  Eigen::Vector3d target_land = (Eigen::Vector3d() << 2.055, -0.7625, 0.0).finished();
                                  // center of opponent's half
  double delta_t_flight = 0.5;   // desired post-strike flight time (s)
  double C_r = 0.842;            // ball-racket normal restitution e_racket
  double racket_radius = 0.075;  // 7.5 cm paddle radius
  double racket_marker_plane_gap = 0.0365;  // ball center to racket marker-plane contact gap (m)
};

}  // namespace common

#endif  // COMMON_CONSTANTS_H
