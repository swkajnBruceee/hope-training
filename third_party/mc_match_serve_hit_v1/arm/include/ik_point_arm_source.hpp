#pragma once

#include "control_source.hpp"
#include "fk.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>

namespace a3_deploy::control {

// Traditional damped-least-squares strike planner for the seven right-arm
// joints.
//
// Position, racket-face normal and racket-centre velocity are expressed in the
// pelvis/base frame. A swing-specific IK branch removes the forehand/backhand
// elbow ambiguity. A quintic approach reaches the requested pose and velocity
// at the requested strike time, then a continuous follow-through decelerates
// to rest. The source never owns the waist; the shared command composer keeps
// all three waist targets at zero.
class IkPointArmSource final : public IArmControlSource {
 public:
  explicit IkPointArmSource(double update_hz = 50.0) noexcept;

  bool Load(const std::string& goal_path, const std::string& robot_xml,
            std::string& error);
  bool SetGoal(const ArmGoal& goal) noexcept override;
  // Live planners use this mode so the IK source first enters a configured
  // forward ready pose, then waits there for validated external goals. It
  // must never fall back to random target generation.
  void RequireExternalGoals(bool required, bool prepare_ready = true) noexcept;
  // Match coordinator owns the rally-ready transition and recovery. Disable
  // the IK source's historical Cartesian preparation/recovery plan so it only
  // executes validated live strikes.
  void SetAutomaticReadyRecovery(bool enabled) noexcept {
    automatic_ready_recovery_ = enabled;
  }
  // Seed the next live strike with the command that the outer match
  // coordinator is currently publishing. This lets a newly arrived 10-D goal
  // preempt a rally-ready/recovery trajectory without a one-frame jump back to
  // measured position or an artificial zero-velocity restart.
  void SeedCommandState(const std::array<double, 7>& q,
                        const std::array<double, 7>& dq) noexcept;
  bool CanAcceptExternalGoal() const noexcept;
  void Reset() noexcept override;
  bool Update(const robot_io::RobotState& state, double time_s,
              ArmTarget& target) noexcept override;
  double UpdateHz() const noexcept override { return update_hz_; }

  const char* PhaseName() const noexcept;
  const std::array<double, 3>& TargetPosition() const noexcept {
    return target_position_b_m_;
  }
  const std::array<double, 3>& RacketPosition() const noexcept {
    return last_racket_position_b_m_;
  }
  bool EvaluateRacketPose(const std::array<double, 31>& q_sdk,
                          std::array<double, 3>& position_b_m,
                          std::array<double, 3>& normal_b) const;
  const std::array<double, 3>& TargetVelocity() const noexcept {
    return target_velocity_b_mps_;
  }
  const std::array<double, 3>& AchievedStrikeVelocity() const noexcept {
    return achieved_strike_velocity_b_mps_;
  }
  const std::array<double, 7>& SolvedJointPosition() const noexcept {
    return solved_q_;
  }
  const std::array<double, 7>& SolvedJointVelocity() const noexcept {
    return solved_dq_;
  }
  const std::array<double, 7>& FollowJointPosition() const noexcept {
    return follow_q_;
  }
  const char* SwingName() const noexcept;
  double TimeToStrikeS() const noexcept { return time_to_strike_s_; }
  double RequestedStrikeTimeS() const noexcept {
    return requested_strike_time_s_;
  }
  double PlannedStrikeTimeS() const noexcept { return strike_time_s_; }
  double ImpactTransitionS() const noexcept { return impact_transition_s_; }
  double TimingExtensionS() const noexcept {
    return std::max(0.0, strike_time_s_ - requested_strike_time_s_);
  }
  // Kinematic lower bound using the empirically conservative sustained arm
  // speed. This is diagnostic only; an external late ball command is still
  // executed rather than discarded.
  double TrackingMinimumStrikeTimeS() const noexcept {
    return tracking_minimum_strike_time_s_;
  }
  double PositionErrorM() const noexcept { return last_position_error_m_; }
  double NormalErrorDeg() const noexcept { return last_normal_error_deg_; }
  double SolvedPositionErrorM() const noexcept {
    return solved_position_error_m_;
  }
  double SolvedNormalErrorDeg() const noexcept {
    return solved_normal_error_deg_;
  }
  int SolveIterations() const noexcept { return solve_iterations_; }
  std::uint64_t StrikeCount() const noexcept { return strike_count_; }
  std::uint64_t RejectedTargetCount() const noexcept {
    return rejected_target_count_;
  }
  std::uint64_t SolveRejectCount() const noexcept {
    return solve_reject_count_;
  }
  std::uint64_t VelocityRejectCount() const noexcept {
    return velocity_reject_count_;
  }
  std::uint64_t FollowRejectCount() const noexcept {
    return follow_reject_count_;
  }
  std::uint64_t TrajectoryRejectCount() const noexcept {
    return trajectory_reject_count_;
  }
  double LastVelocitySolveErrorMps() const noexcept {
    return last_velocity_solve_error_mps_;
  }
  double LastNormalRateRadS() const noexcept {
    return last_normal_rate_rad_s_;
  }
  const std::array<double, 3>& ImpactMeasuredVelocity() const noexcept {
    return impact_measured_velocity_b_mps_;
  }
  double ImpactPositionErrorM() const noexcept {
    return impact_position_error_m_;
  }
  double ImpactNormalErrorDeg() const noexcept {
    return impact_normal_error_deg_;
  }
  double ImpactTimingErrorS() const noexcept {
    return impact_timing_error_s_;
  }
  double ImpactSourceDeadlineErrorS() const noexcept {
    return impact_source_deadline_error_s_;
  }
  double ImpactMaxJointErrorRad() const noexcept {
    return impact_max_joint_error_rad_;
  }
  double LastPlanningDurationMs() const noexcept {
    return last_planning_duration_ms_;
  }
  // Minimum planned racket-centre distance in front of the conservative
  // torso protection plane. A non-negative value means every sampled point
  // of the live strike stayed outside the protected body envelope.
  double PlannedMinimumRacketBodyClearanceM() const noexcept {
    return planned_min_racket_body_clearance_m_;
  }
  const char* LastTrajectoryRejectReason() const noexcept {
    return last_trajectory_reject_reason_;
  }
  int PlannedMinimumClearanceSegment() const noexcept {
    return planned_min_clearance_segment_;
  }
  const std::array<double, 3>& PlannedMinimumClearancePosition() const noexcept {
    return planned_min_racket_position_b_m_;
  }
  bool IsPreparationPlan() const noexcept { return preparation_plan_; }
  bool IsReady() const noexcept { return prepared_; }
  // True between a successful live Strike commit and the end of its
  // follow-through; mirrored in CanAcceptExternalGoal() to lock new SetGoal
  // calls.
  bool CommittedStrikeActive() const noexcept {
    return committed_strike_active_;
  }
  // Sequence of the goal that produced the currently committed live strike,
  // or 0 if no live strike is committed.
  std::uint64_t CommittedExternalGoalSequence() const noexcept {
    return committed_external_goal_sequence_;
  }

 private:
  enum class SwingType {
    kForehand = 1,
    kBackhand = -1,
  };

  enum class Phase {
    kSolve,
    kApproach,
    kStrike,
    kFollowThrough,
    kHold,
  };

  struct RacketPose {
    std::array<double, 3> position_b_m{};
    std::array<double, 3> normal_b{0.0, 1.0, 0.0};
  };

  bool LoadGoal(const std::string& path, std::string& error);
  void QueueReadyPlan() noexcept;
  bool StartTarget(const robot_io::RobotState& state, double time_s) noexcept;
  void GenerateRandomTarget(SwingType swing) noexcept;
  bool ComputePose(const std::array<double, 31>& q_sdk,
                   RacketPose& pose) const;
  bool ComputeJacobian(const std::array<double, 31>& q_sdk,
                       Eigen::Matrix<double, 6, 7>& jacobian) const;
  bool Solve(const robot_io::RobotState& state, SwingType swing) noexcept;
  bool SolveFromSeed(const std::array<double, 31>& base_q,
                     const std::array<double, 7>& seed,
                     const std::array<double, 3>& target_position,
                     const std::array<double, 3>& target_normal,
                     std::array<double, 7>& solution,
                     double& position_error_m,
                     double& normal_error_deg,
                     int& iterations) const;
  bool ComputeStrikeVelocity(const std::array<double, 31>& base_q) noexcept;
  bool PrepareImpactTransition() noexcept;
  bool SolveFollowThrough(const std::array<double, 31>& base_q,
                          SwingType swing) noexcept;
  // The planner must never turn a reachable paddle pose into an artificial
  // "unreachable" rejection merely because the arm starts from a folded
  // posture.  This finds the shortest safe quintic approach duration under
  // the configured joint-speed envelope.
  bool FitTrajectoryDuration() noexcept;
  double EstimateTrackingMinimumStrikeTimeS() const noexcept;
  int TrajectorySegmentCount() const noexcept;
  double TrajectorySegmentDuration(int segment) const noexcept;
  void TrajectorySegmentBoundary(
      int segment, std::size_t joint, double& q0, double& dq0,
      double& q1, double& dq1) const noexcept;
  bool TrajectoryWithinLimits() const noexcept;
  bool TrajectoryHasBodyClearance() noexcept;
  void FillSafeHold(ArmTarget& target) const noexcept;
  void FillTarget(const robot_io::RobotState& state, double time_s,
                  ArmTarget& target) noexcept;
  void UpdateMeasuredError(const robot_io::RobotState& state) noexcept;
  void SnapshotImpactTelemetry(const robot_io::RobotState& state,
                               double time_s) noexcept;
  SwingType SelectSwingForPosition(
      const std::array<double, 3>& position) const noexcept;

  std::unique_ptr<RobotFK> fk_;
  int wrist_body_index_{-1};
  double update_hz_{50.0};
  double strike_time_s_{1.2};
  double requested_strike_time_s_{1.2};
  double external_min_strike_time_s_{0.0};
  double external_ready_time_s_{1.40};
  std::array<double, 3> external_ready_position_b_m_{0.48, 0.10, 0.33};
  std::array<double, 3> external_ready_normal_b_{-1.0, 0.0, 0.0};
  double follow_through_s_{0.30};
  double follow_through_distance_m_{0.12};
  double max_joint_velocity_rad_s_{6.0};
  bool random_targets_{false};
  bool require_external_goals_{false};
  bool automatic_ready_recovery_{true};
  bool have_external_goal_{false};
  bool alternate_swings_{true};
  double target_interval_s_{5.0};
  std::array<double, 3> forehand_position_min_{0.42, -0.72, -0.08};
  std::array<double, 3> forehand_position_max_{0.65, -0.38, 0.12};
  std::array<double, 3> backhand_position_min_{0.36, -0.05, -0.12};
  std::array<double, 3> backhand_position_max_{0.68, 0.40, 0.12};
  std::array<double, 3> forehand_velocity_b_mps_{1.8, 0.0, 0.8};
  std::array<double, 3> backhand_velocity_b_mps_{1.6, -0.05, 0.7};
  std::array<double, 3> random_velocity_jitter_mps_{0.20, 0.12, 0.15};
  double random_normal_yaw_rad_{0.20943951023931953};
  double forehand_normal_pitch_min_rad_{0.08726646259971647};
  double forehand_normal_pitch_max_rad_{0.5235987755982988};
  double backhand_normal_pitch_min_rad_{-0.3490658503988659};
  double backhand_normal_pitch_max_rad_{0.2617993877991494};
  double next_target_time_s_{0.0};
  double next_plan_retry_time_s_{0.0};
  std::mt19937_64 rng_{std::random_device{}()};
  std::array<double, 3> target_position_b_m_{0.48, -0.50, 0.02};
  std::array<double, 3> target_normal_b_{1.0, 0.0, 0.0};
  std::array<double, 3> target_velocity_b_mps_{1.8, 0.0, 0.8};
  SwingType requested_swing_{SwingType::kForehand};
  bool requested_swing_auto_{true};
  SwingType active_swing_{SwingType::kForehand};
  SwingType next_random_swing_{SwingType::kForehand};

  Phase phase_{Phase::kSolve};
  bool initialized_{false};
  bool goal_dirty_{true};
  bool strike_reported_{false};
  // Preparation is a one-time startup move, not a synthetic ball strike.
  // A real external command clears preparation_plan_ before replanning.
  bool preparation_plan_{false};
  bool prepared_{false};
  // True between a successful live Strike commit and the end of its
  // follow-through. While this flag is set, CanAcceptExternalGoal() refuses
  // new SetGoal calls so a 100 Hz prediction stream cannot keep rebuilding
  // the approach. Released when phase_ returns to kHold.
  bool committed_strike_active_{false};
  // Sequence of the most recent accepted external goal. Becomes the
  // committed sequence once StartTarget succeeds.
  std::uint64_t pending_external_goal_sequence_{0};
  // Sequence of the goal that produced the currently locked trajectory. Used
  // for telemetry and to detect "same goal, replanned" vs. "new ball,
  // rejected" once the lock lifts.
  std::uint64_t committed_external_goal_sequence_{0};
  double plan_origin_s_{0.0};
  double time_to_strike_s_{0.0};
  double tracking_minimum_strike_time_s_{0.0};
  std::array<double, 7> approach_from_q_{};
  std::array<double, 7> approach_from_dq_{};
  // Non-arm joints used when evaluating the sampled racket FK of a planned
  // trajectory. Right-arm entries are overwritten by each trajectory sample.
  std::array<double, 31> trajectory_base_q_{};
  std::array<double, 7> last_command_q_{};
  std::array<double, 7> last_command_dq_{};
  bool have_last_command_{false};
  // A rejected live target holds this fixed pose with normal arm gains.
  // Reissuing the measured angle every tick removes gravity support.
  std::array<double, 7> safe_hold_q_{};
  bool safe_hold_valid_{false};
  // Optional safety/visibility waypoint: lift the racket clear of the body
  // and move it toward the table before committing to the impact launch.
  // A short, terminal launch segment reaches the exact impact velocity.  The
  // preceding rest-to-rest approach reaches launch_q_ without demanding an
  // artificial speed overshoot from a single Hermite curve.
  std::array<double, 7> launch_q_{};
  double impact_transition_s_{0.24};
  std::array<double, 7> solved_q_{};
  std::array<double, 7> solved_dq_{};
  std::array<double, 7> follow_q_{};

  std::array<double, 3> last_racket_position_b_m_{};
  std::array<double, 3> achieved_strike_velocity_b_mps_{};
  double last_position_error_m_{0.0};
  double last_normal_error_deg_{0.0};
  double solved_position_error_m_{0.0};
  double solved_normal_error_deg_{0.0};
  int solve_iterations_{0};
  std::uint64_t strike_count_{0};
  std::uint64_t rejected_target_count_{0};
  std::uint64_t solve_reject_count_{0};
  std::uint64_t velocity_reject_count_{0};
  std::uint64_t follow_reject_count_{0};
  std::uint64_t trajectory_reject_count_{0};
  double last_velocity_solve_error_mps_{0.0};
  double last_normal_rate_rad_s_{0.0};
  std::array<double, 3> impact_measured_velocity_b_mps_{};
  double impact_position_error_m_{0.0};
  double impact_normal_error_deg_{0.0};
  double impact_timing_error_s_{0.0};
  double impact_source_deadline_error_s_{0.0};
  double impact_max_joint_error_rad_{0.0};
  std::int64_t last_source_deadline_ns_{0};
  double last_planning_duration_ms_{0.0};
  double planned_min_racket_body_clearance_m_{
      std::numeric_limits<double>::infinity()};
  const char* last_trajectory_reject_reason_{"none"};
  int planned_min_clearance_segment_{-1};
  std::array<double, 3> planned_min_racket_position_b_m_{};
};

}  // namespace a3_deploy::control
