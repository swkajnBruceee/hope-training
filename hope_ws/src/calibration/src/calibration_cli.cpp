#include <cstdio>
#include <string>
#include <vector>

#include "calibration.h"

int main(int argc, char ** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <csv> [<csv> ...]\n", argv[0]);
    return 1;
  }
  std::vector<std::vector<double>> timestamps;
  std::vector<std::vector<Eigen::Vector3d>> trajectories;
  for (int i = 1; i < argc; ++i) {
    auto csv = calibration::loadTrajectoryCsv(argv[i]);
    timestamps.push_back(csv.t);
    trajectories.push_back(csv.p);
  }
  auto phys = calibration::calibrateBallPhysics(timestamps, trajectories);
  std::printf("Calibrated ball physics:\n");
  std::printf("  drag_k        = %.4f\n", phys.k);
  std::printf("  restitution_h = %.4f\n", phys.C_h);
  std::printf("  restitution_v = %.4f\n", phys.C_v);
  std::printf("\nPaste into config/solver.yaml:\n");
  std::printf("    drag_k: %.4f\n", phys.k);
  std::printf("    restitution_h: %.4f\n", phys.C_h);
  std::printf("    restitution_v: %.4f\n", phys.C_v);
  return 0;
}
