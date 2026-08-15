#ifndef CALIBRATION_SPLIT_CALIBRATION_CSV_H
#define CALIBRATION_SPLIT_CALIBRATION_CSV_H

#include <string>
#include <vector>

#include <Eigen/Dense>

namespace calibration {

struct ChunkSummary {
  int chunk_id = 0;
  int rows = 0;
  double duration_s = 0.0;
  double scale_to_m = 1.0;
  double z_offset_m = 0.0;
  double x_min_m = 0.0, x_max_m = 0.0;
  double y_min_m = 0.0, y_max_m = 0.0;
  double z_min_m = 0.0, z_max_m = 0.0;
};

struct SplitOptions {
  std::string input;
  std::string output_dir;
  double gap_seconds = 1.0;
  std::string select;       // comma-separated ids; empty = all
  std::string prefix = "traj";
};

/// Infer scale factor: 0.001 if median abs coords > 10, else 1.0.
double inferScale(const std::vector<Eigen::Vector3d> & pos);

/// Split a single trajectory into chunks separated by gaps > gap_seconds.
std::vector<std::vector<Eigen::Vector3d>> splitRows(
  const std::vector<double> & times,
  const std::vector<Eigen::Vector3d> & pos,
  double gap_seconds);

/// Normalize one chunk: scale to meters, zero time at chunk start,
/// shift z so the low-z band is near 0.
struct NormalizedChunk {
  std::vector<double> times;
  std::vector<Eigen::Vector3d> pos;
  double z_offset_m = 0.0;
};
NormalizedChunk normalizeChunk(
  const std::vector<double> & times,
  const std::vector<Eigen::Vector3d> & pos,
  double scale_to_m);

/// Write one chunk to <output_dir>/<prefix>_chunkNN.csv.
void writeChunkCsv(
  const std::string & output_dir,
  const std::string & prefix,
  int chunk_id,
  const std::vector<double> & times,
  const std::vector<Eigen::Vector3d> & pos);

/// Top-level driver: split input CSV into per-chunk CSVs plus manifest.csv.
int splitCalibrationCsv(const SplitOptions & opts);

}  // namespace calibration

#endif  // CALIBRATION_SPLIT_CALIBRATION_CSV_H
