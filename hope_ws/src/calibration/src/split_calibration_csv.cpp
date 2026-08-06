#include "split_calibration_csv.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace calibration {

namespace {

double median(std::vector<double> v) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n % 2 == 0) {
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }
  return v[n / 2];
}

std::set<int> parseSelectedIds(const std::string & text) {
  std::set<int> out;
  if (text.empty()) {
    return out;
  }
  std::stringstream ss(text);
  std::string part;
  while (std::getline(ss, part, ',')) {
    auto first = part.find_first_not_of(" \t");
    auto last = part.find_last_not_of(" \t");
    if (first == std::string::npos) {
      continue;
    }
    out.insert(std::stoi(part.substr(first, last - first + 1)));
  }
  return out;
}

std::vector<std::pair<double, Eigen::Vector3d>> loadCsv(const std::string & path) {
  std::ifstream fh(path);
  if (!fh) {
    throw std::runtime_error("Cannot open CSV: " + path);
  }
  std::vector<std::pair<double, Eigen::Vector3d>> rows;
  std::string line;
  while (std::getline(fh, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream ss(line);
    std::string tok;
    std::vector<std::string> cols;
    while (std::getline(ss, tok, ',')) {
      cols.push_back(tok);
    }
    if (cols.size() < 4) {
      continue;
    }
    std::string first = cols[0];
    std::transform(first.begin(), first.end(), first.begin(),
      [](unsigned char c) { return std::tolower(c); });
    if (first == "t" || first == "time") {
      continue;
    }
    double t = std::stod(cols[0]);
    Eigen::Vector3d p(std::stod(cols[1]), std::stod(cols[2]), std::stod(cols[3]));
    rows.emplace_back(t, p);
  }
  if (rows.empty()) {
    throw std::runtime_error("No numeric rows found in CSV: " + path);
  }
  return rows;
}

void writeManifest(
  const std::string & path,
  const std::vector<ChunkSummary> & manifests)
{
  std::ofstream mf(path);
  if (!mf) {
    throw std::runtime_error("Cannot write manifest.csv: " + path);
  }
  mf << "chunk_id,rows,duration_s,scale_to_m,z_offset_m,"
        "x_min_m,x_max_m,y_min_m,y_max_m,z_min_m,z_max_m\n";
  mf << std::fixed << std::setprecision(6);
  for (const auto & s : manifests) {
    mf << s.chunk_id << "," << s.rows << "," << s.duration_s << ","
       << s.scale_to_m << "," << s.z_offset_m << ","
       << s.x_min_m << "," << s.x_max_m << ","
       << s.y_min_m << "," << s.y_max_m << ","
       << s.z_min_m << "," << s.z_max_m << "\n";
  }
}

ChunkSummary summarize(
  int chunk_id,
  const std::vector<double> & times,
  const std::vector<Eigen::Vector3d> & pos,
  double scale_to_m,
  double z_offset_m)
{
  ChunkSummary s;
  s.chunk_id = chunk_id;
  s.rows = static_cast<int>(times.size());
  s.duration_s = (times.size() > 1) ? times.back() : 0.0;
  s.scale_to_m = scale_to_m;
  s.z_offset_m = z_offset_m;
  if (!pos.empty()) {
    s.x_min_m = pos[0].x(); s.x_max_m = pos[0].x();
    s.y_min_m = pos[0].y(); s.y_max_m = pos[0].y();
    s.z_min_m = pos[0].z(); s.z_max_m = pos[0].z();
    for (const auto & p : pos) {
      s.x_min_m = std::min(s.x_min_m, p.x()); s.x_max_m = std::max(s.x_max_m, p.x());
      s.y_min_m = std::min(s.y_min_m, p.y()); s.y_max_m = std::max(s.y_max_m, p.y());
      s.z_min_m = std::min(s.z_min_m, p.z()); s.z_max_m = std::max(s.z_max_m, p.z());
    }
  }
  return s;
}

}  // namespace

double inferScale(const std::vector<Eigen::Vector3d> & pos) {
  if (pos.empty()) {
    return 1.0;
  }
  std::vector<double> coords;
  coords.reserve(pos.size() * 3);
  for (const auto & p : pos) {
    coords.push_back(std::abs(p.x()));
    coords.push_back(std::abs(p.y()));
    coords.push_back(std::abs(p.z()));
  }
  double median_abs = median(coords);
  return (median_abs > 10.0) ? 0.001 : 1.0;
}

std::vector<std::vector<Eigen::Vector3d>> splitRows(
  const std::vector<double> & times,
  const std::vector<Eigen::Vector3d> & pos,
  double gap_seconds)
{
  std::vector<std::vector<Eigen::Vector3d>> chunks;
  if (times.size() <= 1) {
    chunks.push_back(pos);
    return chunks;
  }
  size_t start = 0;
  for (size_t i = 0; i + 1 < times.size(); ++i) {
    if (times[i + 1] - times[i] > gap_seconds) {
      std::vector<Eigen::Vector3d> chunk(pos.begin() + start, pos.begin() + i + 1);
      chunks.push_back(chunk);
      start = i + 1;
    }
  }
  std::vector<Eigen::Vector3d> last(pos.begin() + start, pos.end());
  chunks.push_back(last);
  return chunks;
}

NormalizedChunk normalizeChunk(
  const std::vector<double> & times,
  const std::vector<Eigen::Vector3d> & pos,
  double scale_to_m)
{
  NormalizedChunk out;
  if (times.empty()) {
    return out;
  }
  out.times.reserve(times.size());
  out.pos.reserve(pos.size());
  double t0 = times.front();
  for (size_t i = 0; i < times.size(); ++i) {
    out.times.push_back(times[i] - t0);
    out.pos.push_back(pos[i] * scale_to_m);
  }
  std::vector<double> zs;
  zs.reserve(out.pos.size());
  for (const auto & p : out.pos) {
    zs.push_back(p.z());
  }
  std::sort(zs.begin(), zs.end());
  size_t low_count = std::max<size_t>(3, zs.size() / 10);
  double z_offset = 0.0;
  for (size_t i = 0; i < low_count; ++i) {
    z_offset += zs[i];
  }
  z_offset /= static_cast<double>(low_count);
  out.z_offset_m = z_offset;
  for (auto & p : out.pos) {
    p.z() -= z_offset;
  }
  return out;
}

void writeChunkCsv(
  const std::string & output_dir,
  const std::string & prefix,
  int chunk_id,
  const std::vector<double> & times,
  const std::vector<Eigen::Vector3d> & pos)
{
  char fname[512];
  std::snprintf(fname, sizeof(fname), "%s/%s_chunk%02d.csv",
    output_dir.c_str(), prefix.c_str(), chunk_id);
  std::ofstream fh(fname);
  if (!fh) {
    throw std::runtime_error("Cannot write chunk CSV: " + std::string(fname));
  }
  fh << "t,x,y,z\n";
  fh << std::fixed << std::setprecision(9);
  for (size_t i = 0; i < times.size(); ++i) {
    fh << times[i] << "," << pos[i].x() << "," << pos[i].y() << "," << pos[i].z() << "\n";
  }
}

int splitCalibrationCsv(const SplitOptions & opts) {
  auto rows = loadCsv(opts.input);
  std::vector<double> times;
  std::vector<Eigen::Vector3d> pos;
  times.reserve(rows.size());
  pos.reserve(rows.size());
  for (const auto & [t, p] : rows) {
    times.push_back(t);
    pos.push_back(p);
  }
  double scale = inferScale(pos);
  std::set<int> selected = parseSelectedIds(opts.select);

  auto chunks = splitRows(times, pos, opts.gap_seconds);
  std::vector<ChunkSummary> manifests;
  int written = 0;
  for (size_t i = 0; i < chunks.size(); ++i) {
    int chunk_id = static_cast<int>(i + 1);
    auto norm = normalizeChunk(times, pos, scale);
    auto sum = summarize(chunk_id, norm.times, norm.pos, scale, norm.z_offset_m);
    manifests.push_back(sum);
    if (!selected.empty() && selected.count(chunk_id) == 0) {
      continue;
    }
    writeChunkCsv(opts.output_dir, opts.prefix, chunk_id, norm.times, norm.pos);
    written++;
  }

  writeManifest(opts.output_dir + "/manifest.csv", manifests);
  std::cout << "Input rows: " << rows.size() << "\n";
  std::cout << "Inferred scale_to_m: " << scale << "\n";
  std::cout << "Detected chunks: " << manifests.size() << "\n";
  std::cout << "Wrote chunk CSVs: " << written << "\n";
  std::cout << "Manifest: " << opts.output_dir << "/manifest.csv\n";
  return 0;
}

}  // namespace calibration
