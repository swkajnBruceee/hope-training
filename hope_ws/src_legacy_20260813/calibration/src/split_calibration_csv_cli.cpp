#include <cstdlib>
#include <iostream>
#include <string>

#include "split_calibration_csv.h"

int main(int argc, char ** argv) {
  calibration::SplitOptions opts;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto eq = a.find('=');
    if (eq == std::string::npos) {
      std::cerr << "Unrecognized arg: " << a << "\n";
      return 1;
    }
    std::string k = a.substr(0, eq);
    std::string v = a.substr(eq + 1);
    if (k == "--input") {
      opts.input = v;
    } else if (k == "--output-dir") {
      opts.output_dir = v;
    } else if (k == "--gap-seconds") {
      opts.gap_seconds = std::stod(v);
    } else if (k == "--select") {
      opts.select = v;
    } else if (k == "--prefix") {
      opts.prefix = v;
    } else {
      std::cerr << "Unknown option: " << k << "\n";
      return 1;
    }
  }
  if (opts.input.empty() || opts.output_dir.empty()) {
    std::cerr << "Usage: " << argv[0] << " --input=<csv> --output-dir=<dir> "
              << "[--gap-seconds=1.0] [--select=1,2,3] [--prefix=traj]\n";
    return 1;
  }
  return calibration::splitCalibrationCsv(opts);
}
