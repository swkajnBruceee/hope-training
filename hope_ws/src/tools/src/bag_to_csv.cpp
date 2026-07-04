#include "bag_to_csv.h"

#include <stdexcept>

namespace tools {

int bagPointTopicToCsv(
  const std::string & /*bag_path*/,
  const std::string & /*topic*/,
  const std::string & /*output_csv*/)
{
  // TODO: implement rosbag2_cpp-based reader.
  throw std::runtime_error("bag_to_csv: C++ migration pending; use the legacy Python helper for now");
}

}  // namespace tools
