#ifndef TOOLS_BAG_TO_CSV_H
#define TOOLS_BAG_TO_CSV_H

#include <string>

namespace tools {

/**
 * Convert a rosbag2 topic stream into HOPE calibration CSV (t,x,y,z).
 *
 * TODO: full rosbag2_cpp-based reader migration. The original Python
 * implementation used rclpy.serialization + rosbag2_py to deserialize
 * PointStamped messages; the C++ port will mirror that behavior.
 *
 * For now this is a stub header so the package layout is consistent.
 */
int bagPointTopicToCsv(const std::string & bag_path, const std::string & topic,
                       const std::string & output_csv);

}  // namespace tools

#endif  // TOOLS_BAG_TO_CSV_H
