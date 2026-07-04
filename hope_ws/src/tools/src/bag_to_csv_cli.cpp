#include <cstdio>
#include <string>

#include "bag_to_csv.h"

int main(int argc, char ** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "Usage: %s --bag <dir> --topic <name> --output <csv>\n", argv[0]);
    return 1;
  }
  std::string bag, topic, output;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    std::string v = argv[i + 1];
    if (k == "--bag") bag = v;
    else if (k == "--topic") topic = v;
    else if (k == "--output") output = v;
  }
  if (bag.empty() || topic.empty() || output.empty()) {
    std::fprintf(stderr, "Missing --bag/--topic/--output\n");
    return 1;
  }
  try {
    return tools::bagPointTopicToCsv(bag, topic, output);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
}
