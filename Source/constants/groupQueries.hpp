#ifndef BOSSHAZARDADAPTIVEENGINE_GROUPQUERIES_H
#define BOSSHAZARDADAPTIVEENGINE_GROUPQUERIES_H

#include <string>

namespace adaptive {

enum GROUP_QUERIES {
  Bytes_8 = 1,
  Bytes_12,
  Bytes_16,
  Bytes_20,
  Bytes_24,
  Bytes_28,
  Bytes_32,
  Bytes_36,
  Bytes_40,
  Bytes_44,
  Bytes_48 // = 11
};

struct GroupConstants {
  std::string tlbMissRate;
  std::string llcMissRate;
};

GroupConstants getGroupMachineConstantNames(GROUP_QUERIES groupQuery, uint32_t dop) {
  int groupQueryIndex = static_cast<int>(groupQuery);
  uint32_t numBytes = 4 + (4 * groupQueryIndex);
  std::string name1 = "Group_" + std::to_string(numBytes) + "B_" + std::to_string(dop) + "_dop_TLB";
  std::string name2 = "Group_" + std::to_string(numBytes) + "B_" + std::to_string(dop) + "_dop_LLC";
  return {name1, name2};
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_GROUPQUERIES_H
