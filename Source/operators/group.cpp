#ifndef BOSSHAZARDADAPTIVEENGINE_GROUP_CPP
#define BOSSHAZARDADAPTIVEENGINE_GROUP_CPP

#include "group.hpp"

namespace adaptive {

std::string getGroupName(Group groupImplementation) {
  switch (groupImplementation) {
  case Group::Hash:
    return "Group_Hash";
  case Group::Sort:
    return "Group_Sort";
  case Group::GroupAdaptive:
    return "Group_Adaptive";
  case Group::GroupAdaptiveParallel:
    return "Group_Adaptive_Parallel";
  default:
    throw std::runtime_error("Invalid selection of 'Group' implementation!");
  }
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_GROUP_CPP
