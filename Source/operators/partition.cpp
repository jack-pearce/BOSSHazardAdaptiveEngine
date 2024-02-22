#ifndef BOSSHAZARDADAPTIVEENGINE_PARTITION_CPP
#define BOSSHAZARDADAPTIVEENGINE_PARTITION_CPP

#include <iostream>

#include "partition.hpp"

namespace adaptive {

std::string getPartitionName(PartitionOperators partitionImplementation) {
  switch(partitionImplementation) {
  case PartitionOperators::RadixBitsFixed:
    return "RadixPartition_Static";
  case PartitionOperators::RadixBitsAdaptive:
    return "RadixPartition_Adaptive";
  default:
    throw std::runtime_error("Invalid selection of 'Partition' implementation!");
  }
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_PARTITION_CPP