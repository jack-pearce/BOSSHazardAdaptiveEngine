#ifndef BOSSHAZARDADAPTIVEENGINE_PARTITION_CPP
#define BOSSHAZARDADAPTIVEENGINE_PARTITION_CPP

#include <iostream>

#include "constants/machineConstants.hpp"
#include "partition.hpp"

namespace adaptive {

std::string getPartitionName(PartitionOperators partitionImplementation) {
  if(partitionImplementation == PartitionOperators::RadixBitsFixedMin) {
    std::string name = "Partition_minRadixBits";
    auto radixBitsMin = static_cast<int>(MachineConstants::getInstance().getMachineConstant(name));
    return "RadixPartition_Fixed_" + std::to_string(radixBitsMin) + "Bits";
  } else if(partitionImplementation == PartitionOperators::RadixBitsFixedMax) {
    std::string name = "Partition_startRadixBits";
    auto radixBitsMax = static_cast<int>(MachineConstants::getInstance().getMachineConstant(name));
    return "RadixPartition_Fixed_" + std::to_string(radixBitsMax) + "Bits";
  } else if(partitionImplementation == PartitionOperators::RadixBitsAdaptive ||
            partitionImplementation == PartitionOperators::RadixBitsAdaptiveParallel) {
    return "RadixPartition_Adaptive";
  } else {
    throw std::runtime_error("Invalid selection of 'Partition' implementation!");
  }
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_PARTITION_CPP