#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <json/json.h>
#include <optional>

#include "groupQueries.hpp"
#include "machineConstants.hpp"
#include "machineConstantsImplementation.hpp"
#include "utilities/systemInformation.hpp"

namespace adaptive {

void calculatePartitionMachineConstants() {
  std::cout << "Calculating machine constants for Partition_minRadixBits... ";
  double minimumRadixBits = log2(static_cast<double>(l2TlbEntriesFor4KbytePages()) / 4);
  int roundedMinimumRadixBits = static_cast<int>(std::floor(minimumRadixBits));
  MachineConstants::getInstance().updateMachineConstant("Partition_minRadixBits",
                                                        roundedMinimumRadixBits);
  std::cout << " Complete" << std::endl;

  std::cout << "Calculating machine constants for Partition_startRadixBits... ";
  int startRadixBits = roundedMinimumRadixBits + 8;
  MachineConstants::getInstance().updateMachineConstant("Partition_startRadixBits", startRadixBits);
  std::cout << " Complete" << std::endl;
}

MachineConstants& MachineConstants::getInstance() {
  static MachineConstants instance;
  return instance;
}

MachineConstants::MachineConstants() {
  char* constantsPath = std::getenv("HAZARD_ADAPTIVE_CONSTANTS");
  if(constantsPath != nullptr) {
    std::cout << "Environment variable set, will get constants from '" << constantsPath << "'\n";
    machineConstantsFilePath = constantsPath;
  } else {
    machineConstantsFilePath =
        getProjectRootDirectory() + "Source/constants/machineConstantValues.json";
    std::cout << "Environment variable for constants not set, will attempt to get constants from '"
              << machineConstantsFilePath << "'" << std::endl;
  }
  loadMachineConstants();
}

double MachineConstants::getMachineConstant(const std::string& key) const {
  if(machineConstants.count(key) == 0) {
    std::cout << "Machine constant for " << key << " not found. Exiting..." << std::endl;
    exit(1);
  }
  return machineConstants.at(key);
}

void MachineConstants::updateMachineConstant(const std::string& key, double value) {
  Json::Value jsonRoot;
  std::ifstream inputFile(machineConstantsFilePath);
  if(inputFile.is_open()) {
    inputFile >> jsonRoot;
    inputFile.close();
  } else {
    std::cerr << "Error opening file: " << machineConstantsFilePath << std::endl;
  }
  jsonRoot[key] = Json::Value(value);

  std::ofstream outputFile(machineConstantsFilePath);
  if(outputFile.is_open()) {
    outputFile << jsonRoot;
    outputFile.close();
  } else {
    std::cerr << "Error opening file: " << machineConstantsFilePath << std::endl;
  }

  loadMachineConstants();
}

void MachineConstants::loadMachineConstants() {
  machineConstants.clear();

  std::ifstream file(machineConstantsFilePath);
  if(file.is_open()) {
    Json::Value jsonRoot;
    file >> jsonRoot;
    file.close();

    for(const auto& key : jsonRoot.getMemberNames()) {
      machineConstants[key] = jsonRoot[key].asDouble();
    }
  } else {
    writeEmptyFile();
    machineConstants = {};
  }
}

void MachineConstants::writeEmptyFile() {
  std::ofstream file(machineConstantsFilePath, std::ofstream::out | std::ofstream::trunc);
  if(file.is_open()) {
    file << "{}";
    file.close();
    std::filesystem::perms permissions =
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write |
        std::filesystem::perms::group_read | std::filesystem::perms::group_write |
        std::filesystem::perms::others_read | std::filesystem::perms::others_write;
    std::filesystem::permissions(machineConstantsFilePath, permissions);
  } else {
    std::cerr << "Error creating file: " << machineConstantsFilePath << std::endl;
  }
}

void MachineConstants::calculateMissingMachineConstants() {
  // Temporarily change number of threads in thread pool to the maximum value so that
  // AdaptiveParallel can be used to calculate any missing machine constants
  auto nonVectorizedDOPvalue = adaptive::config::nonVectorizedDOP;
  adaptive::config::nonVectorizedDOP = static_cast<int32_t>(adaptive::logicalCoresCount());
  ThreadPool::getInstance(std::nullopt).changeNumThreads(adaptive::logicalCoresCount());

  // Update eventSet in main thread to include CPU cycles counters
  auto& eventSet = getThreadEventSet();
  switchEventSetToCycles(eventSet);

  // Update eventSets in worker threads to include CPU cycles counters
  auto& threadPool = ThreadPool::getInstance(std::nullopt);
  auto& synchroniser = Synchroniser::getInstance();
  for(uint32_t threadNum = 0; threadNum < adaptive::logicalCoresCount(); ++threadNum) {
    threadPool.enqueue([&synchroniser] {
      auto& eventSet = getThreadEventSet();
      switchEventSetToCycles(eventSet);
      synchroniser.taskComplete();
    });
  }
  synchroniser.waitUntilComplete(static_cast<int>(adaptive::logicalCoresCount()));

  uint32_t dop = 1;
  while(dop <= logicalCoresCount()) {
    std::string dopStr = std::to_string(dop);

    if(machineConstants.count("SelectLower_4B_elements_" + dopStr + "_dop") == 0 ||
       machineConstants.count("SelectUpper_4B_elements_" + dopStr + "_dop") == 0) {
      std::cout << "Machine constant for Select (4B elements, DOP=" + dopStr +
                       ") does not exist. Calculating now (this may take a while)."
                << std::endl;
      calculateSelectMachineConstants<int32_t>(dop);
    }

    if(machineConstants.count("SelectLower_8B_elements_" + dopStr + "_dop") == 0 ||
       machineConstants.count("SelectUpper_8B_elements_" + dopStr + "_dop") == 0) {
      std::cout << "Machine constant for Select (8B elements, DOP=" + dopStr +
                       ") does not exist. Calculating now (this may take a while)."
                << std::endl;
      calculateSelectMachineConstants<int64_t>(dop);
    }

    for(int groupQueryIdx = 1; groupQueryIdx <= 11; ++groupQueryIdx) {
      auto names = getGroupMachineConstantNames(static_cast<GROUP_QUERIES>(groupQueryIdx), dop);
      if(machineConstants.count(names.pageFaultDecreaseRate) == 0 ||
         machineConstants.count(names.llcMissRate) == 0) {
        uint32_t numBytes = 4 + (4 * groupQueryIdx);
        std::cout << "Machine constant for Group (" << std::to_string(numBytes)
                  << "Bytes, DOP=" + dopStr +
                         ") does not exist. Calculating now (this may take a while)."
                  << std::endl;
        calculateGroupMachineConstants(static_cast<GROUP_QUERIES>(groupQueryIdx),
                                       static_cast<int>(dop));
      }
    }

    if(dop == logicalCoresCount()) {
      break;
    }
    dop = (dop * 2) <= logicalCoresCount() ? dop * 2 : logicalCoresCount();
  }

  if(machineConstants.count("Partition_minRadixBits") == 0 ||
     machineConstants.count("Partition_startRadixBits") == 0) {
    calculatePartitionMachineConstants();
  }

  // Revert eventSet in main thread to the default
  switchEventSetToPartition(eventSet);

  // Revert number of threads in thread pool to that set in the config
  // Additionally, by recreating all the worker threads, we revert the eventSets to the default
  adaptive::config::nonVectorizedDOP = nonVectorizedDOPvalue;
  ThreadPool::getInstance(std::nullopt).changeNumThreads(adaptive::config::nonVectorizedDOP);
}

} // namespace adaptive
