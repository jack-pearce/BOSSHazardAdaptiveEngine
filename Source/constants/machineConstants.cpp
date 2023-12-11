#include <fstream>
#include <iostream>
#include <json/json.h>

#include "machineConstants.hpp"
#include "utilities/systemInformation.hpp"

namespace adaptive {

void calculateMissingMachineConstants() {
  MachineConstants::getInstance().calculateMissingMachineConstants();
}

void clearAndRecalculateMachineConstants() {
  MachineConstants::getInstance().clearAndRecalculateMachineConstants();
}

void printMachineConstants() { MachineConstants::getInstance().printMachineConstants(); }

MachineConstants& MachineConstants::getInstance() {
  static MachineConstants instance;
  return instance;
}

MachineConstants::MachineConstants() {
  machineConstantsFilePath =
      getProjectRootDirectory() + "/Source/constants/machineConstantValues.json";
  loadMachineConstants();
}

double MachineConstants::getMachineConstant(std::string& key) {
  if(machineConstants.count(key) == 0) {
    std::cout << "Machine constant for " << key
              << " does not exist, calculating all missing values now. "
                 "This may take a while."
              << std::endl;
    calculateMissingMachineConstants();
  }
  return machineConstants[key];
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

  jsonRoot[key] = value;

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
  } else {
    std::cerr << "Error opening file: " << machineConstantsFilePath << std::endl;
  }
}

void MachineConstants::calculateMissingMachineConstants() {
  uint32_t dop = 1;
  while(dop <= logicalCoresCount()) {
    std::string dopStr = std::to_string(dop);

    if(machineConstants.count("SelectLower_4B_elements_" + dopStr + "_dop") == 0 ||
       machineConstants.count("SelectUpper_4B_elements_" + dopStr + "_dop") == 0) {
      calculateSelectMachineConstants<int32_t>(dop);
    }
    if(machineConstants.count("SelectLower_8B_elements_" + dopStr + "_dop") == 0 ||
       machineConstants.count("SelectUpper_8B_elements_" + dopStr + "_dop") == 0) {
      calculateSelectMachineConstants<int64_t>(dop);
    }

    dop *= 2;
  }
}

void MachineConstants::clearAndRecalculateMachineConstants() {
  writeEmptyFile();
  loadMachineConstants();
  calculateMissingMachineConstants();
}

void MachineConstants::printMachineConstants() {
  std::cout << "Machine Constants:" << std::endl;
  for(const auto& machineConstant : machineConstants) {
    std::cout << "Constant: '" << machineConstant.first << "', Value: " << machineConstant.second
              << std::endl;
  }
}

} // namespace adaptive
