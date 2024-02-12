#include <cstdlib>
#include <fstream>
#include <iostream>
#include <json/json.h>

#include "machineConstants.hpp"
#include "utilities/systemInformation.hpp"

namespace adaptive {

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
  calculateMissingMachineConstants();
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
      std::cout << "Machine constant for Select (4B elements, constantsDOP=" + dopStr +
                       ") does not exist. Calculating now (this may take a while)."
                << std::endl;
      calculateSelectMachineConstants<int32_t>(dop);
    }

    if(machineConstants.count("SelectLower_8B_elements_" + dopStr + "_dop") == 0 ||
       machineConstants.count("SelectUpper_8B_elements_" + dopStr + "_dop") == 0) {
      std::cout << "Machine constant for Select (8B elements, constantsDOP=" + dopStr +
                       ") does not exist. Calculating now (this may take a while)."
                << std::endl;
      calculateSelectMachineConstants<int64_t>(dop);
    }

    if(dop == logicalCoresCount()) {
      break;
    }
    dop = (dop * 2) <= logicalCoresCount() ? dop * 2 : logicalCoresCount();
  }
}

} // namespace adaptive
