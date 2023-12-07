#ifndef BOSSHAZARDADAPTIVEENGINE_MACHINECONSTANTS_HPP
#define BOSSHAZARDADAPTIVEENGINE_MACHINECONSTANTS_HPP

#include <string>
#include <map>


namespace adaptive {

void printMachineConstants();
void calculateMissingMachineConstants();
void clearAndRecalculateMachineConstants();

class MachineConstants {
public:
    static MachineConstants& getInstance();
    MachineConstants(const MachineConstants&) = delete;
    void operator=(const MachineConstants&) = delete;

    double getMachineConstant(std::string &key);
    void updateMachineConstant(const std::string& key, double value);
    void printMachineConstants();
    void calculateMissingMachineConstants();
    void clearAndRecalculateMachineConstants();

private:
    MachineConstants();
    ~MachineConstants() = default;
    void loadMachineConstants();
    void writeEmptyFile();

    std::map<std::string, double> machineConstants;
    std::string machineConstantsFilePath;
};

}

#include "machineConstantsImplementation.hpp"

#endif //BOSSHAZARDADAPTIVEENGINE_MACHINECONSTANTS_HPP
