#ifndef BOSSHAZARDADAPTIVEENGINE_MACHINECONSTANTS_HPP
#define BOSSHAZARDADAPTIVEENGINE_MACHINECONSTANTS_HPP

#include <map>
#include <string>

namespace adaptive {

class MachineConstants {
public:
  static MachineConstants& getInstance();
  MachineConstants(const MachineConstants&) = delete;
  void operator=(const MachineConstants&) = delete;
  MachineConstants(MachineConstants&&) = delete;
  MachineConstants& operator=(MachineConstants&&) = delete;

  [[nodiscard]] double getMachineConstant(const std::string& key) const;
  void updateMachineConstant(const std::string& key, double value);
  void calculateMissingMachineConstants();

private:
  MachineConstants();
  ~MachineConstants() = default;
  void loadMachineConstants();
  void writeEmptyFile();

  std::map<std::string, double> machineConstants;
  std::string machineConstantsFilePath;
};

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_MACHINECONSTANTS_HPP
