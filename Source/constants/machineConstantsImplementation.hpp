#ifndef BOSSHAZARDADAPTIVEENGINE_MACHINECONSTANTSIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_MACHINECONSTANTSIMPLEMENTATION_HPP

#include <iostream>

#include "operators/select.hpp"
#include "utilities/dataGeneration.hpp"
#include "utilities/papiWrapper.hpp"

namespace adaptive {

constexpr size_t SELECT_DATA_SIZE = 250 * 1000 * 1000;
constexpr int NUMBER_OF_TESTS = 9;
constexpr int SELECT_ITERATIONS = 12;

template <typename T> T getThreshold(uint32_t n, double selectivity) {
  return static_cast<T>(n * selectivity);
}

template <typename T> double calculateSelectLowerMachineConstant(uint32_t dop) {
  auto data = generateRandomisedUniqueValuesInMemory<T>(SELECT_DATA_SIZE);

  auto predicate = std::greater();

  auto eventSet = PAPI_eventSet({"PERF_COUNT_HW_CPU_CYCLES"});
  long_long branchCycles, predicationCycles;
  double upperSelectivity = 0.5;
  double lowerSelectivity = 0;
  double midSelectivity;

  std::string machineConstantLowerName =
      "SelectLower_" + std::to_string(sizeof(T)) + "B_elements_" + std::to_string(dop) + "_dop";
  std::string machineConstantUpperName =
      "SelectUpper_" + std::to_string(sizeof(T)) + "B_elements_" + std::to_string(dop) + "_dop";

  for(auto i = 0; i < SELECT_ITERATIONS; ++i) {

    midSelectivity = (lowerSelectivity + upperSelectivity) / 2;

    {
      std::vector<T> column1(SELECT_DATA_SIZE);
      copyVector(data, column1);
      Span<T> columnSpan1 = Span<T>(std::move(std::vector(column1)));

      if(dop == 1) {
        eventSet.readCounters();
        select(Select::Branch, columnSpan1, 1 + getThreshold<T>(SELECT_DATA_SIZE, midSelectivity),
               false, predicate, {});
        eventSet.readCountersAndUpdateDiff();
        branchCycles = *eventSet.getCounterDiffsPtr();
      } else {
        MachineConstants::getInstance().updateMachineConstant(machineConstantLowerName, 1);
        MachineConstants::getInstance().updateMachineConstant(machineConstantUpperName, 0);
        branchCycles = PAPI_get_real_usec();
        select(Select::AdaptiveParallel, columnSpan1,
               1 + getThreshold<T>(SELECT_DATA_SIZE, midSelectivity), false, predicate, {}, dop,
               nullptr, true);
        branchCycles = PAPI_get_real_usec() - branchCycles;
      }
    }

    {
      std::vector<T> column2(SELECT_DATA_SIZE);
      copyVector(data, column2);
      Span<T> columnSpan2 = Span<T>(std::move(std::vector(column2)));

      if(dop == 1) {
        eventSet.readCounters();
        select(Select::Predication, columnSpan2,
               1 + getThreshold<T>(SELECT_DATA_SIZE, midSelectivity), false, predicate, {});
        eventSet.readCountersAndUpdateDiff();
        predicationCycles = *eventSet.getCounterDiffsPtr();
      } else {
        MachineConstants::getInstance().updateMachineConstant(machineConstantLowerName, 0);
        MachineConstants::getInstance().updateMachineConstant(machineConstantUpperName, 1);
        predicationCycles = PAPI_get_real_usec();
        select(Select::AdaptiveParallel, columnSpan2,
               1 + getThreshold<T>(SELECT_DATA_SIZE, midSelectivity), false, predicate, {}, dop,
               nullptr, true);
        predicationCycles = PAPI_get_real_usec() - predicationCycles;
      }
    }

    if(branchCycles > predicationCycles) {
      upperSelectivity = midSelectivity;
    } else {
      lowerSelectivity = midSelectivity;
    }

#ifdef DEBUG
    std::cout << "Selectivity: " << (lowerSelectivity + upperSelectivity) / 2
              << ", branch cycles: " << branchCycles
              << ", predication cycles: " << predicationCycles << std::endl;
#endif
  }

  std::cout << ".";
  std::cout.flush();

  return (upperSelectivity + lowerSelectivity) / 2;
}

template <typename T> double calculateSelectUpperMachineConstant(uint32_t dop) {
  auto data = generateRandomisedUniqueValuesInMemory<T>(SELECT_DATA_SIZE);

  auto predicate = std::greater();

  auto eventSet = PAPI_eventSet({"PERF_COUNT_HW_CPU_CYCLES"});
  long_long branchCycles, predicationCycles;
  double upperSelectivity = 1.0;
  double lowerSelectivity = 0.5;
  double midSelectivity;

  std::string machineConstantLowerName =
      "SelectLower_" + std::to_string(sizeof(T)) + "B_elements_" + std::to_string(dop) + "_dop";
  std::string machineConstantUpperName =
      "SelectUpper_" + std::to_string(sizeof(T)) + "B_elements_" + std::to_string(dop) + "_dop";

  for(auto i = 0; i < SELECT_ITERATIONS; ++i) {

    midSelectivity = (lowerSelectivity + upperSelectivity) / 2;

    {
      std::vector<T> column1(SELECT_DATA_SIZE);
      copyVector(data, column1);
      Span<T> columnSpan1 = Span<T>(std::move(std::vector(column1)));

      if(dop == 1) {
        eventSet.readCounters();
        select(Select::Branch, columnSpan1, 1 + getThreshold<T>(SELECT_DATA_SIZE, midSelectivity),
               false, predicate, {});
        eventSet.readCountersAndUpdateDiff();
        branchCycles = *eventSet.getCounterDiffsPtr();
      } else {
        MachineConstants::getInstance().updateMachineConstant(machineConstantLowerName, 1);
        MachineConstants::getInstance().updateMachineConstant(machineConstantUpperName, 0);
        branchCycles = PAPI_get_real_usec();
        select(Select::AdaptiveParallel, columnSpan1,
               1 + getThreshold<T>(SELECT_DATA_SIZE, midSelectivity), false, predicate, {}, dop,
               nullptr, true);
        branchCycles = PAPI_get_real_usec() - branchCycles;
      }
    }

    {
      std::vector<T> column2(SELECT_DATA_SIZE);
      copyVector(data, column2);
      Span<T> columnSpan2 = Span<T>(std::move(std::vector(column2)));

      if(dop == 1) {
        eventSet.readCounters();
        select(Select::Predication, columnSpan2,
               1 + getThreshold<T>(SELECT_DATA_SIZE, midSelectivity), false, predicate, {});
        eventSet.readCountersAndUpdateDiff();
        predicationCycles = *eventSet.getCounterDiffsPtr();
      } else {
        MachineConstants::getInstance().updateMachineConstant(machineConstantLowerName, 0);
        MachineConstants::getInstance().updateMachineConstant(machineConstantUpperName, 1);
        predicationCycles = PAPI_get_real_usec();
        select(Select::AdaptiveParallel, columnSpan2,
               1 + getThreshold<T>(SELECT_DATA_SIZE, midSelectivity), false, predicate, {}, dop,
               nullptr, true);
        predicationCycles = PAPI_get_real_usec() - predicationCycles;
      }
    }

    if(branchCycles > predicationCycles) {
      lowerSelectivity = midSelectivity;
    } else {
      upperSelectivity = midSelectivity;
    }

#ifdef DEBUG
    std::cout << "Selectivity: " << (lowerSelectivity + upperSelectivity) / 2
              << ", branch cycles: " << branchCycles
              << ", predication cycles: " << predicationCycles << std::endl;
#endif
  }

  std::cout << ".";
  std::cout.flush();

  return (upperSelectivity + lowerSelectivity) / 2;
}

template <typename T> void calculateSelectMachineConstants(uint32_t dop) {
  std::cout << "Calculating machine constants for Select_" << sizeof(T) << "B_elements_"
            << std::to_string(dop) << "_dop" << std::endl;
  std::cout << " - Running tests for lower crossover point";

  std::vector<double> lowerCrossoverPoints;
  for(auto i = 0; i < NUMBER_OF_TESTS; ++i) {
    lowerCrossoverPoints.push_back(calculateSelectLowerMachineConstant<T>(dop));
  }
  std::sort(lowerCrossoverPoints.begin(), lowerCrossoverPoints.end());
  std::cout << " Complete" << std::endl;

  std::cout << " - Running tests for upper crossover point";
  std::vector<double> upperCrossoverPoints;
  for(auto i = 0; i < NUMBER_OF_TESTS; ++i) {
    upperCrossoverPoints.push_back(calculateSelectUpperMachineConstant<T>(dop));
  }
  std::sort(upperCrossoverPoints.begin(), upperCrossoverPoints.end());
  std::cout << " Complete" << std::endl;

  std::string machineConstantLowerName =
      "SelectLower_" + std::to_string(sizeof(T)) + "B_elements_" + std::to_string(dop) + "_dop";
  std::string machineConstantUpperName =
      "SelectUpper_" + std::to_string(sizeof(T)) + "B_elements_" + std::to_string(dop) + "_dop";

  MachineConstants::getInstance().updateMachineConstant(machineConstantLowerName,
                                                        lowerCrossoverPoints[NUMBER_OF_TESTS / 2]);
  MachineConstants::getInstance().updateMachineConstant(machineConstantUpperName,
                                                        upperCrossoverPoints[NUMBER_OF_TESTS / 2]);
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_MACHINECONSTANTSIMPLEMENTATION_HPP
