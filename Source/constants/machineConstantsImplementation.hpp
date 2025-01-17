#ifndef BOSSHAZARDADAPTIVEENGINE_MACHINECONSTANTSIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_MACHINECONSTANTSIMPLEMENTATION_HPP

#include <iostream>
#include <optional>

#include "HazardAdaptiveEngine.hpp"
#include "groupQueries.hpp"
#include "lazy_hash_map/robin_map.h"
#include "operators/group.hpp"
#include "operators/select.hpp"
#include "utilities/dataGeneration.hpp"
#include "utilities/memory.hpp"
#include "utilities/papiWrapper.hpp"
#include "utilities/utilities.hpp"

// #define DEBUG_MACHINE_CONSTANTS

/**************************************** CONFIG **************************************/

constexpr int SELECT_NUMBER_OF_TESTS = 9; // Can be reduced down to 1 to speed up calibration
constexpr size_t SELECT_DATA_SIZE = 100 * 1000 * 1000;
constexpr int SELECT_ITERATIONS = 12;

constexpr int GROUP_NUMBER_OF_CARDINALITY_TESTS = 1;
constexpr int GROUP_NUMBER_OF_CONSTANTS_TESTS = 1;
constexpr size_t GROUP_DATA_SIZE = 200 * 1000 * 1000;
constexpr int GROUP_ITERATIONS = 12;
constexpr double GROUP_VARIABILITY_MARGIN_PAGE_FAULTS = 0.0; // 0%
constexpr double GROUP_VARIABILITY_MARGIN_LLC = 0.3;         // 30%

constexpr float HASHMAP_OVERALLOCATION_FACTOR = 2.5;
constexpr float PERCENT_INPUT_IN_PAGE_FAULTS_CHECK = 0.00005; // 0.005%

/**************************************** UTILITIES **************************************/

namespace adaptive {

template <typename T> using Aggregator = std::function<T(const T, const T)>;

template <typename T>
const Aggregator<T> minAggregator = [](const T currentAggregate, const T numberToInclude) -> T {
  return std::min(currentAggregate, numberToInclude);
};

template <typename T>
const Aggregator<T> maxAggregator = [](const T currentAggregate, const T numberToInclude) -> T {
  return std::max(currentAggregate, numberToInclude);
};

template <typename T>
const Aggregator<T> sumAggregator = [](const T currentAggregate, const T numberToInclude) -> T {
  return currentAggregate + numberToInclude;
};

struct PerfCounterResults {
  long_long cycles;
  double pageFaultDecreaseRatePerTuple;
  double llcMissRate;
};

enum Measurement { CYCLES, HAZARDS };

/**************************************** SELECT **************************************/

template <typename T> T getThreshold(uint32_t n, double selectivity) {
  return static_cast<T>(n * selectivity);
}

template <typename T> double calculateSelectLowerMachineConstant(uint32_t dop) {
  auto data = generateRandomisedUniqueValuesInMemory<T>(SELECT_DATA_SIZE);

  auto predicate = std::greater();

  auto& eventSet = getThreadEventSet();
  long_long* cycles = eventSet.getCounterDiffsPtr() + EVENT::CPU_CYCLES;
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
        branchCycles = *cycles;
      } else {
        MachineConstants::getInstance().updateMachineConstant(machineConstantLowerName, 1);
        MachineConstants::getInstance().updateMachineConstant(machineConstantUpperName, 0);
        branchCycles = PAPI_get_real_usec();
        select(Select::AdaptiveParallel, columnSpan1,
               1 + getThreshold<T>(SELECT_DATA_SIZE, midSelectivity), false, predicate, {}, dop);
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
        predicationCycles = *cycles;
      } else {
        MachineConstants::getInstance().updateMachineConstant(machineConstantLowerName, 0);
        MachineConstants::getInstance().updateMachineConstant(machineConstantUpperName, 1);
        predicationCycles = PAPI_get_real_usec();
        select(Select::AdaptiveParallel, columnSpan2,
               1 + getThreshold<T>(SELECT_DATA_SIZE, midSelectivity), false, predicate, {}, dop);
        predicationCycles = PAPI_get_real_usec() - predicationCycles;
      }
    }

    if(branchCycles > predicationCycles) {
      upperSelectivity = midSelectivity;
    } else {
      lowerSelectivity = midSelectivity;
    }

#ifdef DEBUG_MACHINE_CONSTANTS
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

  auto& eventSet = getThreadEventSet();
  long_long* cycles = eventSet.getCounterDiffsPtr() + EVENT::CPU_CYCLES;
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
        branchCycles = *cycles;
      } else {
        MachineConstants::getInstance().updateMachineConstant(machineConstantLowerName, 1);
        MachineConstants::getInstance().updateMachineConstant(machineConstantUpperName, 0);
        branchCycles = PAPI_get_real_usec();
        select(Select::AdaptiveParallel, columnSpan1,
               1 + getThreshold<T>(SELECT_DATA_SIZE, midSelectivity), false, predicate, {}, dop);
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
        predicationCycles = *cycles;
      } else {
        MachineConstants::getInstance().updateMachineConstant(machineConstantLowerName, 0);
        MachineConstants::getInstance().updateMachineConstant(machineConstantUpperName, 1);
        predicationCycles = PAPI_get_real_usec();
        select(Select::AdaptiveParallel, columnSpan2,
               1 + getThreshold<T>(SELECT_DATA_SIZE, midSelectivity), false, predicate, {}, dop);
        predicationCycles = PAPI_get_real_usec() - predicationCycles;
      }
    }

    if(branchCycles > predicationCycles) {
      lowerSelectivity = midSelectivity;
    } else {
      upperSelectivity = midSelectivity;
    }

#ifdef DEBUG_MACHINE_CONSTANTS
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
  for(auto i = 0; i < SELECT_NUMBER_OF_TESTS; ++i) {
    lowerCrossoverPoints.push_back(calculateSelectLowerMachineConstant<T>(dop));
  }
  std::sort(lowerCrossoverPoints.begin(), lowerCrossoverPoints.end());
  std::cout << " Complete" << std::endl;

  std::cout << " - Running tests for upper crossover point";
  std::vector<double> upperCrossoverPoints;
  for(auto i = 0; i < SELECT_NUMBER_OF_TESTS; ++i) {
    upperCrossoverPoints.push_back(calculateSelectUpperMachineConstant<T>(dop));
  }
  std::sort(upperCrossoverPoints.begin(), upperCrossoverPoints.end());
  std::cout << " Complete" << std::endl;

  std::string machineConstantLowerName =
      "SelectLower_" + std::to_string(sizeof(T)) + "B_elements_" + std::to_string(dop) + "_dop";
  std::string machineConstantUpperName =
      "SelectUpper_" + std::to_string(sizeof(T)) + "B_elements_" + std::to_string(dop) + "_dop";

  auto& constants = MachineConstants::getInstance();
  constants.updateMachineConstant(machineConstantLowerName,
                                  lowerCrossoverPoints[SELECT_NUMBER_OF_TESTS / 2]);
  constants.updateMachineConstant(machineConstantUpperName,
                                  upperCrossoverPoints[SELECT_NUMBER_OF_TESTS / 2]);
}

/**************************************** GROUP **************************************/

template <typename K, typename... As>
inline void groupByHashAux(HA_tsl::robin_map<K, std::tuple<As...>>& map, int startIndex, int n,
                           const K* keys, const As*... aggregates, Aggregator<As>... aggregators) {
  typename HA_tsl::robin_map<K, std::tuple<As...>>::iterator it;

  for(int index = startIndex; index < startIndex + n; ++index) {
    auto key = keys[index];
    it = map.find(key);
    if(it != map.end()) {
      it.value() = std::apply(
          [&](auto&&... args) { return std::make_tuple(aggregators(args, aggregates[index])...); },
          std::move(it->second));
    } else {
      map.insert({key, std::make_tuple(aggregates[index]...)});
    }
  }
}

template <typename K, typename... As>
PerfCounterResults groupByHash(int cardinality, int n, const K* keys, const As*... aggregates,
                               Aggregator<As>... aggregators) {
  int initialSize = std::max(
      static_cast<int>(HASHMAP_OVERALLOCATION_FACTOR * static_cast<float>(cardinality)), 400000);

  HA_tsl::robin_map<K, std::tuple<As...>> map(initialSize);
  auto& eventSet = getThreadEventSet();

  int tuplesInTransientCheck =
      static_cast<int>(static_cast<float>(n) * PERCENT_INPUT_IN_PAGE_FAULTS_CHECK);
  int tuplesPerTransientCheckReading = static_cast<int>(std::ceil(tuplesInTransientCheck / 10.0));
  std::vector<int> tuplesPerReading;
  tuplesPerReading.reserve(10);
  for(int i = 1; i < 11; i++) {
    tuplesPerReading.push_back(tuplesPerTransientCheckReading * i);
  }

  std::vector<long_long> pageFaults;
  pageFaults.reserve(10);
  long_long lastLevelCacheMisses = 0;
  long_long* pageFaultsPtr = eventSet.getCounterDiffsPtr() + EVENT::PAGE_FAULTS;
  long_long* lastLevelCacheMissesPtr =
      eventSet.getCounterDiffsPtr() + EVENT::LAST_LEVEL_CACHE_MISSES;
  int tuplesProcessed = 0;

  for(int i = 0; i < 10; i++) {
    eventSet.readCounters();
    groupByHashAux<K, As...>(map, tuplesProcessed, tuplesPerTransientCheckReading, keys,
                             aggregates..., aggregators...);
    eventSet.readCountersAndUpdateDiff();
    pageFaults.push_back(*pageFaultsPtr);
    std::cout << "pageFaults: " << pageFaults.back() << std::endl;
    lastLevelCacheMisses += *lastLevelCacheMissesPtr;
    tuplesProcessed += tuplesPerTransientCheckReading;
  }

  size_t remaining = n - tuplesProcessed;
  eventSet.readCounters();
  groupByHashAux<K, As...>(map, tuplesProcessed, remaining, keys, aggregates..., aggregators...);
  eventSet.readCountersAndUpdateDiff();
  lastLevelCacheMisses += *lastLevelCacheMissesPtr;

  double pageFaultDecreaseRatePerTuple =
      std::abs(linearRegressionSlope(tuplesPerReading, pageFaults));
  double lastLevelCacheMissRate =
      static_cast<double>(n) / static_cast<double>(lastLevelCacheMisses);

#ifdef DEBUG_MACHINE_CONSTANTS
  std::cout << "pageFaultDecreaseRatePerTuple: " << pageFaultDecreaseRatePerTuple << '\n';
  std::cout << "lastLevelCacheMissRate: " << lastLevelCacheMissRate << std::endl;
#endif

  return {0, pageFaultDecreaseRatePerTuple, lastLevelCacheMissRate};
}

template <typename K, typename... As>
PerfCounterResults runGroupFunctionMeasureHazards(int cardinality, int dop,
                                                  ExpressionSpanArguments&& keySpans,
                                                  std::vector<Span<As>>&&... typedAggCols,
                                                  Aggregator<As>... aggregators) {
  auto& threadPool = ThreadPool::getInstance(std::nullopt);
  auto& synchroniser = Synchroniser::getInstance();
  auto& keySpan = std::get<Span<K>>(keySpans.at(0));
  int n = keySpan.size();
  if(dop == 1) {
    return groupByHash<K, As...>(cardinality, n, &(keySpan[0]), &(typedAggCols[0][0])...,
                                 aggregators...);
  } else {
    int tuplesPerThreadBaseline = n / dop;
    int remainingTuples = n % dop;
    int tuplesPerThread;
    int start = 0;

    std::atomic<double> totalPageFaultDecreaseRate = 0;
    std::atomic<double> totalLlcMissRate = 0;

    for(auto taskNum = 0; taskNum < dop; ++taskNum) {
      tuplesPerThread = tuplesPerThreadBaseline + (taskNum < remainingTuples);

      threadPool.enqueue([&synchroniser, &totalPageFaultDecreaseRate, &totalLlcMissRate,
                          cardinality, start, tuplesPerThread, &keySpan, &typedAggCols...,
                          aggregators...] {
        auto measurements = groupByHash<K, As...>(cardinality, tuplesPerThread, &(keySpan[start]),
                                                  &(typedAggCols[0][start])..., aggregators...);
        totalPageFaultDecreaseRate += measurements.pageFaultDecreaseRatePerTuple;
        totalLlcMissRate += measurements.llcMissRate;
        synchroniser.taskComplete();
      });

      start += tuplesPerThread;
    }
    synchroniser.waitUntilComplete(dop);

    return {0, totalPageFaultDecreaseRate / static_cast<double>(dop),
            totalLlcMissRate / static_cast<double>(dop)};
  }
}

template <typename K, typename... As>
PerfCounterResults runGroupFunctionMeasureCycles(Group implementation, uint32_t dop,
                                                 ExpressionSpanArguments&& keySpans,
                                                 std::vector<Span<As>>&&... typedAggCols,
                                                 Aggregator<As>... aggregators) {
  long_long cycles;
  ExpressionSpanArguments keySpans2;
  if(dop == 1) {
    assert(implementation == Group::Hash || implementation == Group::Sort);
    auto& eventSet = getThreadEventSet();
    long_long* cyclesPtr = eventSet.getCounterDiffsPtr() + EVENT::CPU_CYCLES;
    eventSet.readCounters();
    group<K, As...>(implementation, dop, 1, std::move(keySpans), std::move(keySpans2),
                    std::move(typedAggCols)..., aggregators...);
    eventSet.readCountersAndUpdateDiff();
    cycles = *cyclesPtr;
  } else {
    assert(implementation == Group::GroupAdaptiveParallel);
    cycles = PAPI_get_real_usec();
    group<K, As...>(implementation, dop, 1, std::move(keySpans), std::move(keySpans2),
                    std::move(typedAggCols)..., aggregators...);
    cycles = PAPI_get_real_usec() - cycles;
  }
  return {cycles, 0, 0};
}

template <typename K, typename... As>
PerfCounterResults runGroupFunctionDataCollection(int cardinality, Measurement measurement,
                                                  Group implementation, uint32_t dop,
                                                  ExpressionSpanArguments&& keySpans,
                                                  std::vector<Span<As>>&&... typedAggCols,
                                                  Aggregator<As>... aggregators) {
  if(measurement == Measurement::CYCLES) {
    setCardinalityEnvironmentVariable(cardinality);
    return runGroupFunctionMeasureCycles<K, As...>(implementation, dop, std::move(keySpans),
                                                   std::move(typedAggCols)..., aggregators...);
  } else {
    return runGroupFunctionMeasureHazards<K, As...>(cardinality, dop, std::move(keySpans),
                                                    std::move(typedAggCols)..., aggregators...);
  }
}

template <typename K>
PerfCounterResults runGroupFunctionCreatePayload(int cardinality, Measurement measurement,
                                                 Group implementation, GROUP_QUERIES groupQuery,
                                                 uint32_t dop, size_t n_,
                                                 ExpressionSpanArguments&& keySpans) {
  int n = static_cast<int>(n_);
  auto masterPayload_32 = loadVectorIntoSpan(generateUniformDistribution<int32_t>(n, 1, 100000));
  auto payload_32 = shallowCopySpan<int32_t>(masterPayload_32);

  if(groupQuery == GROUP_QUERIES::Bytes_8 || groupQuery == GROUP_QUERIES::Bytes_12) {
    return runGroupFunctionDataCollection<K, int32_t>(cardinality, measurement, implementation, dop,
                                                      std::move(keySpans), std::move(payload_32),
                                                      maxAggregator<int32_t>);
  }

  auto masterPayload_64_1 =
      loadVectorIntoSpan(generateUniformDistribution<int64_t>(n, 1, 100000, 1));
  auto payload_64_1 = shallowCopySpan<int64_t>(masterPayload_64_1);
  auto payload_64_2 = shallowCopySpan<int64_t>(masterPayload_64_1);
  auto payload_64_3 = shallowCopySpan<int64_t>(masterPayload_64_1);

  if(groupQuery == GROUP_QUERIES::Bytes_16) {
    return runGroupFunctionDataCollection<K, int64_t>(cardinality, measurement, implementation, dop,
                                                      std::move(keySpans), std::move(payload_64_1),
                                                      maxAggregator<int64_t>);
  } else if(groupQuery == GROUP_QUERIES::Bytes_20) {
    return runGroupFunctionDataCollection<K, int64_t, int32_t>(
        cardinality, measurement, implementation, dop, std::move(keySpans), std::move(payload_64_1),
        std::move(payload_32), maxAggregator<int64_t>, maxAggregator<int32_t>);
  } else if(groupQuery == GROUP_QUERIES::Bytes_24) {
    return runGroupFunctionDataCollection<K, int64_t, int64_t>(
        cardinality, measurement, implementation, dop, std::move(keySpans), std::move(payload_64_1),
        std::move(payload_64_2), maxAggregator<int64_t>, minAggregator<int64_t>);
  } else if(groupQuery == GROUP_QUERIES::Bytes_28) {
    return runGroupFunctionDataCollection<K, int64_t, int64_t, int32_t>(
        cardinality, measurement, implementation, dop, std::move(keySpans), std::move(payload_64_1),
        std::move(payload_64_2), std::move(payload_32), maxAggregator<int64_t>,
        minAggregator<int64_t>, maxAggregator<int32_t>);
  } else if(groupQuery == GROUP_QUERIES::Bytes_32) {
    return runGroupFunctionDataCollection<K, int64_t, int64_t, int64_t>(
        cardinality, measurement, implementation, dop, std::move(keySpans), std::move(payload_64_1),
        std::move(payload_64_2), std::move(payload_64_3), maxAggregator<int64_t>,
        minAggregator<int64_t>, sumAggregator<int64_t>);
  } else if(groupQuery == GROUP_QUERIES::Bytes_36) {
    return runGroupFunctionDataCollection<K, int64_t, int64_t, int64_t, int32_t>(
        cardinality, measurement, implementation, dop, std::move(keySpans), std::move(payload_64_1),
        std::move(payload_64_2), std::move(payload_64_3), std::move(payload_32),
        maxAggregator<int64_t>, minAggregator<int64_t>, sumAggregator<int64_t>,
        maxAggregator<int32_t>);
  }

  auto masterPayload_64_2 =
      loadVectorIntoSpan(generateUniformDistribution<int64_t>(n, 1, 100000, 2));
  auto payload_64_4 = shallowCopySpan<int64_t>(masterPayload_64_2);
  auto payload_64_5 = shallowCopySpan<int64_t>(masterPayload_64_2);

  if(groupQuery == GROUP_QUERIES::Bytes_40) {
    return runGroupFunctionDataCollection<K, int64_t, int64_t, int64_t, int64_t>(
        cardinality, measurement, implementation, dop, std::move(keySpans), std::move(payload_64_1),
        std::move(payload_64_2), std::move(payload_64_3), std::move(payload_64_4),
        maxAggregator<int64_t>, minAggregator<int64_t>, sumAggregator<int64_t>,
        maxAggregator<int64_t>);
  } else if(groupQuery == GROUP_QUERIES::Bytes_44) {
    return runGroupFunctionDataCollection<K, int64_t, int64_t, int64_t, int64_t, int32_t>(
        cardinality, measurement, implementation, dop, std::move(keySpans), std::move(payload_64_1),
        std::move(payload_64_2), std::move(payload_64_3), std::move(payload_64_4),
        std::move(payload_32), maxAggregator<int64_t>, minAggregator<int64_t>,
        sumAggregator<int64_t>, maxAggregator<int64_t>, maxAggregator<int32_t>);
  } else if(groupQuery == GROUP_QUERIES::Bytes_48) {
    return runGroupFunctionDataCollection<K, int64_t, int64_t, int64_t, int64_t, int64_t>(
        cardinality, measurement, implementation, dop, std::move(keySpans), std::move(payload_64_1),
        std::move(payload_64_2), std::move(payload_64_3), std::move(payload_64_4),
        std::move(payload_64_5), maxAggregator<int64_t>, minAggregator<int64_t>,
        sumAggregator<int64_t>, maxAggregator<int64_t>, minAggregator<int64_t>);
  }

  std::cout << "Invalid GROUP_QUERY value" << std::endl;
  std::exit(1);
}

PerfCounterResults runGroupFunction(Measurement measurement, Group implementation,
                                    GROUP_QUERIES groupQuery, uint32_t dop, size_t n_,
                                    int cardinality) {
  int n = static_cast<int>(n_);
  if(groupQuery == GROUP_QUERIES::Bytes_8) {
    auto keySpans = loadVectorIntoSpans(
        generateUniformDistributionWithSetCardinality<int32_t>(n, n, cardinality));
    return runGroupFunctionCreatePayload<int32_t>(cardinality, measurement, implementation,
                                                  groupQuery, dop, n_, std::move(keySpans));
  } else {
    auto keySpans = loadVectorIntoSpans(
        generateUniformDistributionWithSetCardinality<int64_t>(n, n, cardinality));
    return runGroupFunctionCreatePayload<int64_t>(cardinality, measurement, implementation,
                                                  groupQuery, dop, n_, std::move(keySpans));
  }
}

int calculateGroupByCrossoverCardinality(GROUP_QUERIES groupQuery, int dop) {
  int n = static_cast<int>(GROUP_DATA_SIZE);

  long_long hashCycles, sortCycles;
  int upperCardinality = n;
  int lowerCardinality = 1;
  int midCardinality;

  auto& constants = MachineConstants::getInstance();
  auto names = getGroupMachineConstantNames(groupQuery, dop);

  for(int i = 0; i < GROUP_ITERATIONS; ++i) {

    midCardinality = (lowerCardinality + upperCardinality) / 2;

    if(dop == 1) {
      auto results =
          runGroupFunction(Measurement::CYCLES, Group::Hash, groupQuery, dop, n, midCardinality);
      hashCycles = results.cycles;
    } else {
      constants.updateMachineConstant(names.pageFaultDecreaseRate, 0);
      constants.updateMachineConstant(names.llcMissRate, 0);
      auto results = runGroupFunction(Measurement::CYCLES, Group::GroupAdaptiveParallel, groupQuery,
                                      dop, n, midCardinality);
      hashCycles = results.cycles;
    }

    if(dop == 1) {
      auto results =
          runGroupFunction(Measurement::CYCLES, Group::Sort, groupQuery, dop, n, midCardinality);
      sortCycles = results.cycles;
    } else {
      constants.updateMachineConstant(names.pageFaultDecreaseRate, 1000000);
      constants.updateMachineConstant(names.llcMissRate, 1000000);
      auto results = runGroupFunction(Measurement::CYCLES, Group::GroupAdaptiveParallel, groupQuery,
                                      dop, n, midCardinality);
      sortCycles = results.cycles;
    }

    if(hashCycles > sortCycles) {
      upperCardinality = midCardinality;
    } else {
      lowerCardinality = midCardinality;
    }

#ifdef DEBUG_MACHINE_CONSTANTS
    std::cout << "Cardinality: " << (lowerCardinality + upperCardinality) / 2
              << ", hash cycles: " << hashCycles << ", sort cycles: " << sortCycles << std::endl;
#endif
  }

  std::cout << ".";
  std::cout.flush();

  return (lowerCardinality + upperCardinality) / 2;
}

void calculateGroupMachineConstants(GROUP_QUERIES groupQuery, int dop) {

  auto names = getGroupMachineConstantNames(groupQuery, dop);
  auto name = names.llcMissRate.substr(0, names.llcMissRate.size() - 4);
  std::cout << "Calculating machine constants for " << name << std::endl;
  std::cout << " - Running tests for crossover point";
  std::cout.flush();

  std::vector<double> crossoverPoints;
  for(int i = 0; i < GROUP_NUMBER_OF_CARDINALITY_TESTS; ++i) {
    crossoverPoints.push_back(calculateGroupByCrossoverCardinality(groupQuery, dop));
  }
  std::sort(crossoverPoints.begin(), crossoverPoints.end());
  int crossoverCardinality =
      static_cast<int>(crossoverPoints[GROUP_NUMBER_OF_CARDINALITY_TESTS / 2]);
  std::cout << " Complete" << std::endl;

  std::cout << " - Running tests for page fault decrease rate and last level cache misses";
  std::vector<double> pageFaultDecreaseRatePerTuple;
  std::vector<double> tuplesPerLastLevelCacheMiss;
  int n = static_cast<int>(GROUP_DATA_SIZE);
  for(int i = 0; i < GROUP_NUMBER_OF_CONSTANTS_TESTS; ++i) {
    auto [_, pageFaultDecreaseRate, llcMissRate] = runGroupFunction(
        Measurement::HAZARDS, Group::Hash, groupQuery, dop, n, crossoverCardinality);
    pageFaultDecreaseRatePerTuple.push_back(pageFaultDecreaseRate);
    tuplesPerLastLevelCacheMiss.push_back(llcMissRate);
  }
  std::sort(pageFaultDecreaseRatePerTuple.begin(), pageFaultDecreaseRatePerTuple.end());
  std::sort(tuplesPerLastLevelCacheMiss.begin(), tuplesPerLastLevelCacheMiss.end());
  std::cout << " Complete" << std::endl;

  auto& constants = MachineConstants::getInstance();
  constants.updateMachineConstant(
      names.pageFaultDecreaseRate,
      (1.0 - GROUP_VARIABILITY_MARGIN_PAGE_FAULTS) *
          pageFaultDecreaseRatePerTuple[GROUP_NUMBER_OF_CONSTANTS_TESTS / 2]);
  constants.updateMachineConstant(
      names.llcMissRate, (1.0 - GROUP_VARIABILITY_MARGIN_LLC) *
                             tuplesPerLastLevelCacheMiss[GROUP_NUMBER_OF_CONSTANTS_TESTS / 2]);
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_MACHINECONSTANTSIMPLEMENTATION_HPP
