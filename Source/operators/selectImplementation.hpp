#ifndef BOSSHAZARDADAPTIVEENGINE_SELECTIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_SELECTIMPLEMENTATION_HPP

#include "HazardAdaptiveEngine.hpp"
#include "constants/machineConstants.hpp"
#include "engineInstanceState.hpp"
#include "utilities/dataStructures.hpp"
#include "utilities/memory.hpp"
#include "utilities/papiWrapper.hpp"
#include "utilities/sharedDataTypes.hpp"
#include "utilities/systemInformation.hpp"
#include "utilities/utilities.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>

// #define PRINT_SELECTIVITY
// #define CONSTANTS_DEBUG
// #define STATE_DEBUG
// #define CONSTRUCT_DEBUG
// #define DEBUG
// #define VERBOSE_DEBUG

// #define RANDOMISE_SINGLE_THREADED_ADAPTIVE_PERIOD

namespace adaptive {

// NOLINTBEGIN(readability-function-cognitive-complexity)

namespace selectConfig {
constexpr int PREFETCH_DISTANCE = 100;
constexpr uint32_t TUPLES_PER_HAZARD_CHECK = 1'000;
constexpr uint32_t MAX_CONSECUTIVE_PREDICATIONS = 10;
constexpr uint32_t TUPLES_IN_BRANCH_BURST = 1'000;
} // namespace selectConfig

/****************************** FOUNDATIONAL ALGORITHMS ******************************/

template <typename T, typename P> class SelectBranch {
public:
  inline size_t processMicroBatch(size_t startCandidates, size_t numCandidates,
                                  const Span<T>& column, T value, bool columnIsFirstArg,
                                  P& predicate, const Span<int32_t>* candidateIndexes,
                                  int32_t* selectedIndexes) {
    size_t numSelected = 0;
    if(candidateIndexes == nullptr) {
      if(columnIsFirstArg) {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          if(predicate(column[i], value)) {
            selectedIndexes[numSelected++] = static_cast<int32_t>(i);
          }
        }
      } else {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          if(predicate(value, column[i])) {
            selectedIndexes[numSelected++] = static_cast<int32_t>(i);
          }
        }
      }
    } else {
      if(columnIsFirstArg) {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          if(predicate(column[(*candidateIndexes)[i]], value)) {
            selectedIndexes[numSelected++] = (*candidateIndexes)[i];
          }
        }
      } else {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          if(predicate(value, column[(*candidateIndexes)[i]])) {
            selectedIndexes[numSelected++] = (*candidateIndexes)[i];
          }
        }
      }
    }
    return numSelected;
  }
};

template <typename T, typename P> class SelectPredication {
public:
  inline size_t processMicroBatch(size_t startCandidates, size_t numCandidates,
                                  const Span<T>& column, T value, bool columnIsFirstArg,
                                  P& predicate, const Span<int32_t>* candidateIndexes,
                                  int32_t* selectedIndexes) {
    size_t numSelected = 0;
    if(candidateIndexes == nullptr) {
      if(columnIsFirstArg) {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          selectedIndexes[numSelected] = static_cast<int32_t>(i);
          numSelected += predicate(column[i], value);
        }
      } else {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          selectedIndexes[numSelected] = static_cast<int32_t>(i);
          numSelected += predicate(value, column[i]);
        }
      }
    } else {
      /* For Predication the compiler does not automatically prefetch as it does for
       * Branch. Therefore, we explicitly prefetch. Prefetching 100 values ahead was
       * found to be optimal. The first '0' parameter means the prefetch will be
       * read only, and the second '0' parameter means the data does not need to be
       * kept in the cache after the access (i.e. low temporal locality). */
      if(columnIsFirstArg) {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          __builtin_prefetch(&column[(*candidateIndexes)[i + selectConfig::PREFETCH_DISTANCE]], 0,
                             0);
          selectedIndexes[numSelected] = (*candidateIndexes)[i];
          numSelected += predicate(column[(*candidateIndexes)[i]], value);
        }
      } else {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          __builtin_prefetch(&column[(*candidateIndexes)[i + selectConfig::PREFETCH_DISTANCE]], 0,
                             0);
          selectedIndexes[numSelected] = (*candidateIndexes)[i];
          numSelected += predicate(value, column[(*candidateIndexes)[i]]);
        }
      }
    }
    return numSelected;
  }
};

template <typename T, typename P> static SelectBranch<T, P>& getSelectBranchOperator() {
  static SelectBranch<T, P> selectBranchOperator;
  return selectBranchOperator;
}

template <typename T, typename P> static SelectPredication<T, P>& getSelectPredicationOperator() {
  static SelectPredication<T, P> selectPredicationOperator;
  return selectPredicationOperator;
}

/****************************** SINGLE-THREADED ******************************/

namespace tuplesBetweenChecks {
constexpr int32_t AVERAGE = 50'000;
constexpr int32_t RANGE = 10'000;

int32_t inline getNext(int32_t previous) {
  int64_t value = 1664525 * static_cast<int64_t>(previous) + 1013904223; // NOLINT
  return AVERAGE - RANGE + static_cast<int32_t>((value % (2 * RANGE + 1)));
}
} // namespace tuplesBetweenChecks

template <typename T, typename P> class MonitorSelect;

// This class acts as the Dispatcher
template <typename T, typename P> class SelectAdaptive {
public:
  SelectAdaptive();
  void adjustRobustness(int adjustment);
  Span<int32_t> processInput(const Span<T>& column, T value, bool columnIsFirstArg, P& predicate,
                             Span<int32_t>&& candidateIndexes_, SelectOperatorState* state,
                             int32_t engineDOP);

private:
  inline void processMicroBatch(const Span<T>& column, T value, bool columnIsFirstArg,
                                P& predicate);
  Span<int32_t> candidateIndexes;

  int32_t tuplesPerHazardCheck;
  int32_t maxConsecutivePredications;
  int32_t tuplesInBranchBurst;

  int32_t microBatchStartIndex = {};
  int32_t totalSelected = {};
  int32_t consecutivePredications;
  SelectImplementation activeOperator;

  PAPI_eventSet& eventSet;
  MonitorSelect<T, P> monitor;
  SelectBranch<T, P> branchOperator;
  SelectPredication<T, P> predicationOperator;

  int32_t remainingTuples = {};
  Span<int32_t>* candidateIndexesPtr = nullptr;
  int32_t* selectedIndexes = {};
  int32_t microBatchSize = {};
  int32_t microBatchSelected = {};
};

template <typename T, typename P> class MonitorSelect {
public:
  explicit MonitorSelect(SelectAdaptive<T, P>* selectOperator_,
                         const long_long* branchMispredictions_)
      : selectOperator(selectOperator_), branchMispredictions(branchMispredictions_),
        constantsDOP(-1) {}

  inline void updateConstants(int32_t engineDOP) {
    if(constantsDOP == engineDOP) {
      return;
    }

    constantsDOP = convertToValidDopValue(engineDOP);
    std::string lowerName = "SelectLower_" + std::to_string(sizeof(T)) + "B_elements_" +
                            std::to_string(constantsDOP) + "_dop";
    std::string upperName = "SelectUpper_" + std::to_string(sizeof(T)) + "B_elements_" +
                            std::to_string(constantsDOP) + "_dop";
    lowerCrossoverSelectivity =
        static_cast<float>(MachineConstants::getInstance().getMachineConstant(lowerName));
    upperCrossoverSelectivity =
        static_cast<float>(MachineConstants::getInstance().getMachineConstant(upperName));

    // Gradient of number of branch misses between lower and upper cross-over selectivity
    m = ((1 - upperCrossoverSelectivity) - lowerCrossoverSelectivity) /
        (upperCrossoverSelectivity - lowerCrossoverSelectivity);
#ifdef CONSTANTS_DEBUG
    std::cout << "Updated select machine constants for " << std::to_string(sizeof(T))
              << "B elements, dop=" << std::to_string(constantsDOP) << std::endl;
#endif
  }

  void checkHazards(int32_t n, int32_t selected) const {
    float selectivity = static_cast<float>(selected) / static_cast<float>(n);

    // NOLINTBEGIN
    if(__builtin_expect(
           (static_cast<float>(*branchMispredictions) / static_cast<float>(n)) >
               (((selectivity - lowerCrossoverSelectivity) * m) + lowerCrossoverSelectivity),
           false)) {
      selectOperator->adjustRobustness(1);
    }

    if(__builtin_expect((selectivity < lowerCrossoverSelectivity) ||
                            (selectivity > upperCrossoverSelectivity),
                        false)) {
      selectOperator->adjustRobustness(-1);
    }
  }

  void checkHazards(int32_t n, int32_t selected, int32_t additionalBranchMispredictions) const {
    float selectivity = static_cast<float>(selected) / static_cast<float>(n);

    if(__builtin_expect(
           (static_cast<float>((*branchMispredictions) + additionalBranchMispredictions) /
            static_cast<float>(n)) >
               (((selectivity - lowerCrossoverSelectivity) * m) + lowerCrossoverSelectivity),
           false)) {
      selectOperator->adjustRobustness(1);
    }

    if(__builtin_expect((selectivity < lowerCrossoverSelectivity) ||
                            (selectivity > upperCrossoverSelectivity),
                        false)) {
      selectOperator->adjustRobustness(-1);
    }
  }
  // NOLINTEND

  [[nodiscard]] int32_t getMispredictions() const {
    return static_cast<int32_t>(*branchMispredictions);
  }

private:
  SelectAdaptive<T, P>* selectOperator;
  const long_long* branchMispredictions;
  int32_t constantsDOP;
  float lowerCrossoverSelectivity{};
  float upperCrossoverSelectivity{};
  float m{};
};

template <typename T, typename P>
SelectAdaptive<T, P>::SelectAdaptive()
    : tuplesPerHazardCheck(tuplesBetweenChecks::AVERAGE),
      maxConsecutivePredications(selectConfig::MAX_CONSECUTIVE_PREDICATIONS),
      tuplesInBranchBurst(selectConfig::TUPLES_IN_BRANCH_BURST),
      consecutivePredications(maxConsecutivePredications),
      activeOperator(SelectImplementation::Predication_), eventSet(getThreadEventSet()),
      monitor(MonitorSelect<T, P>(this,
                                  (eventSet.getCounterDiffsPtr() + EVENT::BRANCH_MISPREDICTIONS))),
      branchOperator(SelectBranch<T, P>()), predicationOperator(SelectPredication<T, P>()) {
#ifdef CONSTRUCT_DEBUG
  std::cout << "Constructing Select Adaptive operator object" << std::endl;
#endif
}

// NOLINTBEGIN
template <typename T, typename P> void SelectAdaptive<T, P>::adjustRobustness(int adjustment) {
  if(__builtin_expect((adjustment > 0) && activeOperator == SelectImplementation::Branch_, false)) {
#ifdef DEBUG
    std::cout << "Switched to select predication" << std::endl;
#endif
    activeOperator = SelectImplementation::Predication_;
  } else if(__builtin_expect(
                (adjustment < 0) && activeOperator == SelectImplementation::Predication_, false)) {
// NOLINTEND
#ifdef DEBUG
    std::cout << "Switched to select branch" << std::endl;
#endif
    activeOperator = SelectImplementation::Branch_;
    consecutivePredications = 0;
  }
}

template <typename T, typename P>
Span<int32_t> SelectAdaptive<T, P>::processInput(const Span<T>& column, T value,
                                                 bool columnIsFirstArg, P& predicate,
                                                 Span<int32_t>&& candidateIndexes_,
                                                 SelectOperatorState* state, int32_t engineDOP) {
  monitor.updateConstants(engineDOP); // Ensure monitor is using correct machine constants

  microBatchStartIndex = 0; // Reset tracking parameters
  totalSelected = 0;

  candidateIndexes = std::move(candidateIndexes_);
  remainingTuples = candidateIndexes.size() == 0 ? column.size() : candidateIndexes.size();
  candidateIndexesPtr = candidateIndexes.size() > 0 ? &candidateIndexes : nullptr;
  if(candidateIndexes.size() == 0) {
    auto* indexesArray = new int32_t[column.size()];
    candidateIndexes =
        Span<int32_t>(indexesArray, column.size(), [indexesArray]() { delete[] indexesArray; });
  }
  selectedIndexes = candidateIndexes.begin();

#ifdef PRINT_SELECTIVITY
  std::cout << candidateIndexes.size() << " input values into predicate" << std::endl;
#endif

  // If a previous state exists then start from this point
  if(state != nullptr && state->consecutivePredications != -1) {
    activeOperator = state->activeOperator;               // Load operator state
    if(state->tuplesUntilHazardCheck > remainingTuples) { // We will not complete a hazard check
      microBatchSize = remainingTuples;
      processMicroBatch(column, value, columnIsFirstArg, predicate);

      state->tuplesProcessed += microBatchSize; // Update state with operator results and return
      state->branchMispredictions += monitor.getMispredictions();
      state->selected += microBatchSelected;
      state->tuplesUntilHazardCheck -= microBatchSize;
#ifdef STATE_DEBUG
      std::cout << "Completed predicate chunk, state: "
                << "\n"
                << *state << std::endl;
#endif
#ifdef PRINT_SELECTIVITY
      std::cout << totalSelected << " values selected by predicate" << std::endl;
#endif
      return std::move(candidateIndexes).subspan(0, totalSelected);
    } else { // NOLINT
      consecutivePredications = state->consecutivePredications;
      microBatchSize = state->tuplesUntilHazardCheck;
      processMicroBatch(column, value, columnIsFirstArg, predicate);
      consecutivePredications +=
          static_cast<int>(activeOperator == SelectImplementation::Predication_);
      monitor.checkHazards(microBatchSize + state->tuplesProcessed,
                           microBatchSelected + state->selected, state->branchMispredictions);
    }
  }

  // Run as many full microBatches as possible
  while(remainingTuples > tuplesPerHazardCheck ||
        (consecutivePredications >= maxConsecutivePredications &&
         remainingTuples > tuplesInBranchBurst)) {
    // NOLINTBEGIN
    if(__builtin_expect(consecutivePredications >= maxConsecutivePredications, false)) {
// NOLINTEND
#ifdef DEBUG
      std::cout << "Running branch burst" << std::endl;
#endif
      activeOperator = SelectImplementation::Branch_;
      consecutivePredications = 0;
      microBatchSize = tuplesInBranchBurst;
    } else {
      microBatchSize = tuplesPerHazardCheck;
    }

    processMicroBatch(column, value, columnIsFirstArg, predicate);
    consecutivePredications +=
        static_cast<int>(activeOperator == SelectImplementation::Predication_);
    monitor.checkHazards(microBatchSize, microBatchSelected);

#ifdef RANDOMISE_SINGLE_THREADED_ADAPTIVE_PERIOD
    tuplesPerHazardCheck = tuplesBetweenChecks::getNext(tuplesPerHazardCheck);
#endif
  }

  // Run final (partial-)microBatch without updating robustness (either we are at the final tuple of
  // the whole input so no reason to update robustness, or it will be updated on the next chunk)
  microBatchSize = remainingTuples;
  processMicroBatch(column, value, columnIsFirstArg, predicate);

  if(state != nullptr) {
    state->activeOperator = activeOperator; // Save operator state
    state->consecutivePredications = consecutivePredications;
    state->tuplesProcessed = microBatchSize;
    state->branchMispredictions = monitor.getMispredictions();
    state->selected = microBatchSelected;
    state->tuplesUntilHazardCheck = consecutivePredications >= maxConsecutivePredications
                                        ? tuplesInBranchBurst - microBatchSize
                                        : tuplesPerHazardCheck - microBatchSize;
#ifdef STATE_DEBUG
    std::cout << "Completed predicate chunk, state: "
              << "\n"
              << *state << std::endl;
#endif
  }
#ifdef PRINT_SELECTIVITY
  std::cout << totalSelected << " values selected by predicate" << std::endl;
#endif
  return std::move(candidateIndexes).subspan(0, totalSelected);
}

template <typename T, typename P>
inline void SelectAdaptive<T, P>::processMicroBatch(const Span<T>& column, T value,
                                                    bool columnIsFirstArg, P& predicate) {
  microBatchSelected = 0;
  if(activeOperator == SelectImplementation::Branch_) {
#ifdef VERBOSE_DEBUG
    std::cout << "Processing micro-batch with Branch" << std::endl;
#endif
    eventSet.readCounters();
    microBatchSelected = branchOperator.processMicroBatch(
        microBatchStartIndex, microBatchSize, column, value, columnIsFirstArg, predicate,
        candidateIndexesPtr, selectedIndexes);
    eventSet.readCountersAndUpdateDiff();
  } else {
#ifdef VERBOSE_DEBUG
    std::cout << "Processing micro-batch with Predication" << std::endl;
#endif
    eventSet.readCounters();
    microBatchSelected = predicationOperator.processMicroBatch(
        microBatchStartIndex, microBatchSize, column, value, columnIsFirstArg, predicate,
        candidateIndexesPtr, selectedIndexes);
    eventSet.readCountersAndUpdateDiff();
  }
  remainingTuples -= microBatchSize;
  microBatchStartIndex += microBatchSize;
  selectedIndexes += microBatchSelected;
  totalSelected += microBatchSelected;
}

template <typename T, typename P> static SelectAdaptive<T, P>& getSelectAdaptiveOperator() {
  thread_local static SelectAdaptive<T, P> selectAdaptiveOperator;
  return selectAdaptiveOperator;
}

/****************************** MULTI-THREADED ******************************/

template <typename T, typename P> struct SelectThreadArgs {
  size_t startCandidates;
  size_t numCandidates;
  const Span<T>& column;
  T value;
  bool columnIsFirstArg;
  P& predicate;
  const Span<int32_t>* candidateIndexes;
  int32_t* selectedIndexes;
  std::atomic<int32_t>* threadToMerge;
  std::atomic<int32_t>* positionToWrite;
  int32_t dop;
  int32_t threadNum;

  SelectThreadArgs(size_t startCandidates_, size_t numCandidates_, const Span<T>& column_, T value_,
                   bool columnIsFirstArg_, P& predicate_, const Span<int32_t>* candidateIndexes_,
                   int32_t* selectedIndexes_, std::atomic<int32_t>* threadToMerge_,
                   std::atomic<int32_t>* positionToWrite_, int32_t dop_, int32_t threadNum_)
      : startCandidates(startCandidates_), numCandidates(numCandidates_), column(column_),
        value(value_), columnIsFirstArg(columnIsFirstArg_), predicate(predicate_),
        candidateIndexes(candidateIndexes_), selectedIndexes(selectedIndexes_),
        threadToMerge(threadToMerge_), positionToWrite(positionToWrite_), dop(dop_),
        threadNum(threadNum_) {}
};

template <typename T, typename P> class MonitorSelectParallel;

template <typename T, typename P> class SelectAdaptiveParallelAux { // NOLINT
public:
  explicit SelectAdaptiveParallelAux(SelectThreadArgs<T, P>* args);
  void adjustRobustness(int adjustment);
  void processInput();
  void mergeOutput();
  ~SelectAdaptiveParallelAux();

private:
  inline void processMicroBatch();

  size_t microBatchStartIndex;
  size_t remainingTuples;
  const Span<T>& column;
  T value;
  bool columnIsFirstArg;
  P& predicate;
  const Span<int32_t>* candidateIndexes;
  int32_t* selectedIndexes;

  std::atomic<int32_t>* threadToMerge;
  std::atomic<int32_t>* positionToWrite;
  int32_t dop;
  int32_t threadNum;

  size_t tuplesPerHazardCheck;
  size_t maxConsecutivePredications;
  size_t tuplesInBranchBurst;

  SelectImplementation activeOperator;
  SelectBranch<T, P> branchOperator;
  SelectPredication<T, P> predicationOperator;

  int32_t* threadSelectionBuffer;
  int32_t* threadSelection;
  PAPI_eventSet& eventSet;
  MonitorSelectParallel<T, P> monitor;

  size_t consecutivePredications;
  size_t threadSelected;

  size_t microBatchSize{};
  size_t microBatchSelected{};
};

template <typename T, typename P> class MonitorSelectParallel {
public:
  MonitorSelectParallel(SelectAdaptiveParallelAux<T, P>* selectOperator_, int32_t dop,
                        const long_long* branchMispredictions_)
      : selectOperator(selectOperator_), branchMispredictions(branchMispredictions_) {

    std::string lowerName =
        "SelectLower_" + std::to_string(sizeof(T)) + "B_elements_" + std::to_string(dop) + "_dop";
    std::string upperName =
        "SelectUpper_" + std::to_string(sizeof(T)) + "B_elements_" + std::to_string(dop) + "_dop";
    lowerCrossoverSelectivity =
        static_cast<float>(MachineConstants::getInstance().getMachineConstant(lowerName));
    upperCrossoverSelectivity =
        static_cast<float>(MachineConstants::getInstance().getMachineConstant(upperName));

    // NOLINTBEGIN
    // Gradient of number of branch misses between lower and upper cross-over selectivity
    m = ((1 - upperCrossoverSelectivity) - lowerCrossoverSelectivity) /
        (upperCrossoverSelectivity - lowerCrossoverSelectivity);
  }

  void checkHazards(size_t n, size_t selected) const {
    float selectivity = static_cast<float>(selected) / static_cast<float>(n);

    if(__builtin_expect(
           (static_cast<float>(*branchMispredictions) / static_cast<float>(n)) >
               (((selectivity - lowerCrossoverSelectivity) * m) + lowerCrossoverSelectivity),
           false)) {
      selectOperator->adjustRobustness(1);
    }

    if(__builtin_expect((selectivity < lowerCrossoverSelectivity) ||
                            (selectivity > upperCrossoverSelectivity),
                        false)) {
      selectOperator->adjustRobustness(-1);
    }
  }
  // NOLINTEND

private:
  float lowerCrossoverSelectivity;
  float upperCrossoverSelectivity;
  float m;
  SelectAdaptiveParallelAux<T, P>* selectOperator;
  const long_long* branchMispredictions;
};

template <typename T, typename P>
SelectAdaptiveParallelAux<T, P>::SelectAdaptiveParallelAux(SelectThreadArgs<T, P>* args)
    : microBatchStartIndex(args->startCandidates), remainingTuples(args->numCandidates),
      column(args->column), value(args->value), columnIsFirstArg(args->columnIsFirstArg),
      predicate(args->predicate), candidateIndexes(args->candidateIndexes),
      selectedIndexes(args->selectedIndexes), threadToMerge(args->threadToMerge),
      positionToWrite(args->positionToWrite), dop(args->dop), threadNum(args->threadNum),
      tuplesPerHazardCheck(selectConfig::TUPLES_PER_HAZARD_CHECK),
      maxConsecutivePredications(selectConfig::MAX_CONSECUTIVE_PREDICATIONS),
      tuplesInBranchBurst(selectConfig::TUPLES_IN_BRANCH_BURST),
      activeOperator(SelectImplementation::Predication_), branchOperator(SelectBranch<T, P>()),
      predicationOperator(SelectPredication<T, P>()), eventSet(getThreadEventSet()),
      monitor(MonitorSelectParallel<T, P>(
          this, dop, (eventSet.getCounterDiffsPtr() + EVENT::BRANCH_MISPREDICTIONS))),
      consecutivePredications(maxConsecutivePredications), threadSelected(0) {

  if(threadNum == 0) {
    threadSelectionBuffer = selectedIndexes;
    threadSelection = selectedIndexes;
  } else {
    threadSelectionBuffer = new int32_t[remainingTuples];
    threadSelection = threadSelectionBuffer;
  }
}

template <typename T, typename P>
void SelectAdaptiveParallelAux<T, P>::adjustRobustness(int adjustment) {
  // NOLINTBEGIN
  if(__builtin_expect((adjustment > 0) && activeOperator == SelectImplementation::Branch_, false)) {
#ifdef DEBUG
    std::cout << "Switched to select predication" << std::endl;
#endif
    activeOperator = SelectImplementation::Predication_;
  } else if(__builtin_expect(
                (adjustment < 0) && activeOperator == SelectImplementation::Predication_, false)) {
#ifdef DEBUG
    std::cout << "Switched to select branch" << std::endl;
#endif
    activeOperator = SelectImplementation::Branch_;
    consecutivePredications = 0;
  }
}

template <typename T, typename P> void SelectAdaptiveParallelAux<T, P>::processInput() {
  while(remainingTuples > 0) {
    if(__builtin_expect(consecutivePredications >= maxConsecutivePredications, false)) {
      // NOLINTEND
#ifdef DEBUG
      std::cout << "Running branch burst" << std::endl;
#endif
      activeOperator = SelectImplementation::Branch_;
      consecutivePredications = 0;
      microBatchSize = std::min(remainingTuples, tuplesInBranchBurst);
    } else {
      microBatchSize = std::min(remainingTuples, tuplesPerHazardCheck);
    }
    processMicroBatch();
    monitor.checkHazards(microBatchSize, microBatchSelected);
  }
}

template <typename T, typename P> inline void SelectAdaptiveParallelAux<T, P>::processMicroBatch() {
  microBatchSelected = 0;
  if(activeOperator == SelectImplementation::Branch_) {
#ifdef VERBOSE_DEBUG
    std::cout << "Processing micro-batch with Branch" << std::endl;
#endif
    eventSet.readCounters();
    microBatchSelected = branchOperator.processMicroBatch(
        microBatchStartIndex, microBatchSize, column, value, columnIsFirstArg, predicate,
        candidateIndexes, threadSelection);
    eventSet.readCountersAndUpdateDiff();
  } else {
#ifdef VERBOSE_DEBUG
    std::cout << "Processing micro-batch with Predication" << std::endl;
#endif
    eventSet.readCounters();
    microBatchSelected = predicationOperator.processMicroBatch(
        microBatchStartIndex, microBatchSize, column, value, columnIsFirstArg, predicate,
        candidateIndexes, threadSelection);
    eventSet.readCountersAndUpdateDiff();
    consecutivePredications++;
  }
  remainingTuples -= microBatchSize;
  microBatchStartIndex += microBatchSize;
  threadSelection += microBatchSelected;
  threadSelected += microBatchSelected;
}

template <typename T, typename P> void SelectAdaptiveParallelAux<T, P>::mergeOutput() {
  while((*threadToMerge).load(std::memory_order_acquire) != threadNum) {
  }
  auto writeIndex = (*positionToWrite).fetch_add(threadSelected, std::memory_order_release);
  (*threadToMerge).fetch_add(1, std::memory_order_release);

  if(threadNum == 0) {
    return;
  }

  memcpy(selectedIndexes + writeIndex, threadSelectionBuffer, threadSelected * sizeof(int32_t));
}

template <typename T, typename P> SelectAdaptiveParallelAux<T, P>::~SelectAdaptiveParallelAux() {
  if(threadNum != 0) {
    delete[] threadSelectionBuffer;
  }
}

template <typename T, typename P> void* selectAdaptiveParallelAux(void* arg) {
  auto* args = static_cast<SelectThreadArgs<T, P>*>(arg);
  SelectAdaptiveParallelAux<T, P> selectOperator(args);

  selectOperator.processInput();
  selectOperator.mergeOutput();

  return nullptr;
}

template <typename T, typename P> class SelectAdaptiveParallel {
public:
  SelectAdaptiveParallel(const Span<T>& column_, T value_, bool columnIsFirstArg_, P& predicate_,
                         Span<int32_t>&& candidateIndexes_, int32_t dop_)
      : column(column_), value(value_), columnIsFirstArg(columnIsFirstArg_), predicate(predicate_),
        candidateIndexes(std::move(candidateIndexes_)), dop(dop_) {

    n = candidateIndexes.size() == 0 ? column.size() : candidateIndexes.size(); // NOLINT
    while(dop > 1 && (n / dop) < adaptive::config::minTuplesPerThread) {
      dop = roundDownToPowerOf2(dop);
    }

    candidateIndexesPtr = candidateIndexes.size() > 0 ? &candidateIndexes : nullptr;
    if(candidateIndexes.size() == 0) {
      auto* indexesArray = new int32_t[column.size()];
      candidateIndexes =
          Span<int32_t>(indexesArray, column.size(), [indexesArray]() { delete[] indexesArray; });
    }
  }

  Span<int32_t> processInput() {
#ifdef PRINT_SELECTIVITY
    std::cout << candidateIndexes.size() << " input values into predicate" << std::endl;
#endif
    if(n == 0) {
#ifdef PRINT_SELECTIVITY
      std::cout << "0 values selected by predicate" << std::endl;
#endif
      return std::move(candidateIndexes).subspan(0, 0);
    }

    size_t tuplesPerThread = n / dop;
    vectorOfPairs<size_t, size_t> threadIndexes(dop, std::make_pair(0, tuplesPerThread));
    threadIndexes.back().second = n - ((dop - 1) * tuplesPerThread);

    size_t startIndex = 0;
    for(auto& threadIndex : threadIndexes) {
      threadIndex.first = startIndex;
      startIndex += threadIndex.second;
    }

    std::atomic<int32_t> threadToMerge = 0;
    std::atomic<int32_t> positionToWrite = 0;
    std::vector<std::unique_ptr<SelectThreadArgs<T, P>>> threadArgs;

    auto& threadPool = ThreadPool::getInstance(dop);
    assert(threadPool.getNumThreads() >= dop);
    auto& synchroniser = Synchroniser::getInstance();

    for(int32_t i = 0; i < dop; ++i) {
      threadArgs.push_back(std::make_unique<SelectThreadArgs<T, P>>(
          threadIndexes[i].first, threadIndexes[i].second, column, value, columnIsFirstArg,
          predicate, candidateIndexesPtr, candidateIndexes.begin(), &threadToMerge,
          &positionToWrite, dop, i));
      threadPool.enqueue([threadArg = threadArgs[i].get(), &synchroniser] {
        selectAdaptiveParallelAux<T, P>(threadArg);
        synchroniser.taskComplete();
      });
    }
    synchroniser.waitUntilComplete(dop);

#ifdef PRINT_SELECTIVITY
    std::cout << positionToWrite << " values selected by predicate" << std::endl;
#endif
    return std::move(candidateIndexes).subspan(0, positionToWrite);
  }

private:
  const Span<T>& column;
  T value;
  bool columnIsFirstArg;
  P& predicate;
  Span<int32_t> candidateIndexes;

  int32_t dop;
  size_t n;
  Span<int32_t>* candidateIndexesPtr;
};

/****************************** ENTRY FUNCTION ******************************/

template <typename T, typename U, typename P>
Span<int32_t> select(Select implementation, const Span<T>& column, U value, bool columnIsFirstArg,
                     P& predicate, Span<int32_t>&& candidateIndexes, size_t engineDOP,
                     SelectOperatorState* state) {
  assert(1 <= engineDOP && engineDOP <= logicalCoresCount());
  // Adaptive Parallel means engine is not vectorized. Will execute engineDOP threads in engine
  if(implementation == Select::AdaptiveParallel) {
    SelectAdaptiveParallel<T, P> selectOperator(column, value, columnIsFirstArg, predicate,
                                                std::move(candidateIndexes), engineDOP);
    return std::move(selectOperator.processInput());
  }

  // Branch, Predication, and Adaptive are all single threaded in engine instance, but the engine
  // could be vectorized, in this case the engineDOP represents the vectorized DOP.
  if(implementation == Select::Adaptive) {
    return std::move(getSelectAdaptiveOperator<T, P>().processInput(
        column, value, columnIsFirstArg, predicate, std::move(candidateIndexes), state, engineDOP));
  }

  auto* candidateIndexesPtr = candidateIndexes.size() > 0 ? &candidateIndexes : nullptr;
  if(candidateIndexes.size() == 0) {
    auto* indexesArray = new int32_t[column.size()];
    candidateIndexes =
        Span<int32_t>(indexesArray, column.size(), [indexesArray]() { delete[] indexesArray; });
  }
  int32_t* selectedIndexes = candidateIndexes.begin();
  size_t numSelected = 0;

  if(implementation == Select::Branch) {
    numSelected = getSelectBranchOperator<T, P>().processMicroBatch(
        0, candidateIndexesPtr ? candidateIndexes.size() : column.size(), column, value,
        columnIsFirstArg, predicate, candidateIndexesPtr, selectedIndexes);
  } else {
    numSelected = getSelectPredicationOperator<T, P>().processMicroBatch(
        0, candidateIndexesPtr ? candidateIndexes.size() : column.size(), column, value,
        columnIsFirstArg, predicate, candidateIndexesPtr, selectedIndexes);
  }
  return std::move(candidateIndexes).subspan(0, numSelected);
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_SELECTIMPLEMENTATION_HPP

// NOLINTEND(readability-function-cognitive-complexity)