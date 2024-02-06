#ifndef BOSSHAZARDADAPTIVEENGINE_SELECTIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_SELECTIMPLEMENTATION_HPP

#include "constants/machineConstants.hpp"
#include "operators/operatorStats.hpp"
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

//#define PRINT_SELECTIVITY
//#define STATE_DEBUG
//#define DEBUG
//#define VERBOSE_DEBUG

namespace adaptive {

/****************************** FOUNDATIONAL ALGORITHMS ******************************/

template <typename T, typename P> class SelectOperator {
public:
  virtual inline size_t processMicroBatch(size_t startCandidates, size_t numCandidates,
                                          const Span<T>& column, T value, bool columnIsFirstArg,
                                          P& predicate, const Span<int32_t>* candidateIndexes,
                                          int32_t* selectedIndexes) = 0;
};

template <typename T, typename P> class SelectBranch : public SelectOperator<T, P> {
public:
  inline size_t processMicroBatch(size_t startCandidates, size_t numCandidates,
                                  const Span<T>& column, T value, bool columnIsFirstArg,
                                  P& predicate, const Span<int32_t>* candidateIndexes,
                                  int32_t* selectedIndexes) final {
    size_t numSelected = 0;
    if(!candidateIndexes) {
      if(columnIsFirstArg) {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          if(predicate(column[i], value)) {
            selectedIndexes[numSelected++] = i;
          }
        }
      } else {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          if(predicate(value, column[i])) {
            selectedIndexes[numSelected++] = i;
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

template <typename T, typename P> class SelectPredication : public SelectOperator<T, P> {
public:
  inline size_t processMicroBatch(size_t startCandidates, size_t numCandidates,
                                  const Span<T>& column, T value, bool columnIsFirstArg,
                                  P& predicate, const Span<int32_t>* candidateIndexes,
                                  int32_t* selectedIndexes) final {
    size_t numSelected = 0;
    if(!candidateIndexes) {
      if(columnIsFirstArg) {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          selectedIndexes[numSelected] = i;
          numSelected += predicate(column[i], value);
        }
      } else {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          selectedIndexes[numSelected] = i;
          numSelected += predicate(value, column[i]);
        }
      }
    } else {
      if(columnIsFirstArg) {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          selectedIndexes[numSelected] = (*candidateIndexes)[i];
          numSelected += predicate(column[(*candidateIndexes)[i]], value);
        }
      } else {
        for(auto i = startCandidates; i < (startCandidates + numCandidates); ++i) {
          selectedIndexes[numSelected] = (*candidateIndexes)[i];
          numSelected += predicate(value, column[(*candidateIndexes)[i]]);
        }
      }
    }
    return numSelected;
  }
};

/****************************** SINGLE-THREADED ******************************/

template <typename T, typename P> class MonitorSelect;

template <typename T, typename P> class SelectAdaptive {
public:
  SelectAdaptive(const Span<T>& column_, T value_, bool columnIsFirstArg_, P& predicate_,
                 Span<int32_t>&& candidateIndexes_, SelectOperatorState* state_);
  void adjustRobustness(int adjustment);
  Span<int32_t> processInput();

private:
  inline void processMicroBatch();

  const Span<T>& column;
  T value;
  bool columnIsFirstArg;
  P& predicate;
  Span<int32_t> candidateIndexes;

  int32_t tuplesPerHazardCheck;
  int32_t maxConsecutivePredications;
  int32_t tuplesInBranchBurst;

  int32_t microBatchStartIndex;
  int32_t totalSelected;
  int32_t consecutivePredications;
  SelectImplementation activeOperator;

  MonitorSelect<T, P> monitor;
  SelectBranch<T, P> branchOperator;
  SelectPredication<T, P> predicationOperator;

  int32_t remainingTuples;
  Span<int32_t>* candidateIndexesPtr;
  int32_t* selectedIndexes;
  int32_t microBatchSize{};
  int32_t microBatchSelected{};

  SelectOperatorState* state;
};

template <typename T, typename P> class MonitorSelect {
public:
  explicit MonitorSelect(SelectAdaptive<T, P>* selectOperator_) : selectOperator(selectOperator_) {
    branchMispredictions = Counters::getInstance().getBranchMisPredictionsCounter();

    std::string lowerName = "SelectLower_" + std::to_string(sizeof(T)) + "B_elements_1_dop";
    std::string upperName = "SelectUpper_" + std::to_string(sizeof(T)) + "B_elements_1_dop";
    lowerCrossoverSelectivity =
        static_cast<float>(MachineConstants::getInstance().getMachineConstant(lowerName));
    upperCrossoverSelectivity =
        static_cast<float>(MachineConstants::getInstance().getMachineConstant(upperName));

    // Gradient of number of branch misses between lower and upper cross-over selectivity
    m = ((1 - upperCrossoverSelectivity) - lowerCrossoverSelectivity) /
        (upperCrossoverSelectivity - lowerCrossoverSelectivity);
  }

  void checkHazards(int32_t n, int32_t selected) {
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

  void checkHazards(int32_t n, int32_t selected, int32_t additionalBranchMispredictions) {
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

  int32_t getMispredictions() { return static_cast<int32_t>(*branchMispredictions); }

private:
  const long_long* branchMispredictions;
  float lowerCrossoverSelectivity;
  float upperCrossoverSelectivity;
  float m;
  SelectAdaptive<T, P>* selectOperator;
};

template <typename T, typename P>
SelectAdaptive<T, P>::SelectAdaptive(const Span<T>& column_, T value_, bool columnIsFirstArg_,
                                     P& predicate_, Span<int32_t>&& candidateIndexes_,
                                     SelectOperatorState* state_)
    : column(column_), value(value_), columnIsFirstArg(columnIsFirstArg_), predicate(predicate_),
      candidateIndexes(std::move(candidateIndexes_)), tuplesPerHazardCheck(50000),
      maxConsecutivePredications(10), tuplesInBranchBurst(1000), microBatchStartIndex(0),
      totalSelected(0), consecutivePredications(maxConsecutivePredications),
      activeOperator(SelectImplementation::Predication_), monitor(MonitorSelect<T, P>(this)),
      branchOperator(SelectBranch<T, P>()), predicationOperator(SelectPredication<T, P>()),
      state(state_) {
  remainingTuples = candidateIndexes.size() == 0 ? column.size() : candidateIndexes.size();
  candidateIndexesPtr = candidateIndexes.size() > 0 ? &candidateIndexes : nullptr;
  if(candidateIndexes.size() == 0) {
    auto* indexesArray = new int32_t[column.size()];
    candidateIndexes =
        Span<int32_t>(indexesArray, column.size(), [indexesArray]() { delete[] indexesArray; });
  }
  selectedIndexes = candidateIndexes.begin();
}

template <typename T, typename P> void SelectAdaptive<T, P>::adjustRobustness(int adjustment) {
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

template <typename T, typename P> Span<int32_t> SelectAdaptive<T, P>::processInput() {
#ifdef PRINT_SELECTIVITY
  std::cout << candidateIndexes.size() << " input values into predicate" << std::endl;
#endif

  // If a previous state exists then start from this point
  if(state && state->consecutivePredications != -1) {
    activeOperator = state->activeOperator;               // Load operator state
    if(state->tuplesUntilHazardCheck > remainingTuples) { // We will not complete a hazard check
      microBatchSize = remainingTuples;
      processMicroBatch();

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
    } else {
      consecutivePredications = state->consecutivePredications;
      microBatchSize = state->tuplesUntilHazardCheck;
      processMicroBatch();
      consecutivePredications += activeOperator == SelectImplementation::Predication_;
      monitor.checkHazards(microBatchSize + state->tuplesProcessed,
                           microBatchSelected + state->selected, state->branchMispredictions);
    }
  }

  // Run as many full microBatches as possible
  while(remainingTuples > tuplesPerHazardCheck ||
        (consecutivePredications >= maxConsecutivePredications &&
         remainingTuples > tuplesInBranchBurst)) {
    if(__builtin_expect(consecutivePredications >= maxConsecutivePredications, false)) {
#ifdef DEBUG
      std::cout << "Running branch burst" << std::endl;
#endif
      activeOperator = SelectImplementation::Branch_;
      consecutivePredications = 0;
      microBatchSize = tuplesInBranchBurst;
    } else {
      microBatchSize = tuplesPerHazardCheck;
    }

    processMicroBatch();
    consecutivePredications += activeOperator == SelectImplementation::Predication_;
    monitor.checkHazards(microBatchSize, microBatchSelected);
  }

  // Run final (partial-)microBatch without updating robustness (either we are at the final tuple of
  // the whole input so no reason to update robustness, or it will be updated on the next chunk)
  microBatchSize = remainingTuples;
  processMicroBatch();

  if(state) {
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

template <typename T, typename P> inline void SelectAdaptive<T, P>::processMicroBatch() {
  microBatchSelected = 0;
  if(activeOperator == SelectImplementation::Branch_) {
#ifdef VERBOSE_DEBUG
    std::cout << "Processing micro-batch with Branch" << std::endl;
#endif
    Counters::getInstance().readEventSetAndGetCycles();
    microBatchSelected = branchOperator.processMicroBatch(
        microBatchStartIndex, microBatchSize, column, value, columnIsFirstArg, predicate,
        candidateIndexesPtr, selectedIndexes);
    Counters::getInstance().readEventSetAndCalculateDiff();
  } else {
#ifdef VERBOSE_DEBUG
    std::cout << "Processing micro-batch with Predication" << std::endl;
#endif
    Counters::getInstance().readEventSetAndGetCycles();
    microBatchSelected = predicationOperator.processMicroBatch(
        microBatchStartIndex, microBatchSize, column, value, columnIsFirstArg, predicate,
        candidateIndexesPtr, selectedIndexes);
    Counters::getInstance().readEventSetAndCalculateDiff();
  }
  remainingTuples -= microBatchSize;
  microBatchStartIndex += microBatchSize;
  selectedIndexes += microBatchSelected;
  totalSelected += microBatchSelected;
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

template <typename T, typename P> class SelectAdaptiveParallelAux {
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
  std::unique_ptr<long_long> branchMispredictions;
  std::unique_ptr<MonitorSelectParallel<T, P>> monitor;
  int eventSet;

  size_t consecutivePredications;
  size_t threadSelected;

  size_t microBatchSize{};
  size_t microBatchSelected{};
};

template <typename T, typename P> class MonitorSelectParallel {
public:
  MonitorSelectParallel(SelectAdaptiveParallelAux<T, P>* selectOperator_, int32_t dop,
                        long_long* branchMispredictions_)
      : branchMispredictions(branchMispredictions_), selectOperator(selectOperator_) {

    std::string lowerName =
        "SelectLower_" + std::to_string(sizeof(T)) + "B_elements_" + std::to_string(dop) + "_dop";
    std::string upperName =
        "SelectUpper_" + std::to_string(sizeof(T)) + "B_elements_" + std::to_string(dop) + "_dop";
    lowerCrossoverSelectivity =
        static_cast<float>(MachineConstants::getInstance().getMachineConstant(lowerName));
    upperCrossoverSelectivity =
        static_cast<float>(MachineConstants::getInstance().getMachineConstant(upperName));

    // Gradient of number of branch misses between lower and upper cross-over selectivity
    m = ((1 - upperCrossoverSelectivity) - lowerCrossoverSelectivity) /
        (upperCrossoverSelectivity - lowerCrossoverSelectivity);
  }

  void checkHazards(size_t n, size_t selected) {
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

private:
  long_long* branchMispredictions;
  float lowerCrossoverSelectivity;
  float upperCrossoverSelectivity;
  float m;
  SelectAdaptiveParallelAux<T, P>* selectOperator;
};

template <typename T, typename P>
SelectAdaptiveParallelAux<T, P>::SelectAdaptiveParallelAux(SelectThreadArgs<T, P>* args)
    : microBatchStartIndex(args->startCandidates), remainingTuples(args->numCandidates),
      column(args->column), value(args->value), columnIsFirstArg(args->columnIsFirstArg),
      predicate(args->predicate), candidateIndexes(args->candidateIndexes),
      selectedIndexes(args->selectedIndexes), threadToMerge(args->threadToMerge),
      positionToWrite(args->positionToWrite), dop(args->dop), threadNum(args->threadNum),
      tuplesPerHazardCheck(50000), maxConsecutivePredications(10), tuplesInBranchBurst(1000),
      activeOperator(SelectImplementation::Predication_), branchOperator(SelectBranch<T, P>()),
      predicationOperator(SelectPredication<T, P>()),
      consecutivePredications(maxConsecutivePredications), threadSelected(0) {

  if(threadNum == 0) {
    threadSelectionBuffer = selectedIndexes;
    threadSelection = selectedIndexes;
  } else {
    threadSelectionBuffer = new int32_t[remainingTuples];
    threadSelection = threadSelectionBuffer;
  }

  eventSet = PAPI_NULL;
  std::vector<std::string> counters = {"PERF_COUNT_HW_BRANCH_MISSES"};
  createThreadEventSet(&eventSet, counters);

  branchMispredictions = std::make_unique<long_long>();
  monitor = std::make_unique<MonitorSelectParallel<T, P>>(this, dop, branchMispredictions.get());
}

template <typename T, typename P>
void SelectAdaptiveParallelAux<T, P>::adjustRobustness(int adjustment) {
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
    monitor->checkHazards(microBatchSize, microBatchSelected);
  }
}

template <typename T, typename P> inline void SelectAdaptiveParallelAux<T, P>::processMicroBatch() {
  microBatchSelected = 0;
  if(activeOperator == SelectImplementation::Branch_) {
#ifdef VERBOSE_DEBUG
    std::cout << "Processing micro-batch with Branch" << std::endl;
#endif
    readThreadEventSet(eventSet, 1, branchMispredictions.get());
    microBatchSelected = branchOperator.processMicroBatch(
        microBatchStartIndex, microBatchSize, column, value, columnIsFirstArg, predicate,
        candidateIndexes, threadSelection);
    readThreadEventSet(eventSet, 1, branchMispredictions.get());
  } else {
#ifdef VERBOSE_DEBUG
    std::cout << "Processing micro-batch with Predication" << std::endl;
#endif
    readThreadEventSet(eventSet, 1, branchMispredictions.get());
    microBatchSelected = predicationOperator.processMicroBatch(
        microBatchStartIndex, microBatchSize, column, value, columnIsFirstArg, predicate,
        candidateIndexes, threadSelection);
    readThreadEventSet(eventSet, 1, branchMispredictions.get());
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
  destroyThreadEventSet(eventSet, branchMispredictions.get());
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
                         Span<int32_t>&& candidateIndexes_, int32_t dop_, bool calibrationRun)
      : column(column_), value(value_), columnIsFirstArg(columnIsFirstArg_), predicate(predicate_),
        candidateIndexes(std::move(candidateIndexes_)), dop(dop_) {
    if(!calibrationRun) {
      MachineConstants::getInstance().calculateMissingMachineConstants();
    }

    n = candidateIndexes.size() == 0 ? column.size() : candidateIndexes.size();
    while(dop > 1 && (n / dop) < adaptive::config::minTuplesPerThread) {
      dop = roundDownToPowerOf2(dop);
    }

    candidateIndexesPtr = candidateIndexes.size() > 0 ? &candidateIndexes : nullptr;
    if(candidateIndexes.size() == 0) {
      auto* indexesArray = new int32_t[column.size()];
      candidateIndexes =
          Span<int32_t>(indexesArray, column.size(), [indexesArray]() { delete[] indexesArray; });
    }
    Counters::getInstance();
    MachineConstants::getInstance();
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

    for(int32_t i = 0; i < dop; ++i) {
      threadArgs.push_back(std::make_unique<SelectThreadArgs<T, P>>(
          threadIndexes[i].first, threadIndexes[i].second, column, value, columnIsFirstArg,
          predicate, candidateIndexesPtr, candidateIndexes.begin(), &threadToMerge,
          &positionToWrite, dop, i));
      ThreadPool::getInstance().enqueue(
          [threadArg = threadArgs[i].get()] { selectAdaptiveParallelAux<T, P>(threadArg); });
    }
    ThreadPool::getInstance().waitUntilComplete(dop);

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

template <typename T, typename P>
Span<int32_t> select(Select implementation, const Span<T>& column, T value, bool columnIsFirstArg,
                     P& predicate, Span<int32_t>&& candidateIndexes, size_t dop,
                     SelectOperatorState* state, bool calibrationRun) {
  assert(1 <= dop && dop <= logicalCoresCount());
  // TODO - update adaptive parallel to use the select operator state
  // TODO - this is the chance to unify the AdaptiveParallel and Adaptive objects into one
  if(implementation == Select::AdaptiveParallel) {
    SelectAdaptiveParallel<T, P> selectOperator(column, value, columnIsFirstArg, predicate,
                                                std::move(candidateIndexes), dop, calibrationRun);
    return std::move(selectOperator.processInput());
  }

  assert(dop == 1);
  if(implementation == Select::Adaptive) {
    SelectAdaptive<T, P> selectOperator(column, value, columnIsFirstArg, predicate,
                                        std::move(candidateIndexes), state);
    return std::move(selectOperator.processInput());
  }

  auto candidateIndexesPtr = candidateIndexes.size() > 0 ? &candidateIndexes : nullptr;
  if(candidateIndexes.size() == 0) {
    auto* indexesArray = new int32_t[column.size()];
    candidateIndexes =
        Span<int32_t>(indexesArray, column.size(), [indexesArray]() { delete[] indexesArray; });
  }
  int32_t* selectedIndexes = candidateIndexes.begin();
  size_t numSelected;

  if(implementation == Select::Branch) {
    SelectBranch<T, P> selectOperator;
    numSelected = selectOperator.processMicroBatch(
        0, candidateIndexesPtr ? candidateIndexes.size() : column.size(), column, value,
        columnIsFirstArg, predicate, candidateIndexesPtr, selectedIndexes);
  } else {
    SelectPredication<T, P> selectOperator;
    numSelected = selectOperator.processMicroBatch(
        0, candidateIndexesPtr ? candidateIndexes.size() : column.size(), column, value,
        columnIsFirstArg, predicate, candidateIndexesPtr, selectedIndexes);
  }
  return std::move(candidateIndexes).subspan(0, numSelected);
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_SELECTIMPLEMENTATION_HPP
