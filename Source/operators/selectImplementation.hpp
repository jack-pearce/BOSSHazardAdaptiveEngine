#ifndef BOSSHAZARDADAPTIVEENGINE_SELECTIMPLEMENTATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_SELECTIMPLEMENTATION_HPP

#include "constants/machineConstants.hpp"
#include "utilities/dataStructures.hpp"
#include "utilities/papiWrapper.hpp"
#include "utilities/systemInformation.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>

// #define DEBUG

namespace adaptive {

/****************************** FOUNDATIONAL ALGORITHMS ******************************/

template <typename T, typename P> class SelectOperator {
public:
  virtual inline size_t processMicroBatch(size_t startCandidates, size_t numCandidates,
                                          Span<T>& column, T value, bool columnIsFirstArg,
                                          P& predicate, Span<uint32_t>* candidateIndexes,
                                          uint32_t* selectedIndexes) = 0;
};

template <typename T, typename P> class SelectBranch : public SelectOperator<T, P> {
public:
  inline size_t processMicroBatch(size_t startCandidates, size_t numCandidates, Span<T>& column,
                                  T value, bool columnIsFirstArg, P& predicate,
                                  Span<uint32_t>* candidateIndexes,
                                  uint32_t* selectedIndexes) final {
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
  inline size_t processMicroBatch(size_t startCandidates, size_t numCandidates, Span<T>& column,
                                  T value, bool columnIsFirstArg, P& predicate,
                                  Span<uint32_t>* candidateIndexes,
                                  uint32_t* selectedIndexes) final {
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
  SelectAdaptive(Span<T>& column_, T value_, bool columnIsFirstArg_, P& predicate_,
                 Span<uint32_t>&& candidateIndexes_);
  void adjustRobustness(int adjustment);
  Span<uint32_t> processInput();

private:
  inline void processMicroBatch();

  Span<T>& column;
  T value;
  bool columnIsFirstArg;
  Span<uint32_t> candidateIndexes;
  P& predicate;

  uint32_t tuplesPerHazardCheck;
  uint32_t maxConsecutivePredications;
  uint32_t tuplesInBranchBurst;

  uint32_t microBatchStartIndex;
  uint32_t totalSelected;
  uint32_t consecutivePredications;
  SelectImplementation activeOperator;

  MonitorSelect<T, P> monitor;
  SelectBranch<T, P> branchOperator;
  SelectPredication<T, P> predicationOperator;

  uint32_t remainingTuples;
  Span<uint32_t>* candidateIndexesPtr;
  uint32_t* selectedIndexes;
  uint32_t microBatchSize{};
  uint32_t microBatchSelected{};
};

template <typename T, typename P> class MonitorSelect {
public:
  explicit MonitorSelect(SelectAdaptive<T, P>* selectOperator_) : selectOperator(selectOperator_) {
    std::vector<std::string> counters = {"PERF_COUNT_HW_BRANCH_MISSES"};
    branchMispredictions = Counters::getInstance().getSharedEventSetEvents(counters);

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

  void checkHazards(uint32_t n, uint32_t selected) {
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
  const long_long* branchMispredictions;
  float lowerCrossoverSelectivity;
  float upperCrossoverSelectivity;
  float m;
  SelectAdaptive<T, P>* selectOperator;
};

template <typename T, typename P>
SelectAdaptive<T, P>::SelectAdaptive(Span<T>& column_, T value_, bool columnIsFirstArg_,
                                     P& predicate_, Span<uint32_t>&& candidateIndexes_)
    : column(column_), value(value_), columnIsFirstArg(columnIsFirstArg_), predicate(predicate_),
      candidateIndexes(std::move(candidateIndexes_)), tuplesPerHazardCheck(50000),
      maxConsecutivePredications(10), tuplesInBranchBurst(1000), microBatchStartIndex(0),
      totalSelected(0), consecutivePredications(0),
      activeOperator(SelectImplementation::Predication_), monitor(MonitorSelect<T, P>(this)),
      branchOperator(SelectBranch<T, P>()), predicationOperator(SelectPredication<T, P>()) {
  remainingTuples = candidateIndexes.size() == 0 ? column.size() : candidateIndexes.size();
  candidateIndexesPtr = candidateIndexes.size() > 0 ? &candidateIndexes : nullptr;
  if(candidateIndexes.size() == 0) {
    std::vector<uint32_t> vec(column.size());
    candidateIndexes = Span<uint32_t>(std::move(std::vector(vec)));
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

template <typename T, typename P> Span<uint32_t> SelectAdaptive<T, P>::processInput() {
  while(remainingTuples > 0) {
    if(__builtin_expect(consecutivePredications == maxConsecutivePredications, false)) {
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

  return std::move(candidateIndexes).subspan(0, totalSelected);
}

template <typename T, typename P> inline void SelectAdaptive<T, P>::processMicroBatch() {
  microBatchSelected = 0;
  if(activeOperator == SelectImplementation::Branch_) {
    Counters::getInstance().readSharedEventSet();
    microBatchSelected = branchOperator.processMicroBatch(
        microBatchStartIndex, microBatchSize, column, value, columnIsFirstArg, predicate,
        candidateIndexesPtr, selectedIndexes);
    Counters::getInstance().readSharedEventSet();
  } else {
    Counters::getInstance().readSharedEventSet();
    microBatchSelected = predicationOperator.processMicroBatch(
        microBatchStartIndex, microBatchSize, column, value, columnIsFirstArg, predicate,
        candidateIndexesPtr, selectedIndexes);
    Counters::getInstance().readSharedEventSet();
    consecutivePredications++;
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
  Span<T>& column;
  T value;
  bool columnIsFirstArg;
  P& predicate;
  Span<uint32_t>* candidateIndexes;
  uint32_t* selectedIndexes;
  std::atomic<uint32_t>* threadToMerge;
  std::atomic<uint32_t>* positionToWrite;
  uint32_t dop;
  uint32_t threadNum;

  SelectThreadArgs(size_t startCandidates_, size_t numCandidates_, Span<T>& column_, T value_,
                   bool columnIsFirstArg_, P& predicate_, Span<uint32_t>* candidateIndexes_,
                   uint32_t* selectedIndexes_, std::atomic<uint32_t>* threadToMerge_,
                   std::atomic<uint32_t>* positionToWrite_, uint32_t dop_, uint32_t threadNum_)
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
  Span<T>& column;
  T value;
  bool columnIsFirstArg;
  P& predicate;
  Span<uint32_t>* candidateIndexes;
  uint32_t* selectedIndexes;

  std::atomic<uint32_t>* threadToMerge;
  std::atomic<uint32_t>* positionToWrite;
  uint32_t dop;
  uint32_t threadNum;

  size_t tuplesPerHazardCheck;
  size_t maxConsecutivePredications;
  size_t tuplesInBranchBurst;

  SelectImplementation activeOperator;
  SelectBranch<T, P> branchOperator;
  SelectPredication<T, P> predicationOperator;

  uint32_t* threadSelectionBuffer;
  uint32_t* threadSelection;
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
  MonitorSelectParallel(SelectAdaptiveParallelAux<T, P>* selectOperator_, uint32_t dop,
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
      predicationOperator(SelectPredication<T, P>()), consecutivePredications(0),
      threadSelected(0) {

  if(threadNum == 0) {
    threadSelectionBuffer = selectedIndexes;
    threadSelection = selectedIndexes;
  } else {
    threadSelectionBuffer = new uint32_t[remainingTuples];
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
#if DEBUG
    std::cout << "Switched to select predication" << std::endl;
#endif
    activeOperator = SelectImplementation::Predication_;
  } else if(__builtin_expect(
                (adjustment < 0) && activeOperator == SelectImplementation::Predication_, false)) {
#if DEBUG
    std::cout << "Switched to select branch" << std::endl;
#endif
    activeOperator = SelectImplementation::Branch_;
    consecutivePredications = 0;
  }
}

template <typename T, typename P> void SelectAdaptiveParallelAux<T, P>::processInput() {
  while(remainingTuples > 0) {
    if(__builtin_expect(consecutivePredications == maxConsecutivePredications, false)) {
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
    readThreadEventSet(eventSet, 1, branchMispredictions.get());
    microBatchSelected = branchOperator.processMicroBatch(
        microBatchStartIndex, microBatchSize, column, value, columnIsFirstArg, predicate,
        candidateIndexes, threadSelection);
    readThreadEventSet(eventSet, 1, branchMispredictions.get());
  } else {
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

  memcpy(selectedIndexes + writeIndex, threadSelectionBuffer, threadSelected * sizeof(uint32_t));
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
  SelectAdaptiveParallel(Span<T>& column_, T value_, bool columnIsFirstArg_, P& predicate_,
                         Span<uint32_t>&& candidateIndexes_, uint32_t dop_)
      : column(column_), value(value_), columnIsFirstArg(columnIsFirstArg_), predicate(predicate_),
        candidateIndexes(std::move(candidateIndexes_)), dop(dop_) {
    n = candidateIndexes.size() == 0 ? column.size() : candidateIndexes.size();
    while(dop > n) {
      dop /= 2;
    }

    candidateIndexesPtr = candidateIndexes.size() > 0 ? &candidateIndexes : nullptr;
    if(candidateIndexes.size() == 0) {
      std::vector<uint32_t> vec(column.size());
      candidateIndexes = Span<uint32_t>(std::move(std::vector(vec)));
    }
    Counters::getInstance();
    MachineConstants::getInstance();
  }

  Span<uint32_t> processInput() {
    if(n == 0) {
      return std::move(candidateIndexes).subspan(0, 0);
    }

    pthread_t threads[dop];
    size_t tuplesPerThread = n / dop;
    vectorOfPairs<size_t, size_t> threadIndexes(dop, std::make_pair(0, tuplesPerThread));
    threadIndexes.back().second = n - ((dop - 1) * tuplesPerThread);

    size_t startIndex = 0;
    for(auto& threadIndex : threadIndexes) {
      threadIndex.first = startIndex;
      startIndex += threadIndex.second;
    }

    std::atomic<uint32_t> threadToMerge = 0;
    std::atomic<uint32_t> positionToWrite = 0;
    std::vector<std::unique_ptr<SelectThreadArgs<T, P>>> threadArgs;

    for(auto i = 0; i < dop; ++i) {
      threadArgs.push_back(std::make_unique<SelectThreadArgs<T, P>>(
          threadIndexes[i].first, threadIndexes[i].second, column, value, columnIsFirstArg,
          predicate, candidateIndexesPtr, candidateIndexes.begin(), &threadToMerge,
          &positionToWrite, dop, i));
      pthread_create(&threads[i], NULL, selectAdaptiveParallelAux<T, P>, threadArgs[i].get());
    }

    for(int i = 0; i < dop; ++i) {
      pthread_join(threads[i], nullptr);
    }

    return std::move(candidateIndexes).subspan(0, positionToWrite);
  }

private:
  Span<T>& column;
  T value;
  bool columnIsFirstArg;
  Span<uint32_t> candidateIndexes;
  P& predicate;

  uint32_t dop;
  size_t n;
  Span<uint32_t>* candidateIndexesPtr;
};

/****************************** ENTRY FUNCTION ******************************/

template <typename T, typename P>
Span<uint32_t> select(Select implementation, Span<T>& column, T value, bool columnIsFirstArg,
                      P& predicate, Span<uint32_t>&& candidateIndexes, size_t dop) {
  assert(1 <= dop && dop <= logicalCoresCount());
  if(implementation == Select::AdaptiveParallel) {
    SelectAdaptiveParallel<T, P> selectOperator(column, value, columnIsFirstArg, predicate,
                                                std::move(candidateIndexes), dop);
    return std::move(selectOperator.processInput());
  }

  assert(dop == 1);
  if(implementation == Select::Adaptive) {
    SelectAdaptive<T, P> selectOperator(column, value, columnIsFirstArg, predicate,
                                        std::move(candidateIndexes));
    return std::move(selectOperator.processInput());
  }

  auto candidateIndexesPtr = candidateIndexes.size() > 0 ? &candidateIndexes : nullptr;
  if(candidateIndexes.size() == 0) {
    std::vector<uint32_t> vec(column.size());
    candidateIndexes = Span<uint32_t>(std::move(std::vector(vec)));
  }
  uint32_t* selectedIndexes = candidateIndexes.begin();
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
