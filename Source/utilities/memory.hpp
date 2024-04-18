#ifndef BOSSHAZARDADAPTIVEENGINE_MEMORY_HPP
#define BOSSHAZARDADAPTIVEENGINE_MEMORY_HPP

#include "config.hpp"

#include <condition_variable>
#include <pthread.h>
#include <queue>

// #define MEMORY_INFO

class ThreadPool {
public:
  static ThreadPool& getInstance() {
    static ThreadPool instance(adaptive::config::nonVectorizedDOP);
    return instance;
  }

  template <typename Function, typename... Args> void enqueue(Function&& f, Args&&... args) {
    std::lock_guard<std::mutex> lock(queueMutex);
    tasks.emplace([func = std::forward<Function>(f),
                   arguments = std::make_tuple(std::forward<Args>(args)...)]() mutable {
      std::apply(std::move(func), std::move(arguments));
    });
    task_cv.notify_one();
  }

  void waitUntilComplete(uint32_t tasksToComplete) {
    std::unique_lock<std::mutex> lock(tasksCountMutex);
    if(tasksCompletedCount == tasksToComplete) {
      tasksCompletedCount = 0;
      return;
    }
    wait_cv.wait(lock, [this, tasksToComplete] { return tasksCompletedCount == tasksToComplete; });
    tasksCompletedCount = 0;
  }

  void resetTaskCount() { tasksCompletedCount = 0; }

private:
  explicit ThreadPool(size_t numThreads) : stop(false), tasksCompletedCount(0) {
#ifdef MEMORY_INFO
    std::cout << "Constructing threads for thread pool\n";
#endif
    threads.resize(numThreads);
    for(std::size_t i = 0; i < numThreads; ++i) {
      pthread_create(&threads[i], nullptr, workerThread, this);
    }
  }

  ~ThreadPool() {
    stop = true;
    task_cv.notify_all();

    for(auto& thread : threads) {
      pthread_join(thread, nullptr);
    }
  }

  static void* workerThread(void* arg) {
    auto* pool = static_cast<ThreadPool*>(arg);
    while(true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(pool->queueMutex);
        pool->task_cv.wait(lock, [pool] { return pool->stop || !pool->tasks.empty(); });
        if(pool->stop && pool->tasks.empty()) {
          return nullptr;
        }
        task = std::move(pool->tasks.front());
        pool->tasks.pop();
      }
      task();
      {
        std::lock_guard<std::mutex> lock(pool->tasksCountMutex);
        ++(pool->tasksCompletedCount);
        pool->wait_cv.notify_one();
      }
    }
  }

  std::vector<pthread_t> threads;
  std::queue<std::function<void()>> tasks;
  std::mutex tasksCountMutex;
  std::mutex queueMutex;
  std::condition_variable task_cv;
  std::condition_variable wait_cv;
  bool stop;
  uint32_t tasksCompletedCount;
};

#endif // BOSSHAZARDADAPTIVEENGINE_MEMORY_HPP
