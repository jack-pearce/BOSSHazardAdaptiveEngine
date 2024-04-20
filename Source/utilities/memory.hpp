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
    std::lock_guard<std::mutex> lock(mutex);
    tasks.emplace([func = std::forward<Function>(f),
                   arguments = std::make_tuple(std::forward<Args>(args)...)]() mutable {
      std::apply(std::move(func), std::move(arguments));
    });
    cv.notify_one();
  }

private:
  explicit ThreadPool(size_t numThreads) : stop(false) {
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
    cv.notify_all();
    for(auto& thread : threads) {
      pthread_join(thread, nullptr);
    }
  }

  static void* workerThread(void* arg) {
    auto& pool = *static_cast<ThreadPool*>(arg);
    std::function<void()> task;
    while(true) {
      {
        std::unique_lock<std::mutex> lock(pool.mutex);
        pool.cv.wait(lock, [&pool] { return pool.stop || !pool.tasks.empty(); });
        if(pool.stop && pool.tasks.empty()) {
          return nullptr;
        }
        task = std::move(pool.tasks.front());
        pool.tasks.pop();
      }

      task();
    }
  }

  std::vector<pthread_t> threads;
  std::queue<std::function<void()>> tasks;
  std::mutex mutex;
  std::condition_variable cv;
  bool stop;
};

class Synchroniser {
public:
  static Synchroniser& getInstance() {
    static Synchroniser instance;
    return instance;
  }

  void taskComplete() {
    std::lock_guard<std::mutex> lock(mutex);
    tasksCompletedCount++;
    cv.notify_one();
  }

  void waitUntilComplete(int tasksToComplete) {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this, tasksToComplete] { return tasksCompletedCount == tasksToComplete; });
    tasksCompletedCount = 0;
  }

private:
  Synchroniser() : tasksCompletedCount(0) {
#ifdef MEMORY_INFO
    std::cout << "Constructing thread synchrnoiser object\n";
#endif
  }

  std::mutex mutex;
  std::condition_variable cv;
  int tasksCompletedCount;
};

#endif // BOSSHAZARDADAPTIVEENGINE_MEMORY_HPP
