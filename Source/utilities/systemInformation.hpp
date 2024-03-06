#ifndef BOSSHAZARDADAPTIVEENGINE_SYSTEMINFORMATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_SYSTEMINFORMATION_HPP

#include <immintrin.h>
#include "config.hpp"
#include <unistd.h>

namespace adaptive {

template <typename T> bool arrayIsSimd128Aligned(const T* array) {
  const size_t simdAlignment = sizeof(__m128i);
  return reinterpret_cast<uintptr_t>(array) % simdAlignment == 0;
}

template <typename T> bool arrayIsSimd256Aligned(const T* array) {
  const size_t simdAlignment = sizeof(__m256i);
  return reinterpret_cast<uintptr_t>(array) % simdAlignment == 0;
}

inline uint64_t l3cacheSize() { return sysconf(_SC_LEVEL3_CACHE_SIZE); }
inline uint64_t l2cacheSize() { return sysconf(_SC_LEVEL2_CACHE_SIZE); }
inline uint32_t bytesPerCacheLine() { return sysconf(_SC_LEVEL1_DCACHE_LINESIZE); }
inline uint32_t logicalCoresCount() { return adaptive::config::LOGICAL_CORE_COUNT; }
inline std::string getProjectRootDirectory() { return adaptive::config::projectFilePath; }
void printIntelTlbSpecifications();
uint32_t l2TlbEntriesFor4KbytePages();

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_SYSTEMINFORMATION_HPP
