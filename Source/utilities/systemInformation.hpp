#ifndef BOSSHAZARDADAPTIVEENGINE_SYSTEMINFORMATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_SYSTEMINFORMATION_HPP

#include <immintrin.h>

namespace adaptive {

template <typename T> bool arrayIsSimd128Aligned(const T* array) {
  const size_t simdAlignment = sizeof(__m128i);
  return reinterpret_cast<uintptr_t>(array) % simdAlignment == 0;
}

template <typename T> bool arrayIsSimd256Aligned(const T* array) {
  const size_t simdAlignment = sizeof(__m256i);
  return reinterpret_cast<uintptr_t>(array) % simdAlignment == 0;
}

uint64_t l3cacheSize();
uint32_t bytesPerCacheLine();
uint32_t logicalCoresCount();
std::string getProjectRootDirectory();
void printIntelTlbSpecifications();
uint32_t l2TlbEntriesFor4KbytePages();

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_SYSTEMINFORMATION_HPP
