#ifndef BOSSHAZARDADAPTIVEENGINE_DATAGENERATION_HPP
#define BOSSHAZARDADAPTIVEENGINE_DATAGENERATION_HPP

namespace adaptive {

template <typename T> void copyVector(const std::vector<T>& source, std::vector<T>& destination);

template <typename T> std::vector<T> generateRandomisedUniqueValuesInMemory(size_t n);

} // namespace adaptive

#include "dataGenerationImplementation.hpp"

#endif // BOSSHAZARDADAPTIVEENGINE_DATAGENERATION_HPP
