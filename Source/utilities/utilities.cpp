#include "utilities.hpp"

uint32_t roundDownToPowerOf2(uint32_t num) {
  if(num <= 2) {
    return 1;
  }
  uint32_t msbPos = sizeof(uint32_t) * 8 - __builtin_clz(num) - 1;
  if((num & (num - 1)) == 0) {
    return 1 << (msbPos - 1);
  } else {
    return 1 << msbPos;
  }
}

void setCardinalityEnvironmentVariable(int cardinality) {
  std::string value = std::to_string(cardinality);
  setenv("GROUP_RESULT_CARDINALITY", value.c_str(), 1);
}
