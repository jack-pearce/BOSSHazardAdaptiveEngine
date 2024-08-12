#ifndef BOSSHAZARDADAPTIVEENGINE_SELECT_CPP
#define BOSSHAZARDADAPTIVEENGINE_SELECT_CPP

#include "select.hpp"

namespace adaptive {

std::string getSelectName(Select select) {
  switch(select) {
  case Select::Branch:
    return "Select_Branch";
  case Select::Predication:
    return "Select_Predication";
  case Select::Adaptive:
    return "Select_Adaptive";
  default:
    throw std::runtime_error("Invalid selection of 'Select' implementation!");
  }
}

} // namespace adaptive

#endif // BOSSHAZARDADAPTIVEENGINE_SELECT_CPP
