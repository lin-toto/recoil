#ifndef RECOIL_EXAMPLE_FILE_H
#define RECOIL_EXAMPLE_FILE_H

#include <string>
#include <fstream>
#include <vector>
#include "recoil/type_aliases.h"

namespace Recoil::Examples {
    std::string readFile(const std::string &name, std::ios_base::openmode mode = std::ios_base::in);
    std::vector<ValueType> stringToSymbols(const std::string &str);
}

#endif //RECOIL_EXAMPLE_FILE_H
