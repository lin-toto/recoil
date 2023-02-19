#include "file_utils.h"

#include <sstream>

namespace Recoil::Examples {
    std::string readFile(const std::string &name, std::ios::ios_base::openmode mode) {
        std::ifstream file(name, mode);
        if (!file.good()) [[unlikely]] throw std::runtime_error("Error reading strings file");

        std::stringstream strStream;
        strStream << file.rdbuf();
        std::string str = strStream.str();

        return str;
    }
}