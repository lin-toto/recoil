#include "file.h"

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

    std::vector<ValueType> stringToSymbols(const std::string &str) {
        std::vector<ValueType> symbols;
        symbols.resize(str.length());
        std::transform(str.begin(), str.end(), symbols.begin(), [] (char v) {
            return static_cast<unsigned char>(v);
        });

        return symbols;
    }
}