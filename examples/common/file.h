#ifndef RECOIL_EXAMPLE_FILE_H
#define RECOIL_EXAMPLE_FILE_H

#include <string>
#include <fstream>
#include <vector>
#include <concepts>

namespace Recoil::Examples {
    std::string readFile(const std::string &name, std::ios_base::openmode mode = std::ios_base::in);

    template<std::unsigned_integral ValueType>
    std::vector<ValueType> stringToSymbols(const std::string &str) {
        std::vector<ValueType> symbols;
        symbols.resize(str.length());
        std::transform(str.begin(), str.end(), symbols.begin(), [] (char v) {
            return static_cast<unsigned char>(v);
        });

        return symbols;
    }
}

#endif //RECOIL_EXAMPLE_FILE_H
