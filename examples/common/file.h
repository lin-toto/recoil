#ifndef RECOIL_EXAMPLE_FILE_H
#define RECOIL_EXAMPLE_FILE_H

#include <string>
#include <fstream>
#include <iterator>
#include <vector>
#include <concepts>
#include <span>
#include <algorithm>

namespace Recoil::Examples {
    std::string readFile(const std::string &name, std::ios_base::openmode mode = std::ios_base::in);

    template<typename T>
    void writeSpanToFile(const std::string &name, std::span<T> span) {
        std::ofstream fout(name, std::ios::out | std::ios::binary);
        fout.write(reinterpret_cast<const char *>(span.data()), span.size_bytes());
        fout.close();
    }

    template<typename T>
    std::vector<T> readVectorFromFile(const std::string &name) {
        std::ifstream fin(name, std::ios::binary);
        if (!fin.good()) [[unlikely]] throw std::runtime_error("Error reading vector file");
        fin.unsetf(std::ios::skipws);

        fin.seekg(0, std::ios::end);
        auto fileSize = fin.tellg();
        fin.seekg(0, std::ios::beg);

        std::vector<T> vec;
        vec.resize(fileSize / sizeof(T));

        fin.read(reinterpret_cast<char*>(vec.data()), fileSize);

        return vec;
    }

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
