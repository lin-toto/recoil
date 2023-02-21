#ifndef RECOIL_EXAMPLE_FILE_H
#define RECOIL_EXAMPLE_FILE_H

#include <string>
#include <fstream>

namespace Recoil::Examples {
    std::string readFile(const std::string &name, std::ios_base::openmode mode = std::ios_base::in);
}

#endif //RECOIL_EXAMPLE_FILE_H
