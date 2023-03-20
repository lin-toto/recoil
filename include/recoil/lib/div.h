#ifndef RECOIL_DIV_H
#define RECOIL_DIV_H

namespace Recoil {
    template<class T>
    T saveDiv(T a, T b) {
        return (a + b - 1) / b;
    }
}

#endif //RECOIL_DIV_H
