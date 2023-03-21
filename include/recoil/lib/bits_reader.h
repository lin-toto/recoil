#ifndef RECOIL_BITS_READER_H
#define RECOIL_BITS_READER_H

#include "recoil/lib/math.h"

#include <vector>
#include <span>
#include <cstdint>
#include <cmath>
#include <concepts>
#include <bit>

namespace Recoil {
    template<std::unsigned_integral BufferDataType>
    class BitsReader {
    public:
        explicit BitsReader(std::span<BufferDataType> buf): buf(buf), it(buf.cbegin()) {}

        template<std::integral ReadDataType>
        ReadDataType read(uint8_t lengthGranularity = 1) {
            auto actualLength = readLength<ReadDataType>(lengthGranularity);
            return readData<ReadDataType>(actualLength);
        }

        template<std::integral ReadDataType>
        uint8_t readLength(uint8_t lengthGranularity = 1) {
            constexpr auto sizeBits = ceillog2(sizeof(ReadDataType) * 8);
            return (read<uint8_t>(sizeBits - lengthGranularity + 1) + 1) << (lengthGranularity - 1);
        }

        template<std::integral ReadDataType>
        ReadDataType readData(uint8_t actualLength) {
            if (currentBitPosition == sizeof(BufferDataType) * 8) {
                curr = *it;
            }

            ReadDataType data = 0;
            while (actualLength > 0) {
                uint8_t len = std::min(actualLength, currentBitPosition);
                data <<= len;
                BufferDataType mask = ((1 << len) - 1) << (currentBitPosition - len);
                data |= (curr & mask) >> (currentBitPosition - len);

                actualLength -= len;
                currentBitPosition -= len;

                if (currentBitPosition == 0) {
                    currentBitPosition = sizeof(BufferDataType) * 8;
                    it++;
                    curr = *it;
                }
            }

            return data;
        }

        [[nodiscard]] inline size_t currentIteratorPosition() const { return it - buf.cbegin(); }
    private:
        const std::vector<BufferDataType>& buf;
        typename std::vector<BufferDataType>::const_iterator it;
        uint8_t currentBitPosition = sizeof(BufferDataType) * 8;
        BufferDataType curr;
    };
}

#endif //RECOIL_BITS_READER_H
