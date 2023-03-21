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
        explicit BitsReader(std::span<const BufferDataType> buf): buf(buf), it(buf.begin()), curr(*it) {}

        template<std::integral ReadDataType>
        ReadDataType read(uint8_t lengthGranularity = 1) {
            auto actualLength = readLength<ReadDataType>(lengthGranularity);
            return readData<ReadDataType>(actualLength);
        }

        template<std::integral ReadDataType>
        uint8_t readLength(uint8_t lengthGranularity = 1) {
            constexpr auto sizeBits = ceillog2(sizeof(ReadDataType) * 8);
            return (readData<uint8_t>(sizeBits - lengthGranularity + 1) + 1) << (lengthGranularity - 1);
        }

        template<std::unsigned_integral ReadDataType>
        ReadDataType readData(uint8_t actualLength) {
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

        template<std::signed_integral ReadDataType>
        ReadDataType readData(uint8_t actualLength) {
            auto signBit = readData<uint8_t>(1) == 0 ? 1 : -1;
            return signBit * readData<std::make_unsigned_t<ReadDataType>>(actualLength);
        }

        [[nodiscard]] inline size_t currentIteratorPosition() const { return it - buf.begin(); }
    private:
        std::span<const BufferDataType> buf;
        typename std::span<const BufferDataType>::iterator it;
        uint8_t currentBitPosition = sizeof(BufferDataType) * 8;
        BufferDataType curr;
    };
}

#endif //RECOIL_BITS_READER_H
