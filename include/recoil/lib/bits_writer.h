#ifndef RECOIL_BITS_WRITER_H
#define RECOIL_BITS_WRITER_H

#include "recoil/lib/math.h"

#include <vector>
#include <span>
#include <cstdint>
#include <cmath>
#include <concepts>
#include <bit>

namespace Recoil {
    template<std::unsigned_integral BufferDataType>
    class BitsWriter {
    public:
        std::vector<BufferDataType> buf;

        template<std::integral WriteDataType>
        inline void write(WriteDataType data, uint8_t lengthGranularity = 1) {
            uint8_t actualLength = getActualLength(data);
            actualLength = writeLength<WriteDataType>(actualLength, lengthGranularity);
            writeData(data, actualLength);
        }

        template<std::integral WriteDataType>
        inline uint8_t writeLength(uint8_t length, uint8_t lengthGranularity = 1) {
            constexpr auto sizeBits = ceillog2(sizeof(WriteDataType) * 8);
            write((length - 1) >> (lengthGranularity - 1), sizeBits - lengthGranularity + 1);
            return (((length - 1) >> (lengthGranularity - 1)) + 1) << (lengthGranularity - 1);
        }

        template<std::unsigned_integral WriteDataType>
        inline void writeData(WriteDataType data, uint8_t actualLength) {
            if (currentBitPosition == sizeof(BufferDataType) * 8) {
                buf.push_back(0);
            }

            uint8_t remainingLength = actualLength;
            while (remainingLength > 0) {
                uint8_t len = std::min(remainingLength, currentBitPosition);
                WriteDataType mask = ((1 << len) - 1) << (remainingLength - len);
                buf.back() |= ((data & mask) >> (remainingLength - len)) << (currentBitPosition - len);

                remainingLength -= len;
                currentBitPosition -= len;

                if (currentBitPosition == 0) {
                    currentBitPosition = sizeof(BufferDataType) * 8;
                    buf.push_back(0);
                }
            }
        }

        template<std::signed_integral WriteDataType>
        inline void writeData(WriteDataType data, uint8_t actualLength) {
            writeData(static_cast<uint8_t>(data < 0 ? 1 : 0), 1);
            writeData(static_cast<std::make_unsigned_t<WriteDataType>>(std::abs(data)), actualLength);
        }

        template<std::integral WriteDataType>
        static uint8_t getActualLength(WriteDataType data) {
            if (data == 0) return 1;
            if (data < 0) data = std::abs(data);
            return sizeof(WriteDataType) * 8 - std::countl_zero(static_cast<std::make_unsigned_t<WriteDataType>>(data));
        }

        void reset() {
            buf.clear();
            currentBitPosition = sizeof(BufferDataType) * 8;
        }
    private:
        uint8_t currentBitPosition = sizeof(BufferDataType) * 8;
    };
}

#endif //RECOIL_BITS_WRITER_H
