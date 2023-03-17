#ifndef RECOIL_BITS_READWRITER_H
#define RECOIL_BITS_READWRITER_H

#include <vector>
#include <span>
#include <cstdint>
#include <cmath>
#include <concepts>
#include <bit>

namespace Recoil {
    namespace {
        consteval uint8_t floorlog2(uint8_t x) {
            return x == 1 ? 0 : 1 + floorlog2(x >> 1);
        }

        consteval uint8_t ceillog2(uint8_t x) {
            return x == 1 ? 0 : floorlog2(x - 1) + 1;
        }
    }

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

        template<std::integral WriteDataType>
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

        template<std::integral WriteDataType>
        static uint8_t getActualLength(WriteDataType data) {
            if (data == 0) return 1;
            return sizeof(WriteDataType) * 8 - std::countl_zero(data);
        }
    private:
        uint8_t currentBitPosition = sizeof(BufferDataType) * 8;
    };

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

#endif //RECOIL_BITS_READWRITER_H
