#ifndef RECOIL_BITS_READWRITER_H
#define RECOIL_BITS_READWRITER_H

#include <vector>
#include <cstdint>
#include <cmath>

namespace {
    constexpr uint8_t floorlog2(uint8_t x) {
        return x == 1 ? 0 : 1 + floorlog2(x >> 1);
    }

    constexpr uint8_t ceillog2(uint8_t x) {
        return x == 1 ? 0 : floorlog2(x - 1) + 1;
    }
}

template<typename BufferDataType>
class BitsReader {
    static_assert(std::is_unsigned_v<BufferDataType>, "BufferDataType must be unsigned");
public:
    explicit BitsReader(const std::vector<BufferDataType>& buf): buf(buf), it(buf.cbegin()) {}

    template<typename ReadDataType>
    ReadDataType read() {
        static_assert(std::is_unsigned_v<ReadDataType>, "ReadDataType must be unsigned");

        constexpr auto sizeBits = ceillog2(sizeof(ReadDataType) * 8);

        auto actualLength = readBits<uint8_t>(sizeBits) + 1;
        return readBits<ReadDataType>(actualLength);
    }

    [[nodiscard]] inline size_t currentIteratorPosition() const { return it - buf.cbegin(); }
private:
    const std::vector<BufferDataType>& buf;
    typename std::vector<BufferDataType>::const_iterator it;
    uint8_t currentBitPosition = sizeof(BufferDataType) * 8;
    BufferDataType curr;

    template<typename ReadDataType>
    ReadDataType readBits(uint8_t actualLength) {
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
};

template<typename BufferDataType>
class BitsWriter {
    static_assert(std::is_unsigned_v<BufferDataType>, "BufferDataType must be unsigned");
public:
    std::vector<BufferDataType> buf;

    template<typename WriteDataType>
    void write(WriteDataType data) {
        static_assert(std::is_unsigned_v<WriteDataType>, "WriteDataType must be unsigned");

        constexpr auto sizeBits = ceillog2(sizeof(WriteDataType) * 8);
        uint8_t actualLength = getActualLength(data);
        writeBits(actualLength - 1, sizeBits);
        writeBits(data, actualLength);
    }
private:
    uint8_t currentBitPosition = sizeof(BufferDataType) * 8;

    template<typename WriteDataType>
    uint8_t getActualLength(WriteDataType data) const {
        if (data == 0) return 1;

        if constexpr (sizeof(WriteDataType) <= 4) {
            return sizeof(WriteDataType) * 8 - __builtin_clz(data);
        } else {
            return sizeof(WriteDataType) * 8 - __builtin_clzll(data);
        }
    }

    template<typename WriteDataType>
    void writeBits(WriteDataType data, uint8_t actualLength) {
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
};


#endif //RECOIL_BITS_READWRITER_H
