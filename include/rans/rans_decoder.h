#ifndef RECOIL_RANS_DECODER_H
#define RECOIL_RANS_DECODER_H

#include "rans/lib/cdf.h"
#include "rans.h"
#include <span>
#include <array>

namespace Recoil {
    template<typename RansStateType, typename RansBitstreamType,
            BitCountType ProbBits, RansStateType RenormLowerBound, BitCountType WriteBits,
            size_t nInterleaved>
    class RansDecoder {
        using MyRans = Rans<RansStateType, RansBitstreamType, ProbBits, RenormLowerBound, WriteBits>;
    public:
        RansDecoder(std::span<RansBitstreamType> bitstream, std::array<MyRans, nInterleaved> rans)
                : rans(std::move(rans)), bitstream(bitstream),
                  ransIt(this->rans.begin()), bitstreamReverseIt(this->bitstream.rbegin()) {}

        /*
         * Decode the values with a single shared CDF.
         * If the count is 0, then all available symbols are decoded.
         */
        std::vector<ValueType> decode(Cdf cdf, size_t count = 0) {

        }

        /*
         * Decode the values with independent CDF for each symbol.
         */
        std::vector<ValueType> decode(std::span<Cdf> cdfs) {

        }


    protected:
        std::array<MyRans, nInterleaved> rans;
        std::span<RansBitstreamType> bitstream;

        typename std::array<MyRans, nInterleaved>::iterator ransIt;
        typename std::span<RansBitstreamType>::reverse_iterator bitstreamReverseIt;

        ValueType decodeSymbol(MyRans &decoder, Cdf cdf) {

        }
    };
}

#endif //RECOIL_RANS_DECODER_H
