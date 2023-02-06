#ifndef RECOIL_RANS_DECODER_AVX2_32X8N_H
#define RECOIL_RANS_DECODER_AVX2_32X8N_H

#include "recoil/rans_decoder.h"
#include "recoil/lib/simd/avx2_permute.h"
#include <x86intrin.h>
#include <vector>
#include <span>
#include <bit>

namespace Recoil {
    template<BitCountType ProbBits, uint32_t RenormLowerBound, size_t NInterleaved>
    class RansDecoder_AVX2_32x8n : public RansDecoder<uint32_t, uint16_t, ProbBits, RenormLowerBound, 16, NInterleaved> {
        using MyRansDecoder = RansDecoder<uint32_t, uint16_t, ProbBits, RenormLowerBound, 16, NInterleaved>;

        static_assert(NInterleaved % 8 == 0, "AVX2 32x8n decoder must work on 8n streams");
        static_assert(MyRansDecoder::MyRans::oneShotRenorm, "Only one shot renorm decoders are supported");

        static constexpr uint8_t WriteBits = 16;
        static constexpr size_t SimdWidth = 256;
        static constexpr size_t RansPerBatch = SimdWidth / 32;
        static constexpr size_t RansNBatch = NInterleaved / RansPerBatch;
    public:
        using MyRansDecoder::MyRansDecoder;

        std::vector<ValueType> decode(const Cdf cdf) {
            std::vector<ValueType> result;
            // TODO

            return result;
        }

        std::vector<ValueType> decode(const Cdf cdf, const size_t count) {
            std::vector<ValueType> result;
            result.reserve(count);
            size_t completedCount = 0;

            {
                // Step 1: decode initial unaligned parts so that ransIt is now at beginning
                auto unalignedCount = std::min(static_cast<size_t>(this->rans.end() - this->ransIt), count);
                if (unalignedCount != NInterleaved) { // If equal to NInterleaved, it is at beginning; no action needed
                    auto initialUnalignedResult = MyRansDecoder::decode(cdf, unalignedCount);
                    completedCount += unalignedCount;
                    result.insert(result.end(), initialUnalignedResult.begin(), initialUnalignedResult.end());
                }

                if (completedCount == count) [[unlikely]] return result;
            }

            {
                // Step 2: do simd rANS decoding
                std::array<Cdf, RansPerBatch> cdfs = {cdf, cdf, cdf, cdf, cdf, cdf, cdf, cdf};
                auto ransSimds = createRansSimds();

                for (; completedCount + NInterleaved <= count; completedCount += NInterleaved) {
                    for (auto b = 0; b < RansNBatch; b++) {
                        auto& ransSimd = ransSimds[b];
                        auto probabilities = getProbabilities(ransSimd);
                        auto [bypass, symbols, starts, frequencies] = getSymbolsAndStartsAndFrequencies(probabilities, cdfs);
                        // TODO: if probability is a bypass sentinel, handle as bypass symbol

                        renorm(ransSimd, probabilities, starts, frequencies);
                    }
                }

                writeBackRansSimds(ransSimds);
            }

            {
                // Step 3: decode final unaligned parts
                if (completedCount != count) {
                    auto finalUnalignedResult = MyRansDecoder::decode(cdf, count - completedCount);
                    result.insert(result.end(), finalUnalignedResult.begin(), finalUnalignedResult.end());
                }
            }

            return result;
        }

        std::vector<ValueType> decode(const std::span<Cdf> cdfs) {
            std::vector<ValueType> result;
            result.reserve(cdfs.size());
            // TODO

            return result;
        }

    protected:
        using u32x8 = __m256i;

        static u32x8 toSimd(const std::array<uint32_t, RansPerBatch> &val) {
            return _mm256_load_si256(reinterpret_cast<const u32x8*>(val.begin()));
        }

        static std::array<uint32_t, RansPerBatch> fromSimd(const u32x8 simd) {
            std::array<uint32_t, RansPerBatch> val;
            _mm256_store_si256(reinterpret_cast<u32x8*>(val.begin()), simd);
            return val;
        }

        std::array<u32x8, RansNBatch> createRansSimds() {
            std::array<u32x8, RansNBatch> ransSimds{};

            for (auto b = 0; b < RansNBatch; b++) {
                alignas(32) std::array<uint32_t, RansPerBatch> rans{};
                for (auto i = 0; i < RansPerBatch; i++) {
                    rans[i] = this->rans[b * RansPerBatch + i].state;
                }
                ransSimds[b] = toSimd(rans);
            }

            return ransSimds;
        };

        void writeBackRansSimds(std::array<u32x8, RansNBatch> ransSimds) {
            for (auto b = 0; b < RansNBatch; b++) {
                auto rans = fromSimd(ransSimds[b]);
                for (auto i = 0; i < RansPerBatch; i++) {
                    this->rans[b * RansPerBatch + i].state = rans[i];
                }
            }
        };

        u32x8 getProbabilities(const u32x8 ransSimd) {
            static const u32x8 probabilityMask = _mm256_set1_epi32((1 << ProbBits) - 1);
            return _mm256_and_epi32(ransSimd, probabilityMask);
        }

        auto getSymbolsAndStartsAndFrequencies(const u32x8 probabilitiesSimd, const std::array<Cdf, RansPerBatch> &cdfs) {
            std::array<bool, RansPerBatch> bypass{};
            alignas(32) std::array<uint32_t, RansPerBatch> symbols{}, starts{}, frequencies{};
            auto probabilities = fromSimd(probabilitiesSimd);

            for (size_t i = 0; i < RansPerBatch; i++) {
                auto symbol = cdfs[i].findValue(probabilities[i]);
                if (symbol.has_value()) [[likely]] {
                    bypass[i] = false;
                    symbols[i] = symbol.value();
                    std::tie(starts[i], frequencies[i]) = cdfs[i].getStartAndFrequency(symbols[i]).value();
                } else {
                    bypass[i] = true;
                }
            }

            return std::make_tuple(bypass, toSimd(symbols), toSimd(starts), toSimd(frequencies));
        }

        void renorm(u32x8 &ransSimd, const u32x8 lastProbabilities, const u32x8 lastStarts, const u32x8 lastFrequencies) {
            // Advance Symbols
            ransSimd = _mm256_mullo_epi32(_mm256_srli_epi32(ransSimd, ProbBits), lastFrequencies);
            ransSimd = _mm256_add_epi64(ransSimd, lastProbabilities);
            ransSimd = _mm256_sub_epi64(ransSimd, lastStarts);

            // Check renormalization flags
            u32x8 renormMaskSimd = _mm256_cmpgt_epi32(
                    _mm256_set1_epi32(RenormLowerBound - 0x80000000),
                    _mm256_xor_si256(ransSimd, _mm256_set1_epi32(0x80000000)));

            auto renormMask = _mm256_movemask_epi8(renormMaskSimd);
            if (renormMask) {
                /*            <--------------------------vv
                 * Bitstream: 01 23 45 67 89 ab cd ef 01 23 45 67 89 ab cd ef
                 * RenormCount: 2 (Expected to read:  ^^ ^^)
                 * Read from ptr - 7 then set ptr -= 2
                 */

                auto renormCount = std::popcount(static_cast<unsigned int>(renormMask));
                auto bitstreamPtr = &(*this->bitstreamReverseIt) - RansPerBatch + 1;

                // Use _mm_loadu_si128 because it does not require memory alignment.
                u32x8 nextBitstream = _mm256_cvtepu16_epi32(_mm_loadu_si128(bitstreamPtr));
                u32x8 nextStates = _mm256_permutevar8x32_epi32(nextBitstream, AVX2Permute::getPermuteOffsets(renormMask));

                u32x8 renormedRans = _mm256_or_si256(_mm256_slli_epi32(ransSimd, WriteBits), nextStates);
                ransSimd = _mm256_blendv_epi8(ransSimd, renormedRans, renormMaskSimd);

                this->bitstreamReverseIt += renormCount;
            }
        }
    };

    template<BitCountType ProbBits, uint32_t RenormLowerBound, size_t nInterleaved>
    RansDecoder_AVX2_32x8n(std::span<uint16_t>,
                           std::array<Rans<uint32_t, uint16_t, ProbBits, RenormLowerBound, 16>, nInterleaved>)
            -> RansDecoder_AVX2_32x8n<ProbBits, RenormLowerBound, nInterleaved>;
}

#endif //RECOIL_RANS_DECODER_AVX2_32X8N_H
