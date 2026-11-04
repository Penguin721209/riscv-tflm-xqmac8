#pragma once
#include <cstdint>

#if ENABLE_XQMAC8_COUNTERS
extern volatile uint64_t g_xqmac8_calls;
#endif

namespace xqmac8_prepacked {
inline void PackRow(const int8_t* src, int k, uint64_t* dst) {
    const int blocks = (k + 7) / 8;
    for (int block = 0; block < blocks; ++block) {
        uint64_t word = 0;
        for (int lane = 0; lane < 8 && block * 8 + lane < k; ++lane)
            word |= static_cast<uint64_t>(static_cast<uint8_t>(src[block * 8 + lane])) << (8 * lane);
        dst[block] = word;
    }
}

inline uint64_t MAC(uint64_t acc, uint64_t a, uint64_t b) {
#ifdef XQMAC8_SOFTWARE_MODEL
    uint32_t bits = static_cast<uint32_t>(acc);
    for (unsigned lane = 0; lane < 8; ++lane) {
        int av = static_cast<int>((a >> (lane * 8)) & 255);
        int bv = static_cast<int>((b >> (lane * 8)) & 255);
        if (av >= 128) av -= 256;
        if (bv >= 128) bv -= 256;
        bits += static_cast<uint32_t>(av * bv);
    }
    acc = bits;
#elif defined(__riscv) && __riscv_xlen == 64
    __asm__ volatile(".insn r 0x0b, 7, 0, %0, %1, %2"
                     : "+r"(acc) : "r"(a), "r"(b));
#else
#error "RV64 required; software model is for explicit host tests only."
#endif
#if ENABLE_XQMAC8_COUNTERS
    g_xqmac8_calls = g_xqmac8_calls + 1;
#endif
    return acc;
}

inline int64_t Dot(const uint64_t* a, const uint64_t* w, int blocks) {
    uint64_t acc = 0;
    for (int block = 0; block < blocks; ++block) acc = MAC(acc, a[block], w[block]);
    const uint32_t bits = static_cast<uint32_t>(acc);
    return (bits & UINT32_C(0x80000000))
        ? static_cast<int64_t>(bits) - INT64_C(4294967296) : static_cast<int64_t>(bits);
}
} // namespace xqmac8_prepacked
