#include "rocc.h"
#include "processor.h"
#include <cstdint>

class xqmac8_t : public rocc_t
{
 public:
  const char* name() const override
  {
    return "xqmac8";
  }

  reg_t custom0(processor_t* p, rocc_insn_t insn,
                reg_t xs1, reg_t xs2) override
  {
    // RV64、三個操作數旗標皆為 1
    // funct7 = 0 為 MAC，funct7 = 1 為 requant
    if (p->get_xlen() != 64 ||
        !insn.xd || !insn.xs1 || !insn.xs2 ||
        insn.funct > 1)
    {
      illegal_instruction(*p);
      return 0;
    }

    // 新增的 requantization 指令
    if (insn.funct == 1)
    {
      // rs2 高 18 bits 保留，必須為 0
      if ((xs2 >> 46) != 0)
      {
        illegal_instruction(*p);
        return 0;
      }

      int64_t x = signed32(static_cast<uint32_t>(xs1));

      int64_t multiplier =
          signed32(static_cast<uint32_t>(xs2));

      unsigned shift =
          static_cast<unsigned>((xs2 >> 32) & 0x3f);

      int64_t zero_point =
          static_cast<int64_t>((xs2 >> 38) & 0xff);

      if (zero_point >= 128)
        zero_point -= 256;

      // 兩個 INT32 相乘，使用 INT64 保存完整結果
      int64_t product = x * multiplier;
      bool negative = product < 0;

      uint64_t magnitude = negative
          ? static_cast<uint64_t>(-product)
          : static_cast<uint64_t>(product);

      // 除以 2^shift，恰好一半時朝遠離零方向捨入
      if (shift != 0)
      {
        uint64_t half = uint64_t{1} << (shift - 1);
        magnitude = (magnitude + half) >> shift;
      }

      int64_t scaled = static_cast<int64_t>(magnitude);

      if (negative)
        scaled = -scaled;

      scaled += zero_point;

      // 飽和至 INT8 範圍
      if (scaled > 127)
        scaled = 127;
      else if (scaled < -128)
        scaled = -128;

      return static_cast<reg_t>(scaled);
    }

    // 原本的 8 組 INT8 MAC
    int32_t dot = 0;

    for (unsigned i = 0; i < 8; i++)
    {
      int32_t a = (xs1 >> (8 * i)) & 0xff;
      int32_t b = (xs2 >> (8 * i)) & 0xff;

      if (a >= 128)
        a -= 256;
      if (b >= 128)
        b -= 256;

      dot += a * b;
    }

    uint32_t acc =
        static_cast<uint32_t>(p->get_state()->XPR[insn.rd]);

    uint32_t result = acc + static_cast<uint32_t>(dot);

    return static_cast<reg_t>(signed32(result));
  }

 private:
  static int64_t signed32(uint32_t bits)
  {
    int64_t value = static_cast<int64_t>(bits);

    if (bits & 0x80000000u)
      value -= (int64_t{1} << 32);

    return value;
  }
};

REGISTER_EXTENSION(xqmac8, []() { return new xqmac8_t; })