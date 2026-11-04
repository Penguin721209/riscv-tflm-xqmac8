# Packed INT8 MAC and Requantization Extensions for RV64IMC

**Midterm report — ISA design and Spike evaluation**  
Repository: https://github.com/Penguin721209/riscv-tflm-xqmac8

## Abstract

This project evaluates two custom instructions for bare-metal quantized neural-network kernels on RV64IMC. Xqmac8 combines eight signed INT8 products with an INT32 accumulator. Xrequant combines fixed-point scaling, rounding, zero-point addition and INT8 saturation. A synthetic MLP with two fully connected layers provides the workload. Existing C implementation measurements show a reduction from 412,673,634 to 61,516,540 kernel-stage instructions, or 85.093%, including input and weight preparation. Recounting supplied traces for a representative GEMM shows how packed computation reduces arithmetic, load and control instructions. Separate TFLM integration results report 88.435% fewer instructions during one Invoke, with additional arena storage and initialization work. Fresh directed and randomized ISA tests returned exit code zero in the user's WSL environment. These results quantify ISA-level instruction savings. RTL cycles, timing and area remain future work.

## 1. Midterm scope and workload

The assignment explicitly identifies ISA and Spike custom instructions as the midterm evaluation focus. This report covers that stage. It does not claim completion of the final RTL evaluation or every final-project coverage requirement.

The workload uses synthetic INT8 activations and weights. Its dimensions resemble the MLP portion of DeiT-Tiny: 197 tokens, input dimension 192 and hidden dimension 768. FC1 produces a 197 × 768 tensor. Requantization and a fixed GELU lookup table feed FC2, which produces a 197 × 192 tensor followed by requantization. Each FC performs 29,048,832 MACs, totaling 58,097,664 MACs.

This workload is an MLP submodule. It is neither a trained full DeiT model nor a complete Transformer block, and no application-level accuracy result is claimed. The execution target is bare-metal RV64IMC/LP64 without a floating-point or vector hardware requirement.

## 2. Instruction design

### 2.1 Xqmac8

Xqmac8 uses the custom-0 opcode 0x0b, funct3=7 and funct7=0. Each 64-bit source contains eight signed INT8 lanes. Lane i occupies bits [8i+7:8i]. The destination is also the incoming accumulator.

```text
s = signed32(old_rd) + sum(signed8(rs1[i]) * signed8(rs2[i]), i=0..7)
rd = sign_extend_64(low_32_bits(s))
```

The operation defines INT32 wraparound followed by sign extension to RV64. It does not saturate the accumulator. Software must account for this behavior when selecting workload ranges. The inline wrapper uses a read/write destination constraint, `"+r"`, to preserve the old accumulator value.

```text
.insn r 0x0b, 7, 0, rd, rs1, rs2
```

### 2.2 Xrequant

Xrequant shares opcode 0x0b and funct3=7, with funct7=1. Software passes a signed INT32 input in rs1 and packs the scaling parameters into rs2.

| Parameter | Location in rs2 | Meaning |
|---|---|---|
| multiplier | [31:0] | Signed INT32 |
| shift | [37:32] | Unsigned value, 0 through 63 |
| zero point | [45:38] | Signed INT8 |

The software contract multiplies the signed input by the signed multiplier using an INT64 product. For a nonzero shift, it rounds division by 2^shift to the nearest integer, with ties away from zero. It then adds the zero point, clamps to [-128,127] and sign-extends the result to RV64. Shift zero bypasses rounding.

```text
.insn r 0x0b, 7, 1, rd, rs1, rs2
```

These contracts were recovered from the supplied wrappers, C references and tests. The evidence archive does not contain the Spike extension C++ implementation or shared library. The final submission must retain that implementation and its build identity so a reviewer can inspect the implementation directly.

## 3. Correctness verification

The original archive contains test source and ELF files, but not fresh exit-code records for the full directed/random suites. The user rebuilt the tests from `xqmac8/main.c` and `xrequant/main.c` using GCC 14.2.0 in WSL, targeting RV64IMC/LP64 with -O2 and a bare-metal linker script. Both tests ran with `/home/win/riscv/bin/spike` and `/home/win/riscv-isa-sim/build/libxqmac8.so`.

| Test | Directed cases | Deterministic random cases | User-reported exit code |
|---|---:|---:|---:|
| Xqmac8 | 7 | 1,000 | 0 |
| Xrequant | 19 | 2,048 | 0 |

MAC cases cover signed results, sign extension, INT8 extremes and INT32 wraparound. Requantization cases cover signed rounding, saturation, zero points, signed multipliers and shifts from 0 through 63. The final runner used Python subprocess execution with standard input disconnected, a separate process session and a ten-second limit. The exact cause of earlier shell-run timeouts was not established. Adding WFI did not resolve them and is not part of the successful original-startup test build.

These fresh results are transcribed from the user's terminal output in this conversation. The generated log files remain on the user's machine at `midterm_checks/xqmac8_fresh.run.log` and `midterm_checks/xrequant_fresh.run.log`; this report does not include independently received copies of those files or fresh ELF/plugin hashes. An older MAC trace has 32 instruction-byte mismatches against its accompanying ELF and is excluded from quantitative analysis.

## 4. Measurement method and provenance

Two evidence levels are kept separate. Large-MLP stage counts come from existing CSV/text summaries in `midterm_evidence.tar.gz`. Their stage sums were checked, but no complete raw MLP trace was supplied for an independent recount. Representative GEMM instruction mix was independently recounted from the supplied traces, using matching ELF function ranges and instruction bytes.

Function-level counts include prologues and epilogues. They exclude startup, the caller and reference checks unless explicitly named. Compressed instructions count as one dynamic instruction. Branches/jumps include calls and returns. Custom instructions that Spike prints as `unknown` are identified from their encoding and counted separately. Load counts measure instructions, not transferred bytes, cache misses or stall cycles.

Archive SHA-256: `c0469d08b52383c045ceade637eb2349f2d9031369781df0e0eba5fb32d71d8e`.

## 5. Four-configuration C MLP results

| Configuration | Change | Kernel-stage instructions | Reduction from B0 |
|---|---|---:|---:|
| B0 | Scalar baseline | 412,673,634 | — |
| B1 | Software optimization and weight transpose | 284,002,602 | 31.180% |
| X1 | Packed Xqmac8, software requantization | 62,791,772 | 84.784% |
| X2 | Xqmac8 plus Xrequant | 61,516,540 | 85.093% |

The total sums input packing, weight preprocessing, FC, requantization and GELU functions. It includes per-run preparation in this C experiment, while excluding startup, answer verification and other caller instructions. B0/X2 is a 6.708× instruction-count ratio, not a measured cycle or wall-clock speedup.

| Stage | B0 | B1 | X1 | X2 |
|---|---:|---:|---:|---:|
| Input packing | 0 | 0 | 1,894,392 | 1,894,392 |
| Weight preprocessing | 0 | 1,481,296 | 3,030,576 | 3,030,576 |
| FC1 and FC2 | 408,390,118 | 278,237,790 | 53,583,288 | 53,583,288 |
| Requantization | 3,073,139 | 3,073,139 | 3,073,139 | 1,797,907 |
| GELU | 1,210,377 | 1,210,377 | 1,210,377 | 1,210,377 |

Adding Xrequant reduces its stage by 41.496% but reduces the X1 total by only 2.031%. FC remains dominant. X2 records 7,262,208 Xqmac8 instructions, consistent with 58,097,664 / 8, and 189,120 Xrequant instructions, consistent with 197 × (768 + 192).

Raw ELF .text sizes are 904, 1,160, 1,308 and 1,276 bytes for B0, B1, X1 and X2. X2 .text is 41.15% larger than B0. Its .bss increases from 907,784 to 1,206,536 bytes. These ELF sizes include verification-related assets and buffers and are distinct from TFLM arena measurements.

## 6. Representative GEMM instruction mix and bottlenecks

The GEMM multiplies 16 × 64 by 64 × 32, producing 512 outputs and performing 32,768 MACs.

| Dynamic instruction category | Baseline | Software optimized | Xqmac8 |
|---|---:|---:|---:|
| Loads | 65,540 | 65,548 | 8,196 |
| Stores | 516 | 524 | 516 |
| Multiply/divide | 32,768 | 32,768 | 0 |
| Branches and jumps | 33,330 | 9,269 | 5,674 |
| Integer ALU / other | 101,004 | 51,910 | 12,011 |
| Xqmac8 | 0 | 0 | 4,096 |
| Total | 233,158 | 160,019 | 30,493 |

Three measured overheads motivate the design. First, scalar multiplication requires 32,768 multiply instructions, compared with 4,096 packed MAC instructions for the same work. Second, operand loads fall from 65,540 to 8,196 when eight INT8 values share each packed operand. Third, loop, indexing and control operations are substantial: the baseline has 101,004 integer-ALU/other instructions and 33,330 branches/jumps. Software optimization alone reduces them to 51,910 and 9,269.

The GEMM-only instruction ratio is 7.646×. Including weight transpose or matrix packing changes the totals to 233,158, 170,586 and 52,172 instructions. The preparation-inclusive custom ratio is therefore 4.469×, with 77.624% fewer instructions. Requantizing 512 outputs separately takes 8,444 software instructions versus 4,877 custom instructions, a 42.243% reduction, or 16.492 versus 9.525 instructions per output.

| GEMM-only metric | Baseline | Software optimized | Xqmac8 |
|---|---:|---:|---:|
| MACs / dynamic instruction | 0.141 | 0.205 | 1.075 |
| Load + store instructions / MAC | 2.016 | 2.016 | 0.266 |
| Xqmac8 share of instructions | 0% | 0% | 13.432% |

These are instruction-overhead findings for this GEMM. They do not establish cache or bandwidth bottlenecks and must not be described as the complete MLP instruction mix. In the large C MLP, FC accounts for 98.962% of baseline kernel-stage instructions and 87.104% after X2. Packing accounts for 8.006% after X2, motivating evaluation of reuse and amortization.

## 7. Separate TFLM integration experiment

The earlier TFLM experiment replaces the FC dot product with Xqmac8 while retaining TFLM's Q31 requantization and the fixed GELU LUT. Constant weights are packed during preparation. Each activation row is packed once and reused across output channels. Bias correction incorporates the input zero point and weight sum. Xrequant is not used in this experiment because its rounding contract differs from TFLM's.

The existing presentation reports 88.435% fewer dynamic instructions during one complete Invoke, a reference/custom instruction-count ratio of 8.647×. This interval includes FC1, GELU and FC2 for all 197 tokens. It excludes allocation and weight preparation, input copying, validation and console output. Performance builds disable scalar checking and custom counters.

| Metric from the existing TFLM presentation | Reference | Xqmac8 |
|---|---:|---:|
| Preparation instructions | 27,844 | 3,999,425 |
| Arena used, bytes | 303,664 | 606,400 |

The additional arena cost is 302,736 bytes. Previously reported framework comparisons show exact agreement for 37,824 final outputs, 189,120 FC accumulators, and 151,296 values at each of Q1 and GELU. Under the older assignment rounding reference, 6,665 final values differ. That difference is not a mismatch between the two TFLM variants.

These figures are retained from the existing presentation and adapter source, not newly rerun performance measurements. They are separate from the C four-configuration experiment because their implementation, reference rounding and preparation boundaries differ. The two percentage reductions must not be combined into an incremental comparison.

## 8. Limitations and next work

The representative GEMM exceeds the project's alternative target of a twofold reduction in dynamic instructions, including preparation. This does not by itself fulfill every workload or final-project requirement. Full-model inference or a complete Transformer/CNN block remains outside the demonstrated scope. Whole-MLP instruction mix is also not present in the supplied raw traces.

Before repository submission, retain the fresh test logs, exact ISA extension source and build version, test compiler flags and binary identities alongside the existing workload/results. This report does not claim that all of those files have already been added to the public repository. Instructor acceptance of the current midterm workload scope and report format has not been verified.

The final phase should implement representative instructions in CV-Wally RTL and measure cycles, CPI, timing and area. Spike instruction savings alone cannot establish hardware speedup. Packing reuse and the remaining FC work provide concrete directions for subsequent optimization.

## Evidence references

1. Assignment: `rv64imc_quantized_nn_term_project(3).md`, especially sections 2–5 and 9.
2. Uploaded archive: `model_work/results/mlp_all_versions.csv` and associated count files.
3. Matched representative traces and ELFs: `model_work/target/trace_{baseline,opt,xqmac8_requant}_16x64x32.log` and corresponding `gemm_*.elf` files.
4. ISA test sources: `xqmac8/main.c`, `xrequant/main.c`, and their startup/linker files.
5. User-reported fresh terminal results: `xqmac8 exit_code=0`, `xrequant exit_code=0`, 4 November 2026.
6. Existing `RV64IMC_TFLM_Xqmac8_6min_EN_v2.pptx` and local TFLM FC adapter source. TFLM performance figures are retained evidence, not a fresh rerun.
