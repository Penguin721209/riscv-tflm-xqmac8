# RV64IMC INT8 MLP Acceleration with Xqmac8

Integrate a synthetic INT8 MLP into TensorFlow Lite for Microcontrollers
(TFLM / LiteRT Micro) and accelerate FullyConnected kernels using a
custom eight-lane INT8 multiply-accumulate instruction in Spike.

## Model

- Shape: 197 × 192 → 197 × 768 → 197 × 192
- Graph: FullyConnected → custom GELU LUT → FullyConnected
- Scalar MACs: 58,097,664 per complete inference
- INT8 activation/weight scale: 1/16; zero point: 0
- INT32 bias scale: 1/256
- Synthetic test data; this is not a complete pretrained DeiT model.

## Results

| Metric | TFLM reference | Prepacked Xqmac8 |
|---|---:|---:|
| Inference instructions | 535,433,403 | 61,920,684 |
| AllocateTensors / preparation instructions | 27,844 | 3,999,425 |
| Arena used (bytes) | 303,664 | 606,400 |
| Xqmac8 instructions per inference | 0 | 7,262,208 |

Inference instruction reduction: **88.435%**.
Reference/custom instruction-count ratio: **8.647085**.

These are Spike dynamic instruction counts, not measured hardware
cycles or elapsed-time speedups. Preparation is measured separately.

## Correctness

- All 37,824 output values match between reference and Xqmac8.
- All 189,120 custom FC accumulators pass scalar verification.
- Intermediate Q1 and GELU results match the framework golden data.
- The linked framework uses double rounding. Its final output differs
  from the original assignment's ties-away-from-zero reference at
  6,665 positions. This expected difference is reported separately.

## Repository structure

- `src/`: application, custom FC/GELU kernels, runtime and build scripts
- `src/data/`: original synthetic input, weights, biases and reference data
- `spike/`: custom extension source, build patch and base commit
- `framework/`: custom TFLM RV64IMC build target
- `results/`: measured results and correctness log
- `slides/`: presentation materials (to be added)

## Tested environment

- Linux / WSL
- xPack GNU RISC-V Embedded GCC 15.2.0
- Target: RV64IMC, LP64, bare metal
- Spike base commit: `5a760667bf0d0ab893173b9eeff9c109c8d7dd9a`
- Spike memory: 16 MiB; one hart
- TFLM source archive SHA-256:
  `c94b3cc1e04069bc8609a17be76c9e3c7008142caceb967b87b73f7df6fd71b8`

## Build and run with the prepared environment

Prerequisites: the matching TFLM static library and dependencies,
xPack toolchain, Spike, and the compiled Xqmac8 extension.

From the repository root:

```bash
export TFLM_ROOT="$HOME/riscv_nn_project/framework_vendor/tflite-micro-main"
export TFLM_TOOLCHAIN_BIN="$HOME/toolchains/xpack-15.2.0/bin"
export SPIKE="$HOME/riscv/bin/spike"
export XQMAC8_EXTLIB="$HOME/riscv-isa-sim/build/libxqmac8.so"

bash src/build.sh && python3 src/run.py
```

Expected final messages:

```text
TFLM_MLP_XQMAC8_EXACT_MATCH_37824
MLP_RUN_OK exit_code=0
```

## Instruction-count measurement

```bash
python3 src/measure.py --timeout 3600
cat src/build/summary.csv
```

The timeout is per executable. Full instruction tracing can take
considerable time. The measured inference region covers one complete
197-token Invoke plus marker overhead, excluding input copying,
verification and console output. There is no warmup.

The performance binaries disable scalar accumulator verification and
software MAC counters. Allocation and weight prepacking are outside
the inference region.

Submitted measurements are preserved under `results/`.
New build and measurement outputs are written under `src/build/`.

## Rebuild the bundled TFLM library

The exact framework snapshot and prepared dependency sources are bundled
under `vendor/`, with their original license files inside the archives.
An installed xPack RISC-V toolchain and native g++ are required.

From the repository root:

```bash
bash framework/build_tflm.sh
export TFLM_ROOT="$PWD/deps/tflite-micro-main"
bash src/build.sh && python3 src/run.py
```

## Rebuild Spike with the custom extension

Install the host build dependencies required by Spike, then:

```bash
git clone https://github.com/riscv-software-src/riscv-isa-sim.git deps/riscv-isa-sim
git -C deps/riscv-isa-sim checkout "$(cat spike/BASE_COMMIT)"
git -C deps/riscv-isa-sim apply "$PWD/spike/customext-build.patch"
cp spike/customext/xqmac8.cc deps/riscv-isa-sim/customext/

mkdir -p deps/riscv-isa-sim/build
(
  cd deps/riscv-isa-sim/build
  ../configure --prefix="$HOME/riscv-mlp"
  make -j2
)

export SPIKE="$PWD/deps/riscv-isa-sim/build/spike"
export XQMAC8_EXTLIB="$PWD/deps/riscv-isa-sim/build/libcustomext.so"
python3 src/run.py
```

The submitted measurements used the separately built `libxqmac8.so`.
The rebuild instructions use `libcustomext.so`, which includes the same
extension source through the provided build patch. Validate the rebuilt
environment with `src/run.py` before collecting new measurements.

## Presentation

Presentation slides will be added before submission.
