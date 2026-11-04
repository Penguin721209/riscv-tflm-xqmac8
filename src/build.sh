#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TFLM_DIR="${TFLM_ROOT:-$APP_DIR/../tflite-micro-main}"
TC_DIR="${TFLM_TOOLCHAIN_BIN:-$HOME/toolchains/xpack-15.2.0/bin}"
OUT_DIR="$APP_DIR/build"
mkdir -p "$OUT_DIR"

build_all() {
    local cxx="$TC_DIR/riscv-none-elf-g++"
    local readelf="$TC_DIR/riscv-none-elf-readelf"
    local archive="$TFLM_DIR/gen/rv64imc_spike_rv64imc_default_gcc/lib/libtensorflow-microlite.a"
    [[ -x "$cxx" ]] || { echo "Missing compiler: $cxx"; return 1; }
    [[ -f "$archive" ]] || { echo "Missing TFLM library: $archive"; return 1; }
    [[ -f "$TFLM_DIR/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include/flatbuffers/flatbuffers.h" ]] || {
        echo "FlatBuffers headers missing; prepare the offline dependencies first."; return 1;
    }
    local flags=(
        -march=rv64imc -mabi=lp64 -mcmodel=medany -mno-relax -mstrict-align
        -std=c++17 -O2 -g -ffunction-sections -fdata-sections
        -fno-rtti -fno-exceptions -fno-threadsafe-statics -fno-use-cxa-atexit
        -fno-unwind-tables -fno-asynchronous-unwind-tables -funsigned-char
        -DTFLM_REMOVE_VIRTUAL_DELETE -DTF_LITE_DISABLE_X86_NEON
        -DFLATBUFFERS_LOCALE_INDEPENDENT=0 -DFLATBUFFERS_PREFER_PRINTF=1
        -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS -DTF_LITE_USE_GLOBAL_MIN
        -DTF_LITE_USE_GLOBAL_MAX -DKERNELS_OPTIMIZED_FOR_SPEED -DVERIFY_XQMAC8_ACCUMULATORS=0 -DENABLE_XQMAC8_COUNTERS=0 -DVERIFY_MLP_STAGES=0 -DCHECK_MLP=0
        -I"$TFLM_DIR"
        -I"$TFLM_DIR/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include"
        -I"$OUT_DIR"
    )
    echo "Compiler: $cxx"
    "$cxx" --version | head -n 1
    local multidir
    multidir="$("$cxx" -march=rv64imc -mabi=lp64 --print-multi-directory)"
    echo "Selected runtime multilib: $multidir"
    case "$multidir" in
        rv64i/lp64|rv64im/lp64|rv64imc/lp64) ;;
        *) echo "Unexpected runtime multilib; stop before linking an incompatible ISA."; return 1 ;;
    esac

    echo "[1/4] Verify original MLP, generate goldens and build verified FlatBuffer"
    "${HOST_CXX:-g++}" -std=c++17 -O2 -I"$TFLM_DIR" -I"$APP_DIR" \
        -I"$TFLM_DIR/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include" \
        "$APP_DIR/host_prepare.cc" -o "$OUT_DIR/host_prepare"
    "$OUT_DIR/host_prepare" "$OUT_DIR"
    python3 "$APP_DIR/embed_model.py" "$OUT_DIR/mlp.tflite" "$OUT_DIR"
    echo "[2/4] Compile startup and HTIF runtime"
    "$cxx" "${flags[@]}" -c "$APP_DIR/start.S" -o "$OUT_DIR/start.o"
    "$cxx" "${flags[@]}" -Wall -Wextra -Wno-unused-parameter -c "$APP_DIR/runtime.cc" -o "$OUT_DIR/runtime.o"
    echo "[3/4] Build reference and prepacked MLP performance executables"
    "$cxx" "${flags[@]}" -c "$OUT_DIR/model_data.cc" -o "$OUT_DIR/model_data.o"
    "$cxx" "${flags[@]}" -Wall -Wextra -Wno-unused-parameter -c "$APP_DIR/fc_xqmac8.cc" -o "$OUT_DIR/fc_xqmac8.o"
    "$cxx" "${flags[@]}" -Wall -Wextra -Wno-unused-parameter -c "$APP_DIR/gelu_lut.cc" -o "$OUT_DIR/gelu.o"
    for variant in reference prepacked; do
        local use_custom=0
        local prepacked=0
        local extra_objects=()
        if [[ "$variant" == prepacked ]]; then
            use_custom=1
            prepacked=1
            extra_objects=("$OUT_DIR/fc_xqmac8.o")
        fi
        "$cxx" "${flags[@]}" -DUSE_XQMAC8="$use_custom" -DUSE_PREPACKED="$prepacked" -Wall -Wextra -Wno-unused-parameter \
            -c "$APP_DIR/main.cc" -o "$OUT_DIR/main_$variant.o"
        "$cxx" "${flags[@]}" -nostdlib -static -T "$APP_DIR/link.ld" \
            -Wl,--gc-sections,--no-relax,-Map,"$OUT_DIR/$variant.map" \
            "$OUT_DIR/start.o" "$OUT_DIR/runtime.o" "$OUT_DIR/main_$variant.o" "$OUT_DIR/model_data.o" "$OUT_DIR/gelu.o" "${extra_objects[@]}" \
            -Wl,--start-group "$archive" -lstdc++ -lm -lc -lgcc -Wl,--end-group \
            -o "$OUT_DIR/$variant.elf"
    done
    echo "[4/4] Build independent correctness executable and inspect performance binaries"
    "$cxx" "${flags[@]}" -UVERIFY_XQMAC8_ACCUMULATORS -DVERIFY_XQMAC8_ACCUMULATORS=1 -UENABLE_XQMAC8_COUNTERS -DENABLE_XQMAC8_COUNTERS=1 \
        -Wall -Wextra -Wno-unused-parameter -c "$APP_DIR/fc_xqmac8.cc" -o "$OUT_DIR/fc_verify.o"
    "$cxx" "${flags[@]}" -UVERIFY_MLP_STAGES -DVERIFY_MLP_STAGES=1 -c "$APP_DIR/gelu_lut.cc" -o "$OUT_DIR/gelu_verify.o"
    "$cxx" "${flags[@]}" -UCHECK_MLP -DCHECK_MLP=1 -c "$APP_DIR/main.cc" -o "$OUT_DIR/main_check.o"
    "$cxx" "${flags[@]}" -nostdlib -static -T "$APP_DIR/link.ld" \
        -Wl,--gc-sections,--no-relax,-Map,"$OUT_DIR/prepacked_check.map" \
        "$OUT_DIR/start.o" "$OUT_DIR/runtime.o" "$OUT_DIR/main_check.o" "$OUT_DIR/model_data.o" "$OUT_DIR/fc_verify.o" "$OUT_DIR/gelu_verify.o" \
        -Wl,--start-group "$archive" -lstdc++ -lm -lc -lgcc -Wl,--end-group \
        -o "$OUT_DIR/prepacked_check.elf"
    "$TC_DIR/riscv-none-elf-size" "$OUT_DIR/reference.elf" "$OUT_DIR/prepacked.elf"
    "$readelf" -A "$OUT_DIR/reference.elf" "$OUT_DIR/prepacked.elf"
    echo "MLP_BUILD_OK"
}

build_all 2>&1 | tee "$OUT_DIR/build.log"
