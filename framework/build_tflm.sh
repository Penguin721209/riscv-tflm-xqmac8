#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TFLM="$REPO/deps/tflite-micro-main"
TC="${TFLM_TOOLCHAIN_BIN:-$HOME/toolchains/xpack-15.2.0/bin}"

cd "$REPO"
sha256sum -c vendor/SHA256SUMS

mkdir -p deps
tar --no-same-owner -xzf vendor/tflite-micro-main.tar.gz -C deps

DOWNLOADS="$TFLM/tensorflow/lite/micro/tools/make/downloads"
mkdir -p "$DOWNLOADS"
tar --no-same-owner -xzf vendor/tflm_dependencies.tar.gz -C "$DOWNLOADS"

cp framework/rv64imc_spike_makefile.inc \
  "$TFLM/tensorflow/lite/micro/tools/make/targets/"

make -C "$TFLM" \
  -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=rv64imc_spike \
  TARGET_ARCH=rv64imc \
  TARGET_TOOLCHAIN_ROOT="$TC/" \
  SKIP_DOWNLOADS=1 \
  -j2 microlite

echo "TFLM_LIBRARY_BUILD_OK"
