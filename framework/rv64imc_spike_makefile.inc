# RV64IMC/LP64 static-library target for the inspected TFLM source snapshot.
# Install in tensorflow/lite/micro/tools/make/targets/.
# This file builds the library only. Spike startup, logging and ELF linking
# are supplied separately when building the inference executable.

TARGET_ARCH := rv64imc
TARGET_TOOLCHAIN_PREFIX := riscv-none-elf-

ifeq ($(strip $(TARGET_TOOLCHAIN_ROOT)),)
  $(error Pass TARGET_TOOLCHAIN_ROOT=/absolute/path/to/xpack/bin/)
endif

PLATFORM_FLAGS := \
  -march=rv64imc \
  -mabi=lp64 \
  -mcmodel=medany \
  -mstrict-align \
  -funsigned-char \
  -fno-delete-null-pointer-checks \
  -fomit-frame-pointer \
  -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS

CXXFLAGS += $(PLATFORM_FLAGS) \
  -fno-use-cxa-atexit \
  -DTF_LITE_USE_GLOBAL_MIN \
  -DTF_LITE_USE_GLOBAL_MAX \
  -Wno-error

CCFLAGS += $(PLATFORM_FLAGS) -Wno-error

# Keep upstream reference kernels and default logging/time implementations.
# No toolchain download, QEMU invocation, custom op, or runtime ISA extension.
