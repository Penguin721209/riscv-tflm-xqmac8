#pragma once
#include <cstdint>
#include "tensorflow/lite/micro/micro_common.h"

extern volatile uint64_t g_xqmac8_calls;
extern volatile uint64_t g_fc_invocations;
extern volatile uint64_t g_verified_accumulators;
extern volatile uint64_t g_accumulator_mismatches;

TFLMRegistration RegisterFCXqmac8();
