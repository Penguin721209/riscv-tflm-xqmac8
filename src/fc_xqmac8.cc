#include <cstddef>
#include <cstdint>
#include "fc_xqmac8.h"
#include "prepacked_dot.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#ifdef USE_TFLM_COMPRESSION
#error "Prepacked adapter requires uncompressed constant weights."
#endif

volatile uint64_t g_xqmac8_calls = 0;
volatile uint64_t g_fc_invocations = 0;
volatile uint64_t g_verified_accumulators = 0;
volatile uint64_t g_accumulator_mismatches = 0;

namespace {
struct State {
    void* original_data;
    uint64_t* packed_weights;
    int64_t* correction_bias;
    const int8_t* raw_weights;
    const int32_t* raw_bias;
    int k, n, batches, blocks, scratch_index;
    bool prepared;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    auto* state = static_cast<State*>(context->AllocatePersistentBuffer(context, sizeof(State)));
    if (!state) return nullptr;
    *state = {};
    const auto original = tflite::micro::Register_FULLY_CONNECTED();
    state->original_data = original.init(context, buffer, length);
    return state->original_data ? state : nullptr;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    using namespace tflite::micro;
    auto* state = static_cast<State*>(node->user_data);
    if (!state || !state->original_data) return kTfLiteError;
    state->prepared = false;
    // Preserve the original framework's complete quantization preparation.
    node->user_data = state->original_data;
    const auto status = Register_FULLY_CONNECTED().prepare(context, node);
    node->user_data = state;
    if (status != kTfLiteOk) return status;
    auto* mc = GetMicroContext(context);
    auto* input = mc->AllocateTempInputTensor(node, kFullyConnectedInputTensor);
    auto* filter = mc->AllocateTempInputTensor(node, kFullyConnectedWeightsTensor);
    auto* bias = mc->AllocateTempInputTensor(node, kFullyConnectedBiasTensor);
    auto* output = mc->AllocateTempOutputTensor(node, kFullyConnectedOutputTensor);
    const auto cleanup = [&]() {
        if (input) mc->DeallocateTempTfLiteTensor(input);
        if (filter) mc->DeallocateTempTfLiteTensor(filter);
        if (bias) mc->DeallocateTempTfLiteTensor(bias);
        if (output) mc->DeallocateTempTfLiteTensor(output);
    };
    if (!input || !filter || !output || input->type != kTfLiteInt8 ||
        filter->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
        (bias && bias->type != kTfLiteInt32)) {
        MicroPrintf("Prepacked FC requires INT8 input/weights/output and INT32 bias");
        cleanup(); return kTfLiteError;
    }
    const auto readonly = [](const TfLiteTensor* tensor) {
        return tensor && !tensor->is_variable &&
            (tensor->allocation_type == kTfLiteMmapRo || tensor->allocation_type == kTfLitePersistentRo);
    };
    if (!readonly(filter) || (bias && !readonly(bias))) {
        MicroPrintf("Prepacked FC requires constant read-only weights and bias");
        cleanup(); return kTfLiteError;
    }
    const auto& data = *static_cast<const OpDataFullyConnected*>(state->original_data);
    const auto in_shape = GetTensorShape(input);
    const auto w_shape = GetTensorShape(filter);
    const auto out_shape = GetTensorShape(output);
    if (data.filter_zero_point != 0 || w_shape.DimensionsCount() != 2 ||
        out_shape.DimensionsCount() < 1) { cleanup(); return kTfLiteError; }
    state->k = w_shape.Dims(1);
    state->n = out_shape.Dims(out_shape.DimensionsCount() - 1);
    state->batches = FlatSizeSkipDim(out_shape, out_shape.DimensionsCount() - 1);
    if (state->k <= 0 || state->k > 131071 || state->n <= 0 || state->batches <= 0 ||
        w_shape.Dims(0) != state->n || static_cast<int64_t>(state->batches) * state->k != in_shape.FlatSize() ||
        (bias && GetTensorShape(bias).FlatSize() != state->n) ||
        (data.is_per_channel && (!data.per_channel_output_multiplier || !data.per_channel_output_shift))) {
        cleanup(); return kTfLiteError;
    }
    state->blocks = (state->k + 7) / 8;
    const uint64_t weight_bytes = static_cast<uint64_t>(state->n) * state->blocks * sizeof(uint64_t);
    if (weight_bytes > SIZE_MAX) { cleanup(); return kTfLiteError; }
    state->packed_weights = static_cast<uint64_t*>(context->AllocatePersistentBuffer(context, static_cast<size_t>(weight_bytes)));
    state->correction_bias = static_cast<int64_t*>(context->AllocatePersistentBuffer(context, static_cast<size_t>(state->n) * sizeof(int64_t)));
    if (!state->packed_weights || !state->correction_bias) { cleanup(); return kTfLiteError; }
    state->raw_weights = GetTensorData<int8_t>(filter);
    state->raw_bias = bias ? GetTensorData<int32_t>(bias) : nullptr;
    if (!state->raw_weights) { cleanup(); return kTfLiteError; }
    const int32_t input_offset = -data.input_zero_point;
    for (int channel = 0; channel < state->n; ++channel) {
        const int8_t* row = state->raw_weights + channel * state->k;
        xqmac8_prepacked::PackRow(row, state->k, state->packed_weights + channel * state->blocks);
        int64_t weight_sum = 0;
        for (int k = 0; k < state->k; ++k) weight_sum += row[k];
        state->correction_bias[channel] = static_cast<int64_t>(input_offset) * weight_sum +
                                         (state->raw_bias ? state->raw_bias[channel] : 0);
    }
    const int scratch_bytes = state->blocks * static_cast<int>(sizeof(uint64_t));
    const auto scratch_status = context->RequestScratchBufferInArena(context, scratch_bytes, &state->scratch_index);
    cleanup();
    if (scratch_status != kTfLiteOk) return scratch_status;
    state->prepared = true;
    return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
    using namespace tflite::micro;
    const auto* state = static_cast<const State*>(node->user_data);
    if (!state || !state->prepared) return kTfLiteError;
    const auto& data = *static_cast<const OpDataFullyConnected*>(state->original_data);
    const auto params = FullyConnectedParamsQuantized(data);
    const auto* input = GetEvalInput(context, node, kFullyConnectedInputTensor);
    auto* output = GetEvalOutput(context, node, kFullyConnectedOutputTensor);
    if (!input || !output) return kTfLiteError;
    const auto* x = GetTensorData<int8_t>(input);
    auto* y = GetTensorData<int8_t>(output);
    auto* packed_input = static_cast<uint64_t*>(context->GetScratchBuffer(context, state->scratch_index));
    if (!packed_input || !x || !y) return kTfLiteError;
#if ENABLE_XQMAC8_COUNTERS
    g_fc_invocations = g_fc_invocations + 1;
#endif
    for (int batch = 0; batch < state->batches; ++batch) {
        const int8_t* row = x + batch * state->k;
        xqmac8_prepacked::PackRow(row, state->k, packed_input);
        for (int channel = 0; channel < state->n; ++channel) {
            int64_t acc = xqmac8_prepacked::Dot(packed_input, state->packed_weights + channel * state->blocks, state->blocks);
            acc += state->correction_bias[channel];
#if VERIFY_XQMAC8_ACCUMULATORS
            int64_t reference = state->raw_bias ? state->raw_bias[channel] : 0;
            for (int k = 0; k < state->k; ++k)
                reference += (static_cast<int32_t>(row[k]) + params.input_offset) *
                             static_cast<int32_t>(state->raw_weights[channel * state->k + k]);
            g_verified_accumulators = g_verified_accumulators + 1;
            if (acc != reference) {
                g_accumulator_mismatches = g_accumulator_mismatches + 1;
                MicroPrintf("Prepacked accumulator mismatch batch=%d channel=%d", batch, channel);
                return kTfLiteError;
            }
#endif
            if (acc < INT32_MIN || acc > INT32_MAX) return kTfLiteError;
            const int32_t multiplier = data.is_per_channel ? data.per_channel_output_multiplier[channel] : params.output_multiplier;
            const int shift = data.is_per_channel ? data.per_channel_output_shift[channel] : params.output_shift;
            int64_t scaled = MultiplyByQuantizedMultiplier(static_cast<int32_t>(acc), multiplier, shift);
            scaled += params.output_offset;
            if (scaled < params.quantized_activation_min) scaled = params.quantized_activation_min;
            if (scaled > params.quantized_activation_max) scaled = params.quantized_activation_max;
            y[batch * state->n + channel] = static_cast<int8_t>(scaled);
        }
    }
    return kTfLiteOk;
}
}

TFLMRegistration RegisterFCXqmac8() {
    auto registration = tflite::micro::Register_FULLY_CONNECTED();
    registration.init = Init;
    registration.prepare = Prepare;
    registration.invoke = Eval;
    return registration;
}
