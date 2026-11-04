#include <cstdint>
#include <cstdio>
#include <cstring>
#include "model_data.h"
#include "golden.h"
#include "fc_xqmac8.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

extern "C" int _write(int,const void*,size_t);
TFLMRegistration RegisterMLPGelu();
int g_rounding=0;
unsigned g_stage_mismatches=0;
extern "C" __attribute__((noinline,noclone,used)) void perf_begin() { __asm__ volatile("nop" ::: "memory"); }
extern "C" __attribute__((noinline,noclone,used)) void perf_end() { __asm__ volatile("nop\nnop" ::: "memory"); }
extern "C" __attribute__((noinline,noclone,used)) void prepare_begin() { __asm__ volatile("nop\nnop\nnop" ::: "memory"); }
extern "C" __attribute__((noinline,noclone,used)) void prepare_end() { __asm__ volatile("nop\nnop\nnop\nnop" ::: "memory"); }
namespace {
alignas(16) uint8_t arena[2*1024*1024];
int8_t saved_reference[37824];
char hex_outputs[2*37824+32];
bool Emit(const char* p,size_t n) { while(n) { int k=_write(1,p,n); if(k<=0) return false; p+=k; n-=k; } return true; }
int Run(bool custom,bool measured) {
  std::memset(arena,0,sizeof arena);
  const auto* model=tflite::GetModel(g_model);
  if(model->version()!=TFLITE_SCHEMA_VERSION) return 1;
  tflite::MicroMutableOpResolver<2> resolver;
#if CHECK_MLP || USE_XQMAC8
  if(custom) {
    if(resolver.AddFullyConnected(RegisterFCXqmac8())!=kTfLiteOk) return 2;
  } else
#endif
  if(resolver.AddFullyConnected()!=kTfLiteOk) return 2;
  const auto gelu=RegisterMLPGelu();
  if(resolver.AddCustom("MLP_GELU_LUT",&gelu)!=kTfLiteOk) return 3;
  tflite::MicroInterpreter interpreter(model,resolver,arena,sizeof arena);
  if(measured) prepare_begin();
  auto status=interpreter.AllocateTensors();
  if(measured) prepare_end();
  if(status!=kTfLiteOk) return 4;
  auto* input=interpreter.input(0); auto* output=interpreter.output(0);
  if(!input||!output||input->type!=kTfLiteInt8||output->type!=kTfLiteInt8||input->bytes!=37824||output->bytes!=37824) return 5;
  std::memcpy(input->data.int8,input_data,sizeof input_data);
  if(measured) perf_begin();
  status=interpreter.Invoke();
  if(measured) perf_end();
  if(status!=kTfLiteOk) return 6;
  const auto* golden=g_rounding==1?golden_y_double:golden_y_single;
  unsigned errors=0,old_errors=0,pair_errors=0;
  for(int i=0;i<37824;++i) {
    int8_t v=output->data.int8[i]; errors+=v!=golden[i]; old_errors+=v!=original_y[i];
#if CHECK_MLP
    if(custom) pair_errors+=v!=saved_reference[i]; else saved_reference[i]=v;
#endif
  }
  char line[300];
  int n=std::snprintf(line,sizeof line,"MLP_RESULT variant=%s tokens=197 outputs=37824 golden_mismatches=%u original_mismatches=%u pair_mismatches=%u stage_mismatches=%u arena_used=%lu\n",custom?"prepacked":"reference",errors,old_errors,pair_errors,g_stage_mismatches,(unsigned long)interpreter.arena_used_bytes());
  if(n<=0||size_t(n)>=sizeof line||!Emit(line,n)) return 7;
  if(errors||pair_errors||g_stage_mismatches) return 8;
  if(measured) {
    n=std::snprintf(line,sizeof line,"PERF_RESULT variant=%s inputs=197 arena_reserved=2097152 arena_used=%lu\n",custom?"prepacked":"reference",(unsigned long)interpreter.arena_used_bytes());
    if(n<=0||size_t(n)>=sizeof line||!Emit(line,n)) return 9;
    const char* digits="0123456789abcdef";
    constexpr char prefix[]="PERF_OUTPUTS "; std::memcpy(hex_outputs,prefix,sizeof(prefix)-1);
    unsigned at=sizeof(prefix)-1;
    for(int i=0;i<37824;++i) { uint8_t b=output->data.int8[i]; hex_outputs[at++]=digits[b>>4]; hex_outputs[at++]=digits[b&15]; }
    hex_outputs[at++]='\n'; if(!Emit(hex_outputs,at)) return 10;
    constexpr char pass[]="PERF_CHECK_PASSED\n"; if(!Emit(pass,sizeof(pass)-1)) return 11;
  }
  return 0;
}
}
int main() {
  // Probe the linked library rather than assume its compile-time rounding setting.
  const int probe=tflite::micro::MultiplyByQuantizedMultiplier(7,1073741824,-3);
  g_rounding=probe==1?1:(probe==0?2:0);
  if(!g_rounding) return 30;
  char text[120]; int n=std::snprintf(text,sizeof text,"MLP_START tokens=197 dim=192 hidden=768 rounding=%s MACs=58097664\n",g_rounding==1?"double":"single");
  if(n<=0||size_t(n)>=sizeof text||!Emit(text,n)) return 31;
#if CHECK_MLP
  int status=Run(false,false); if(status) return status;
  g_xqmac8_calls=0; g_fc_invocations=0; g_verified_accumulators=0; g_accumulator_mismatches=0;
  status=Run(true,false); if(status) return 40+status;
  char counters[240]; n=std::snprintf(counters,sizeof counters,"MLP_COUNTERS fc_invocations=%lu xqmac8_calls=%lu verified_accumulators=%lu accumulator_mismatches=%lu\n",(unsigned long)g_fc_invocations,(unsigned long)g_xqmac8_calls,(unsigned long)g_verified_accumulators,(unsigned long)g_accumulator_mismatches);
  if(n<=0||size_t(n)>=sizeof counters||!Emit(counters,n)) return 60;
  if(g_fc_invocations!=2||g_xqmac8_calls!=7262208||g_verified_accumulators!=189120||g_accumulator_mismatches) return 61;
  constexpr char pass[]="TFLM_MLP_XQMAC8_EXACT_MATCH_37824\n";
  return Emit(pass,sizeof(pass)-1)?0:62;
#else
  return Run(USE_XQMAC8,true);
#endif
}
