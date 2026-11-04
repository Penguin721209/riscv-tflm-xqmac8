#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#if VERIFY_MLP_STAGES
#include "golden.h"
extern int g_rounding;
extern unsigned g_stage_mismatches;
#endif

namespace {
TfLiteStatus Prepare(TfLiteContext* c,TfLiteNode* node) {
  using namespace tflite::micro;
  if(node->inputs->size!=2 || node->outputs->size!=1) return kTfLiteError;
  auto* mc=GetMicroContext(c);
  auto* x=mc->AllocateTempInputTensor(node,0);
  auto* lut=mc->AllocateTempInputTensor(node,1);
  auto* y=mc->AllocateTempOutputTensor(node,0);
  bool ok=x&&lut&&y&&x->type==kTfLiteInt8&&lut->type==kTfLiteInt8&&y->type==kTfLiteInt8;
  if(ok) ok=GetTensorShape(x).FlatSize()==151296 && GetTensorShape(y).FlatSize()==151296 &&
    GetTensorShape(lut).FlatSize()==256 && x->params.scale==1.f/16 && y->params.scale==1.f/16 &&
    x->params.zero_point==0 && y->params.zero_point==0;
  if(x) mc->DeallocateTempTfLiteTensor(x);
  if(lut) mc->DeallocateTempTfLiteTensor(lut);
  if(y) mc->DeallocateTempTfLiteTensor(y);
  return ok?kTfLiteOk:kTfLiteError;
}
TfLiteStatus Eval(TfLiteContext* c,TfLiteNode* node) {
  using namespace tflite::micro;
  const auto* x=GetTensorData<int8_t>(GetEvalInput(c,node,0));
  const auto* lut=GetTensorData<int8_t>(GetEvalInput(c,node,1));
  auto* y=GetTensorData<int8_t>(GetEvalOutput(c,node,0));
  if(!x||!lut||!y) return kTfLiteError;
  for(int i=0;i<151296;++i) y[i]=lut[int(x[i])+128];
#if VERIFY_MLP_STAGES
  const auto* q=g_rounding==1?golden_q_double:golden_q_single;
  const auto* g=g_rounding==1?golden_g_double:golden_g_single;
  for(int i=0;i<151296;++i) g_stage_mismatches+=(x[i]!=q[i])+(y[i]!=g[i]);
  if(g_stage_mismatches) return kTfLiteError;
#endif
  return kTfLiteOk;
}
}
TFLMRegistration RegisterMLPGelu() {
  return tflite::micro::RegisterOp(nullptr,Prepare,Eval);
}
