#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "data/mlp_data.h"
#include "data/gelu_lut.h"
#include "tensorflow/lite/micro/kernels/internal/fixedpoint.h"
#ifndef GOLDEN_ONLY
#include "tensorflow/lite/micro/schema/schema_generated.h"
#endif

using std::vector;
static int8_t Clamp(int64_t x) { return std::max<int64_t>(-128, std::min<int64_t>(127,x)); }
static int8_t Quant(int32_t x, int mode) {
  if (mode == 0) { // Original assignment: exact /16, ties away from zero.
    return Clamp(x < 0 ? -((-int64_t(x)+8)/16) : (int64_t(x)+8)/16);
  }
  if (mode == 1) { // TFLM default: high multiply, then rounded power-of-two divide.
    using namespace tflite::micro;
    return Clamp(RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, int32_t(1073741824)),3));
  }
  return Clamp((int64_t(x)*1073741824 + (INT64_C(1)<<33)) >> 34);
}
struct Values { vector<int32_t> c1,c2; vector<int8_t> q,g,y; };
static Values Evaluate(int mode) {
  Values v;
  v.c1.resize(MLP_T*MLP_H); v.q.resize(v.c1.size()); v.g.resize(v.c1.size());
  v.c2.resize(MLP_T*MLP_D); v.y.resize(v.c2.size());
  for(int t=0;t<MLP_T;++t) {
    for(int j=0;j<MLP_H;++j) {
      int64_t acc=MLP_B1[j];
      for(int k=0;k<MLP_D;++k) acc+=int32_t(MLP_X[t*MLP_D+k])*MLP_W1[k*MLP_H+j];
      if(acc<INT32_MIN || acc>INT32_MAX) throw std::runtime_error("FC1 overflow");
      int at=t*MLP_H+j; v.c1[at]=acc; v.q[at]=Quant(acc,mode); v.g[at]=GELU_LUT[int(v.q[at])+128];
    }
    for(int j=0;j<MLP_D;++j) {
      int64_t acc=MLP_B2[j];
      for(int k=0;k<MLP_H;++k) acc+=int32_t(v.g[t*MLP_H+k])*MLP_W2[k*MLP_D+j];
      if(acc<INT32_MIN || acc>INT32_MAX) throw std::runtime_error("FC2 overflow");
      int at=t*MLP_D+j; v.c2[at]=acc; v.y[at]=Quant(acc,mode);
    }
  }
  return v;
}
template<class T> static size_t Diff(const vector<T>& a,const T* b) {
  size_t n=0; for(size_t i=0;i<a.size();++i) n+=a[i]!=b[i]; return n;
}
template<class T> static void Array(std::ofstream& f,const char* type,const std::string& name,const vector<T>& a) {
  f<<"static const "<<type<<" "<<name<<"["<<a.size()<<"] = {\n";
  for(size_t i=0;i<a.size();++i) { f<<int64_t(a[i])<<","; if(i%32==31) f<<"\n"; }
  f<<"\n};\n";
}
#ifndef GOLDEN_ONLY
static void Model(const std::string& out) {
  using namespace tflite;
  ModelT m; m.version=3; m.description="Synthetic 197x192x768 MLP; original GELU LUT; TFLM FC quantization";
  auto code=std::make_unique<OperatorCodeT>();
  code->builtin_code=BuiltinOperator_FULLY_CONNECTED; code->deprecated_builtin_code=9; code->version=4;
  m.operator_codes.push_back(std::move(code));
  code=std::make_unique<OperatorCodeT>(); code->builtin_code=BuiltinOperator_CUSTOM;
  code->deprecated_builtin_code=32; code->custom_code="MLP_GELU_LUT";
  m.operator_codes.push_back(std::move(code));
  m.buffers.push_back(std::make_unique<BufferT>());
  auto bytes=[&](const void* p,size_t n) { auto b=std::make_unique<BufferT>();
    const auto* x=static_cast<const uint8_t*>(p); b->data.assign(x,x+n); m.buffers.push_back(std::move(b)); return uint32_t(m.buffers.size()-1); };
  vector<int8_t> w1(MLP_D*MLP_H), w2(MLP_H*MLP_D);
  for(int j=0;j<MLP_H;++j) for(int k=0;k<MLP_D;++k) w1[j*MLP_D+k]=MLP_W1[k*MLP_H+j];
  for(int j=0;j<MLP_D;++j) for(int k=0;k<MLP_H;++k) w2[j*MLP_H+k]=MLP_W2[k*MLP_D+j];
  auto sg=std::make_unique<SubGraphT>(); sg->name="synthetic_mlp";
  auto tensor=[&](const char* name, vector<int32_t> shape, TensorType type,uint32_t buffer,float scale) {
    auto t=std::make_unique<TensorT>(); t->name=name; t->shape=shape; t->has_rank=true;
    t->type=type; t->buffer=buffer; t->quantization=std::make_unique<QuantizationParametersT>();
    t->quantization->scale={scale}; t->quantization->zero_point={0}; sg->tensors.push_back(std::move(t));
  };
  // Serialize INT32 bias in explicit little endian, independent of host byte order.
  auto biasbytes=[](const int32_t* p,int n) { vector<uint8_t> b;
    for(int i=0;i<n;++i) for(int j=0;j<4;++j) b.push_back(uint32_t(p[i])>>(8*j)); return b; };
  auto b1=biasbytes(MLP_B1,MLP_H), b2=biasbytes(MLP_B2,MLP_D);
  tensor("X",{MLP_T,MLP_D},TensorType_INT8,0,1.f/16);
  tensor("W1_transposed",{MLP_H,MLP_D},TensorType_INT8,bytes(w1.data(),w1.size()),1.f/16);
  tensor("B1",{MLP_H},TensorType_INT32,bytes(b1.data(),b1.size()),1.f/256);
  tensor("Q1",{MLP_T,MLP_H},TensorType_INT8,0,1.f/16);
  tensor("GELU_LUT",{256},TensorType_INT8,bytes(GELU_LUT,sizeof GELU_LUT),1.f/16);
  tensor("G1",{MLP_T,MLP_H},TensorType_INT8,0,1.f/16);
  tensor("W2_transposed",{MLP_D,MLP_H},TensorType_INT8,bytes(w2.data(),w2.size()),1.f/16);
  tensor("B2",{MLP_D},TensorType_INT32,bytes(b2.data(),b2.size()),1.f/256);
  tensor("Y",{MLP_T,MLP_D},TensorType_INT8,0,1.f/16);
  auto fc=[&](vector<int32_t> in,int output) { auto op=std::make_unique<OperatorT>();
    op->opcode_index=0; op->inputs=in; op->outputs={output}; FullyConnectedOptionsT options;
    options.quantized_bias_type=TensorType_INT32; op->builtin_options.Set(std::move(options)); sg->operators.push_back(std::move(op)); };
  fc({0,1,2},3);
  auto gelu=std::make_unique<OperatorT>(); gelu->opcode_index=1; gelu->inputs={3,4}; gelu->outputs={5}; sg->operators.push_back(std::move(gelu));
  fc({5,6,7},8); sg->inputs={0}; sg->outputs={8}; m.subgraphs.push_back(std::move(sg));
  flatbuffers::FlatBufferBuilder builder; auto root=CreateModel(builder,&m); FinishModelBuffer(builder,root);
  flatbuffers::Verifier verifier(builder.GetBufferPointer(),builder.GetSize());
  if(!VerifyModelBuffer(verifier)) throw std::runtime_error("FlatBuffer verification failed");
  std::ofstream f(out+"/mlp.tflite",std::ios::binary); f.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),builder.GetSize());
  if(!f) throw std::runtime_error("Cannot save model");
  std::printf("MODEL_VERIFIED bytes=%zu tensors=9 operators=3 MACs=58097664\n",builder.GetSize());
}
#endif
int main(int argc,char** argv) { try {
  if(argc!=2) throw std::runtime_error("usage: host_prepare OUTPUT_DIRECTORY");
  std::string out=argv[1];
  const auto old=Evaluate(0);
  if(Diff(old.c1,REF_C1)||Diff(old.q,REF_Q1)||Diff(old.g,REF_G1)||Diff(old.c2,REF_C2)||Diff(old.y,REF_Y))
    throw std::runtime_error("Original supplied references do not match independent scalar recomputation");
  std::puts("ORIGINAL_MLP_ALL_STAGES_VERIFIED");
  std::ofstream f(out+"/golden.h"), report(out+"/rounding_comparison.csv");
  f<<"#pragma once\n#include <stdint.h>\n";
  Array(f,"int8_t","input_data",vector<int8_t>(MLP_X,MLP_X+MLP_T*MLP_D));
  Array(f,"int8_t","original_y",old.y);
  report<<"rounding,q1_differences,gelu_differences,c2_differences,y_differences\n";
  for(int mode=1;mode<=2;++mode) {
    auto v=Evaluate(mode); const std::string suffix=mode==1?"double":"single";
    Array(f,"int8_t","golden_q_"+suffix,v.q); Array(f,"int8_t","golden_g_"+suffix,v.g); Array(f,"int8_t","golden_y_"+suffix,v.y);
    report<<suffix<<","<<Diff(v.q,REF_Q1)<<","<<Diff(v.g,REF_G1)<<","<<Diff(v.c2,REF_C2)<<","<<Diff(v.y,REF_Y)<<"\n";
    std::printf("rounding=%s old_q1_differences=%zu old_gelu_differences=%zu old_y_differences=%zu\n",suffix.c_str(),Diff(v.q,REF_Q1),Diff(v.g,REF_G1),Diff(v.y,REF_Y));
  }
  if(!f||!report) throw std::runtime_error("Cannot save golden data");
#ifndef GOLDEN_ONLY
  Model(out);
#endif
  return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; } }
