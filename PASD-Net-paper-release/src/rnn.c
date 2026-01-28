#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include "opus_types.h"
#include "common.h"
#include "arch.h"
#include "rnn.h"
#include "pasdnet_data.h"
#include <stdio.h>


#define INPUT_SIZE 42


void compute_rnn(const PASDNet *model, RNNState *rnn, float *gains, float *vad, const float *input, int arch) {
  float cat[CONV2_OUT_SIZE + GRU1_OUT_SIZE + GRU2_OUT_SIZE + GRU3_OUT_SIZE];
  float tmp1[CONV1_OUT_SIZE];
  float tmp_conv[CONV2_OUT_SIZE];
  float attn_out[CONV2_OUT_SIZE];
  /*for (int i=0;i<INPUT_SIZE;i++) printf("%f ", input[i]);printf("\n");*/
  compute_generic_conv1d(&model->conv1, tmp1, rnn->conv1_state, input, CONV1_IN_SIZE, ACTIVATION_TANH, arch);
  compute_generic_conv1d(&model->conv2, tmp_conv, rnn->conv2_state, tmp1, CONV2_IN_SIZE, ACTIVATION_TANH, arch);
  /* 当前权重文件中还没有 attention 层参数，
     这里暂时直接跳过 attention，等价于 attn_out = tmp_conv。*/
  RNN_COPY(attn_out, tmp_conv, CONV2_OUT_SIZE);

  /* GRU1 输入 attn_out（此处等同于 tmp_conv） */
  compute_generic_gru(&model->gru1_input, &model->gru1_recurrent, rnn->gru1_state, attn_out, arch);
  compute_generic_gru(&model->gru2_input, &model->gru2_recurrent, rnn->gru2_state, rnn->gru1_state, arch);
  compute_generic_gru(&model->gru3_input, &model->gru3_recurrent, rnn->gru3_state, rnn->gru2_state, arch);
  /* 与 PyTorch PASDNet.forward 保持一致：cat 的前一段使用 conv2 的输出 tmp，
     后三段依次是三层 GRU 的状态。 */
  RNN_COPY(cat, tmp_conv, CONV2_OUT_SIZE);
  RNN_COPY(&cat[CONV2_OUT_SIZE], rnn->gru1_state, GRU1_OUT_SIZE);
  RNN_COPY(&cat[CONV2_OUT_SIZE+GRU1_OUT_SIZE], rnn->gru2_state, GRU2_OUT_SIZE);
  RNN_COPY(&cat[CONV2_OUT_SIZE+GRU1_OUT_SIZE+GRU2_OUT_SIZE], rnn->gru3_state, GRU3_OUT_SIZE);
  compute_generic_dense(&model->dense_out, gains, cat, ACTIVATION_SIGMOID, arch);
  compute_generic_dense(&model->vad_dense, vad, cat, ACTIVATION_SIGMOID, arch);
  /*for (int i=0;i<22;i++) printf("%f ", gains[i]);printf("\n");*/
  /*printf("%f\n", *vad);*/
}
