#ifndef NNET_H_
#define NNET_H_

#include <stddef.h>
#include "opus_types.h"

#define ACTIVATION_LINEAR  0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH    2
#define ACTIVATION_RELU    3
#define ACTIVATION_SOFTMAX 4
#define ACTIVATION_SWISH   5

#define WEIGHT_BLOB_VERSION 0
#define WEIGHT_BLOCK_SIZE 64
typedef struct {
  const char *name;
  int type;
  int size;
  const void *data;
} WeightArray;

#define WEIGHT_TYPE_float 0
#define WEIGHT_TYPE_int 1
#define WEIGHT_TYPE_qweight 2
#define WEIGHT_TYPE_int8 3

typedef struct {
  char head[4];
  int version;
  int type;
  int size;
  int block_size;
  char name[44];
} WeightHead;

/* Generic sparse affine transformation. */
typedef struct {
  const float *bias;
  const float *subias;
  const opus_int8 *weights;
  const float *float_weights;
  const int *weights_idx;
  const float *diag;
  const float *scale;
  int nb_inputs;
  int nb_outputs;
} LinearLayer;

/* Generic sparse affine transformation. */
typedef struct {
  const float *bias;
  const float *float_weights;
  int in_channels;
  int out_channels;
  int ktime;
  int kheight;
} Conv2dLayer;


/* Changes some symbol names to add the rnn_ prefix so we don't get conflicts with Opus. */
#define linear_init rnn_linear_init
#define conv2d_init rnn_conv2d_init
#define compute_generic_dense rnn_compute_generic_dense
#define compute_generic_gru rnn_compute_generic_gru
#define compute_generic_conv1d rnn_compute_generic_conv1d
#define compute_glu rnn_compute_glu

#define parse_weights rnn_parse_weights

#define compute_linear_c rnn_compute_linear_c
#define compute_activation_c rnn_compute_activation_c
#define compute_conv2d_c rnn_compute_conv2d_c
#define compute_linear_sse4_1 rnn_compute_linear_sse4_1
#define compute_activation_sse4_1 rnn_compute_activation_sse4_1
#define compute_conv2d_sse4_1 rnn_compute_conv2d_sse4_1
#define compute_linear_avx2 rnn_compute_linear_avx2
#define compute_activation_avx2 rnn_compute_activation_avx2
#define compute_conv2d_avx2 rnn_compute_conv2d_avx2


void compute_generic_dense(const LinearLayer *layer, float *output, const float *input, int activation, int arch);
void compute_generic_gru(const LinearLayer *input_weights, const LinearLayer *recurrent_weights, float *state, const float *in, int arch);
void compute_generic_conv1d(const LinearLayer *layer, float *output, float *mem, const float *input, int input_size, int activation, int arch);
void compute_glu(const LinearLayer *layer, float *output, const float *input, int arch);


int parse_weights(WeightArray **list, const void *data, int len);



int linear_init(LinearLayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *subias,
  const char *weights,
  const char *float_weights,
  const char *weights_idx,
  const char *diag,
  const char *scale,
  int nb_inputs,
  int nb_outputs);

int conv2d_init(Conv2dLayer *layer, const WeightArray *arrays,
  const char *bias,
  const char *float_weights,
  int in_channels,
  int out_channels,
  int ktime,
  int kheight);


void compute_linear_c(const LinearLayer *linear, float *out, const float *in);
void compute_activation_c(float *output, const float *input, int N, int activation);
void compute_conv2d_c(const Conv2dLayer *conv, float *out, float *mem, const float *in, int height, int hstride, int activation);

void compute_attention_gate(const LinearLayer *attn_layer,
                            float *output,
                            const float *input,
                            int N,
                            int arch);

#ifdef RNN_ENABLE_X86_RTCD
#include "x86/dnn_x86.h"
#endif

#ifndef OVERRIDE_COMPUTE_LINEAR
#define compute_linear(linear, out, in, arch) ((void)(arch),compute_linear_c(linear, out, in))
#endif

#ifndef OVERRIDE_COMPUTE_ACTIVATION
#define compute_activation(output, input, N, activation, arch) ((void)(arch),compute_activation_c(output, input, N, activation))
#endif

#ifndef OVERRIDE_COMPUTE_CONV2D
#define compute_conv2d(conv, out, mem, in, height, hstride, activation, arch) ((void)(arch),compute_conv2d_c(conv, out, mem, in, height, hstride, activation))
#endif

/* Optional performance hint. Commented out to avoid compiler warnings.
 * If you care about maximum x86 performance, you can re-enable this
 * and pass an appropriate -march= flag when compiling.
 */
/*
#if defined(__x86_64__) && !defined(RNN_ENABLE_X86_RTCD) && !defined(__AVX2__)
#if defined(_MSC_VER)
#pragma message ("Only SSE and SSE2 are available. On newer machines, enable SSSE3/AVX/AVX2 to get better performance")
#else
#warning "Only SSE and SSE2 are available. On newer machines, enable SSSE3/AVX/AVX2 using -march= to get better performance"
#endif
#endif
*/



#endif /* NNET_H_ */
