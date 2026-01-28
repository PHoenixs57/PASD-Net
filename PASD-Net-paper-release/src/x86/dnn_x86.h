#ifndef DNN_X86_H
#define DNN_X86_H

#include "cpu_support.h"
#include "opus_types.h"

void compute_linear_sse4_1(const LinearLayer *linear, float *out, const float *in);
void compute_activation_sse4_1(float *output, const float *input, int N, int activation);
void compute_conv2d_sse4_1(const Conv2dLayer *conv, float *out, float *mem, const float *in, int height, int hstride, int activation);

void compute_linear_avx2(const LinearLayer *linear, float *out, const float *in);
void compute_activation_avx2(float *output, const float *input, int N, int activation);
void compute_conv2d_avx2(const Conv2dLayer *conv, float *out, float *mem, const float *in, int height, int hstride, int activation);



#ifdef RNN_ENABLE_X86_RTCD

extern void (*const RNN_COMPUTE_LINEAR_IMPL[OPUS_ARCHMASK + 1])(
                    const LinearLayer *linear,
                    float *out,
                    const float *in
                    );
#define OVERRIDE_COMPUTE_LINEAR
#define compute_linear(linear, out, in, arch) \
    ((*RNN_COMPUTE_LINEAR_IMPL[(arch) & OPUS_ARCHMASK])(linear, out, in))


extern void (*const RNN_COMPUTE_ACTIVATION_IMPL[OPUS_ARCHMASK + 1])(
                    float *output,
                    const float *input,
                    int N,
                    int activation
                    );
#define OVERRIDE_COMPUTE_ACTIVATION
#define compute_activation(output, input, N, activation, arch) \
    ((*RNN_COMPUTE_ACTIVATION_IMPL[(arch) & OPUS_ARCHMASK])(output, input, N, activation))


extern void (*const RNN_COMPUTE_CONV2D_IMPL[OPUS_ARCHMASK + 1])(
                    const Conv2dLayer *conv,
                    float *out,
                    float *mem,
                    const float *in,
                    int height,
                    int hstride,
                    int activation
                    );
#define OVERRIDE_COMPUTE_CONV2D
#define compute_conv2d(conv, out, mem, in, height, hstride, activation, arch) \
    ((*RNN_COMPUTE_CONV2D_IMPL[(arch) & OPUS_ARCHMASK])(conv, out, mem, in, height, hstride, activation))


#endif



#endif /* DNN_X86_H */
