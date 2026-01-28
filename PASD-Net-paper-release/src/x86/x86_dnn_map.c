#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "x86/x86cpu.h"
#include "nnet.h"

#ifdef RNN_ENABLE_X86_RTCD


void (*const RNN_COMPUTE_LINEAR_IMPL[OPUS_ARCHMASK + 1])(
         const LinearLayer *linear,
         float *out,
         const float *in
) = {
  compute_linear_c,                /* non-sse */
  MAY_HAVE_SSE4_1(compute_linear), /* sse4.1  */
  MAY_HAVE_AVX2(compute_linear)  /* avx  */
};

void (*const RNN_COMPUTE_ACTIVATION_IMPL[OPUS_ARCHMASK + 1])(
         float *output,
         const float *input,
         int N,
         int activation
) = {
  compute_activation_c,                /* non-sse */
  MAY_HAVE_SSE4_1(compute_activation), /* sse4.1  */
  MAY_HAVE_AVX2(compute_activation)  /* avx  */
};

void (*const RNN_COMPUTE_CONV2D_IMPL[OPUS_ARCHMASK + 1])(
         const Conv2dLayer *conv,
         float *out,
         float *mem,
         const float *in,
         int height,
         int hstride,
         int activation
) = {
  compute_conv2d_c,                /* non-sse */
  MAY_HAVE_SSE4_1(compute_conv2d), /* sse4.1  */
  MAY_HAVE_AVX2(compute_conv2d)  /* avx  */
};


#endif
