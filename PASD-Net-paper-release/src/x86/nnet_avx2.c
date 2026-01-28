#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "x86/x86_arch_macros.h"

#ifndef __AVX2__
#error nnet_avx2.c is being compiled without AVX2 enabled
#endif

#define RTCD_ARCH avx2

#include "nnet_arch.h"
