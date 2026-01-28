#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "x86/x86_arch_macros.h"

#ifndef __SSE4_1__
#error nnet_sse4_1.c is being compiled without SSE4.1 enabled
#endif

#define RTCD_ARCH sse4_1

#include "nnet_arch.h"
