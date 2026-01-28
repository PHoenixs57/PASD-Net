#ifndef CPU_SUPPORT_H
#define CPU_SUPPORT_H

#include "opus_types.h"
#include "common.h"

#ifdef RNN_ENABLE_X86_RTCD

#include "x86/x86cpu.h"
/* We currently support 5 x86 variants:
 * arch[0] -> sse2
 * arch[1] -> sse4.1
 * arch[2] -> avx2
 */
#define OPUS_ARCHMASK 3
int rnn_select_arch(void);

#else
#define OPUS_ARCHMASK 0

static OPUS_INLINE int rnn_select_arch(void)
{
  return 0;
}
#endif
#endif
