#ifndef COMMON_H
#define COMMON_H

#include "stdlib.h"
#include "string.h"

#define RNN_INLINE inline
#define OPUS_INLINE inline


/** PASDNet wrapper for malloc(). To do your own dynamic allocation, all you need t
o do is replace this function and pasdnet_free */
#ifndef OVERRIDE_PASDNET_ALLOC
static RNN_INLINE void *pasdnet_alloc (size_t size)
{
   return malloc(size);
}
#endif

/** PASDNet wrapper for free(). To do your own dynamic allocation, all you need to do is replace this function and pasdnet_alloc */
#ifndef OVERRIDE_PASDNET_FREE
static RNN_INLINE void pasdnet_free (void *ptr)
{
   free(ptr);
}
#endif

/** Copy n elements from src to dst. The 0* term provides compile-time type checking  */
#ifndef OVERRIDE_RNN_COPY
#define RNN_COPY(dst, src, n) (memcpy((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Copy n elements from src to dst, allowing overlapping regions. The 0* term
    provides compile-time type checking */
#ifndef OVERRIDE_RNN_MOVE
#define RNN_MOVE(dst, src, n) (memmove((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Set n elements of dst to zero */
#ifndef OVERRIDE_RNN_CLEAR
#define RNN_CLEAR(dst, n) (memset((dst), 0, (n)*sizeof(*(dst))))
#endif

# if !defined(OPUS_GNUC_PREREQ)
#  if defined(__GNUC__)&&defined(__GNUC_MINOR__)
#   define OPUS_GNUC_PREREQ(_maj,_min) \
 ((__GNUC__<<16)+__GNUC_MINOR__>=((_maj)<<16)+(_min))
#  else
#   define OPUS_GNUC_PREREQ(_maj,_min) 0
#  endif
# endif


#endif
