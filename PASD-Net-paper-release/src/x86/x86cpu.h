#if !defined(X86CPU_H)
# define X86CPU_H

#  define MAY_HAVE_SSE4_1(name) name ## _sse4_1

#  define MAY_HAVE_AVX2(name) name ## _avx2

# ifdef RNN_ENABLE_X86_RTCD
int opus_select_arch(void);
# endif

# if defined(__SSE2__)
#  include "common.h"

/*MOVD should not impose any alignment restrictions, but the C standard does,
   and UBSan will report errors if we actually make unaligned accesses.
  Use this to work around those restrictions (which should hopefully all get
   optimized to a single MOVD instruction).
  GCC implemented _mm_loadu_si32() since GCC 11; HOWEVER, there is a bug!
  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=99754
  LLVM implemented _mm_loadu_si32() since Clang 8.0, however the
   __clang_major__ version number macro is unreliable, as vendors
   (specifically, Apple) will use different numbering schemes than upstream.
  Clang's advice is "use feature detection", but they do not provide feature
   detection support for specific SIMD functions.
  We follow the approach from the SIMDe project and instead detect unrelated
   features that should be available in the version we want (see
   <https://github.com/simd-everywhere/simde/blob/master/simde/simde-detect-clang.h>).*/
#  if defined(__clang__)
#   if __has_warning("-Wextra-semi-stmt") || \
 __has_builtin(__builtin_rotateleft32)
#    define OPUS_CLANG_8 (1)
#   endif
#  endif
#  if !defined(_MSC_VER) && !OPUS_GNUC_PREREQ(11,3) && !defined(OPUS_CLANG_8)
#   include <string.h>
#   include <emmintrin.h>

#   ifdef _mm_loadu_si32
#    undef _mm_loadu_si32
#   endif
#   define _mm_loadu_si32 WORKAROUND_mm_loadu_si32
static inline __m128i WORKAROUND_mm_loadu_si32(void const* mem_addr) {
  int val;
  memcpy(&val, mem_addr, sizeof(val));
  return _mm_cvtsi32_si128(val);
}
#  elif defined(_MSC_VER)
    /* MSVC needs this for _mm_loadu_si32 */
#   include <immintrin.h>
#  endif

#  define OP_CVTEPI8_EPI32_M32(x) \
 (_mm_cvtepi8_epi32(_mm_loadu_si32(x)))

#  define OP_CVTEPI16_EPI32_M64(x) \
 (_mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i *)(void*)(x))))

# endif

#endif
