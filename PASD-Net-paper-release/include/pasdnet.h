#ifndef PASDNET_H
#define PASDNET_H 1

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PASDNET_EXPORT
# if defined(WIN32)
#  if defined(PASDNET_BUILD) && defined(DLL_EXPORT)
#   define PASDNET_EXPORT __declspec(dllexport)
#  else
#   define PASDNET_EXPORT
#  endif
# elif defined(__GNUC__) && defined(PASDNET_BUILD)
#  define PASDNET_EXPORT __attribute__ ((visibility ("default")))
# else
#  define PASDNET_EXPORT
# endif
#endif

typedef struct DenoiseState DenoiseState;
typedef struct RNNModel RNNModel;

/**
 * Return the size of DenoiseState
 */
PASDNET_EXPORT int pasdnet_get_size(void);

/**
 * Return the number of samples processed by pasdnet_process_frame at a time
 */
PASDNET_EXPORT int pasdnet_get_frame_size(void);

/**
 * Initializes a pre-allocated DenoiseState
 *
 * If model is NULL the default model is used.
 *
 * See: pasdnet_create() and pasdnet_model_from_file()
 */
PASDNET_EXPORT int pasdnet_init(DenoiseState *st, RNNModel *model);

/**
 * Allocate and initialize a DenoiseState
 *
 * If model is NULL the default model is used.
 *
 * The returned pointer MUST be freed with pasdnet_destroy().
 */
PASDNET_EXPORT DenoiseState *pasdnet_create(RNNModel *model);

/**
 * Free a DenoiseState produced by pasdnet_create.
 *
 * The optional custom model must be freed by pasdnet_model_free() after.
 */
PASDNET_EXPORT void pasdnet_destroy(DenoiseState *st);

/**
 * Denoise a frame of samples
 *
 * in and out must be at least pasdnet_get_frame_size() large.
 */
PASDNET_EXPORT float pasdnet_process_frame(DenoiseState *st, float *out, const float *in);

/**
 * Load a model from a memory buffer
 *
 * It must be deallocated with pasdnet_model_free() and the buffer must remain
 * valid until after the returned object is destroyed.
 */
PASDNET_EXPORT RNNModel *pasdnet_model_from_buffer(const void *ptr, int len);


/**
 * Load a model from a file
 *
 * It must be deallocated with pasdnet_model_free() and the file must not be
 * closed until the returned object is destroyed.
 */
PASDNET_EXPORT RNNModel *pasdnet_model_from_file(FILE *f);

/**
 * Load a model from a file name
 *
 * It must be deallocated with pasdnet_model_free()
 */
PASDNET_EXPORT RNNModel *pasdnet_model_from_filename(const char *filename);

/**
 * Free a custom model
 *
 * It must be called after all the DenoiseStates referring to it are freed.
 */
PASDNET_EXPORT void pasdnet_model_free(RNNModel *model);

#ifdef __cplusplus
}
#endif

#endif
