#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include "nnet.h"
#include "arch.h"
#include "nnet.h"

/* This is a bit of a hack because we need to build nnet_data.c and plc_data.c without USE_WEIGHTS_FILE,
   but USE_WEIGHTS_FILE is defined in config.h. */
#undef HAVE_CONFIG_H
#ifdef USE_WEIGHTS_FILE
#undef USE_WEIGHTS_FILE
#endif
#include "pasdnet_data.c"

void write_weights(const WeightArray *list, FILE *fout)
{
  int i=0;
  unsigned char zeros[WEIGHT_BLOCK_SIZE] = {0};
  while (list[i].name != NULL) {
    WeightHead h;
    if (strlen(list[i].name) >= sizeof(h.name) - 1) {
      printf("[write_weights] warning: name %s too long\n", list[i].name);
    }
    memcpy(h.head, "DNNw", 4);
    h.version = WEIGHT_BLOB_VERSION;
    h.type = list[i].type;
    h.size = list[i].size;
    h.block_size = (h.size+WEIGHT_BLOCK_SIZE-1)/WEIGHT_BLOCK_SIZE*WEIGHT_BLOCK_SIZE;
    RNN_CLEAR(h.name, sizeof(h.name));
    strncpy(h.name, list[i].name, sizeof(h.name));
    h.name[sizeof(h.name)-1] = 0;
    celt_assert(sizeof(h) == WEIGHT_BLOCK_SIZE);
    fwrite(&h, 1, WEIGHT_BLOCK_SIZE, fout);
    fwrite(list[i].data, 1, h.size, fout);
    fwrite(zeros, 1, h.block_size-h.size, fout);
    i++;
  }
}

int main(void)
{
  FILE *fout = fopen("weights_blob.bin", "w");
  write_weights(pasdnet_arrays, fout);
  fclose(fout);
  return 0;
}
