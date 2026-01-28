#include <stdio.h>
#include <string.h>
#include "pasdnet.h"

#define FRAME_SIZE 480
#define DEBUG_PASDNET 1
/* 丢弃前 OFFSET_FRAMES 帧的降噪结果，并在结尾补零以保持总时长 */
#define OFFSET_FRAMES 1

int main(int argc, char **argv) {
  int i;
  float x[FRAME_SIZE];
  float xin[FRAME_SIZE]; /* 用于调试输入/输出对比 */
  FILE *f1, *fout;
  DenoiseState *st;
  long frame_idx = 0;           /* 处理的总帧数 */
  long total_in_samp = 0;       /* 输入总采样数 */
  long total_out_samp = 0;      /* 已写出的采样数 */
#ifdef DEBUG_PASDNET
#endif
#ifdef USE_WEIGHTS_FILE
  RNNModel *model = pasdnet_model_from_filename("weights_blob.bin");
  st = pasdnet_create(model);
#else
  st = pasdnet_create(NULL);
#endif

  if (argc!=3) {
    fprintf(stderr, "usage: %s <noisy speech> <output denoised>\n", argv[0]);
    return 1;
  }
  if (!(f1 = fopen(argv[1],"rb")))
  {
    perror("Failed to open input file");
    return -1;
  }
  if (!(fout = fopen(argv[2],"wb")))
  {
    perror("Failed to open output file");
    return -1;
  }
  short tmp[FRAME_SIZE];
  size_t nread;
  while ((nread = fread(tmp, sizeof(short), FRAME_SIZE, f1)) > 0) {
    total_in_samp += (long)nread;
    // 把读到的样本拷到 x，剩下的填 0
    for (i = 0; i < FRAME_SIZE; i++) {
        if (i < nread) x[i] = tmp[i];
        else x[i] = 0;
        xin[i] = x[i];
    }
    pasdnet_process_frame(st, x, x);
#ifdef DEBUG_PASDNET
    /* 打印前几帧的一部分输入/输出样本，便于对比/调试 */
    frame_idx++;
    if (frame_idx <= 5) {
      int print_n = nread < 10 ? (int)nread : 10;
      fprintf(stderr, "[demo] frame %ld, first %d input samples:", frame_idx, print_n);
      for (i = 0; i < print_n; i++) {
        fprintf(stderr, " %.1f", xin[i]);
      }
      fprintf(stderr, "\n");

      fprintf(stderr, "[demo] frame %ld, first %d output samples:", frame_idx, print_n);
      for (i = 0; i < print_n; i++) {
        fprintf(stderr, " %.1f", x[i]);
      }
      fprintf(stderr, "\n");
    }
#endif
    for (i = 0; i < FRAME_SIZE; i++) {
        tmp[i] = (short)x[i];
    }
    /* 丢弃前 OFFSET_FRAMES 帧的降噪结果，但统计输入长度 */
    if (frame_idx > OFFSET_FRAMES) {
      // 只写回原来读取的 nread 个样本长度
      fwrite(tmp, sizeof(short), nread, fout);
      total_out_samp += (long)nread;
    }
  }

  /* 为了保证输出总时长与输入一致，如果丢弃了前几帧，
     在结尾补足相同数量的采样（这里补 0）。*/
  if (total_out_samp < total_in_samp) {
    long remaining = total_in_samp - total_out_samp;
    short pad[FRAME_SIZE];
    memset(pad, 0, sizeof(pad));
    while (remaining > 0) {
      size_t to_write = remaining > FRAME_SIZE ? FRAME_SIZE : (size_t)remaining;
      fwrite(pad, sizeof(short), to_write, fout);
      remaining -= (long)to_write;
    }
  }
  pasdnet_destroy(st);
  fclose(f1);
  fclose(fout);
#ifdef USE_WEIGHTS_FILE
  pasdnet_model_free(model);
#endif
  return 0;
}