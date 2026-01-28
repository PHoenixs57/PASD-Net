/*
   dump_single_features: dump per-frame PASDNet features for a single audio stream.

   Input:
     - WAV (RIFF/WAVE) PCM 16-bit mono (recommended 48 kHz)
     - or raw PCM 16-bit little-endian mono (48 kHz)

   Output (CSV):
     frame,pitch_index,silence,features[0..NB_FEATURES-1],Ex[0..NB_BANDS-1],Exp[0..NB_BANDS-1]

   Notes:
     - pitch_index is reconstructed from features[2*NB_BANDS] = 0.01*(pitch_index-300)
     - feature extraction assumes 48 kHz / FRAME_SIZE=480 (10 ms)
*/

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pasdnet.h"
#include "denoise.h"
#include "kiss_fft.h"

/*
  When compiled with -DTRAINING, denoise.c expects these globals.
  We set them to keep the full band (no lowpass masking).
*/
#if TRAINING
int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;
#endif

static uint16_t read_le16(const uint8_t *p) {
  return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t read_le32(const uint8_t *p) {
  return (uint32_t)(p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
                    ((uint32_t)p[3] << 24));
}

typedef struct {
  FILE *f;
  int is_wav;
  uint32_t sample_rate;
  uint16_t channels;
  uint16_t bits_per_sample;
  uint32_t data_remaining_bytes;
} AudioReader;

static void die(const char *msg) {
  fprintf(stderr, "%s\n", msg);
  exit(1);
}

static void die_errno(const char *msg) {
  fprintf(stderr, "%s: %s\n", msg, strerror(errno));
  exit(1);
}

static int starts_with(const char *s, const char *prefix) {
  return strncmp(s, prefix, strlen(prefix)) == 0;
}

static void audio_reader_open_pcm(AudioReader *r, const char *path) {
  memset(r, 0, sizeof(*r));
  r->f = fopen(path, "rb");
  if (!r->f) die_errno("cannot open input");
  r->is_wav = 0;
  r->sample_rate = 48000;
  r->channels = 1;
  r->bits_per_sample = 16;
  r->data_remaining_bytes = 0xFFFFFFFFu; /* unknown */
}

static void audio_reader_open_wav(AudioReader *r, const char *path) {
  uint8_t hdr[12];
  uint8_t chunk_hdr[8];
  uint32_t fmt_sample_rate = 0;
  uint16_t fmt_audio_format = 0;
  uint16_t fmt_channels = 0;
  uint16_t fmt_bits = 0;
  int have_fmt = 0;
  int have_data = 0;

  memset(r, 0, sizeof(*r));
  r->f = fopen(path, "rb");
  if (!r->f) die_errno("cannot open input");

  if (fread(hdr, 1, sizeof(hdr), r->f) != sizeof(hdr)) die("invalid WAV: short header");
  if (memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
    die("invalid WAV: missing RIFF/WAVE");
  }

  while (!have_data) {
    uint32_t chunk_size;
    if (fread(chunk_hdr, 1, sizeof(chunk_hdr), r->f) != sizeof(chunk_hdr)) {
      die("invalid WAV: missing data chunk");
    }
    chunk_size = read_le32(chunk_hdr + 4);

    if (memcmp(chunk_hdr, "fmt ", 4) == 0) {
      uint8_t *fmt = (uint8_t *)malloc(chunk_size);
      if (!fmt) die("oom");
      if (fread(fmt, 1, chunk_size, r->f) != chunk_size) die("invalid WAV: short fmt");
      if (chunk_size < 16) die("invalid WAV: fmt too small");
      fmt_audio_format = read_le16(fmt + 0);
      fmt_channels = read_le16(fmt + 2);
      fmt_sample_rate = read_le32(fmt + 4);
      fmt_bits = read_le16(fmt + 14);
      free(fmt);
      have_fmt = 1;
    } else if (memcmp(chunk_hdr, "data", 4) == 0) {
      if (!have_fmt) {
        /* Still allow, but we won't know format; best-effort. */
        fmt_audio_format = 1;
        fmt_channels = 1;
        fmt_sample_rate = 48000;
        fmt_bits = 16;
      }
      r->data_remaining_bytes = chunk_size;
      have_data = 1;
      break;
    } else {
      /* skip unknown chunk */
      if (fseek(r->f, (long)chunk_size, SEEK_CUR) != 0) die("invalid WAV: fseek failed");
    }

    /* chunks are word-aligned */
    if (chunk_size & 1) {
      if (fseek(r->f, 1, SEEK_CUR) != 0) die("invalid WAV: pad fseek failed");
    }
  }

  if (!have_data) die("invalid WAV: no data");
  if (fmt_audio_format != 1) die("unsupported WAV: only PCM (format=1) supported");
  if (fmt_channels != 1) die("unsupported WAV: only mono supported");
  if (fmt_bits != 16) die("unsupported WAV: only 16-bit supported");

  r->is_wav = 1;
  r->sample_rate = fmt_sample_rate;
  r->channels = fmt_channels;
  r->bits_per_sample = fmt_bits;
}

static void audio_reader_close(AudioReader *r) {
  if (r->f) fclose(r->f);
  r->f = NULL;
}

/* Reads up to n int16 samples. Returns number of samples actually read. */
static size_t audio_reader_read_s16(AudioReader *r, int16_t *dst, size_t n) {
  size_t want_bytes = n * sizeof(int16_t);
  size_t got;

  if (r->is_wav) {
    if (r->data_remaining_bytes == 0) return 0;
    if (want_bytes > r->data_remaining_bytes) want_bytes = r->data_remaining_bytes;
  }

  got = fread(dst, 1, want_bytes, r->f);
  if (got % sizeof(int16_t) != 0) {
    /* Truncate partial sample */
    got -= got % sizeof(int16_t);
    if (fseek(r->f, -(long)(got % sizeof(int16_t)), SEEK_CUR) != 0) {
      /* ignore */
    }
  }
  if (r->is_wav) {
    r->data_remaining_bytes -= (uint32_t)got;
  }
  return got / sizeof(int16_t);
}

static void print_csv_header(FILE *out) {
  int i;
  fprintf(out, "frame,pitch_index,silence");
  for (i = 0; i < NB_FEATURES; i++) fprintf(out, ",f%d", i);
  for (i = 0; i < NB_BANDS; i++) fprintf(out, ",Ex%d", i);
  for (i = 0; i < NB_BANDS; i++) fprintf(out, ",Exp%d", i);
  fprintf(out, "\n");
}

int main(int argc, char **argv) {
  const char *in_path = NULL;
  const char *out_path = NULL;
  int force_pcm = 0;
  int max_frames = -1;

  int argi = 1;
  while (argi < argc) {
    if (strcmp(argv[argi], "--pcm") == 0) {
      force_pcm = 1;
      argi++;
    } else if (strcmp(argv[argi], "-o") == 0 || strcmp(argv[argi], "--out") == 0) {
      if (argi + 1 >= argc) die("missing value for -o/--out");
      out_path = argv[argi + 1];
      argi += 2;
    } else if (strcmp(argv[argi], "--max_frames") == 0) {
      if (argi + 1 >= argc) die("missing value for --max_frames");
      max_frames = atoi(argv[argi + 1]);
      argi += 2;
    } else if (starts_with(argv[argi], "-") && strcmp(argv[argi], "-") != 0) {
      die("usage: dump_single_features [--pcm] [-o out.csv] [--max_frames N] <input.wav|input.pcm>");
    } else {
      in_path = argv[argi];
      argi++;
    }
  }

  if (!in_path) {
    die("usage: dump_single_features [--pcm] [-o out.csv] [--max_frames N] <input.wav|input.pcm>");
  }

  FILE *out = stdout;
  if (out_path && strcmp(out_path, "-") != 0) {
    out = fopen(out_path, "wb");
    if (!out) die_errno("cannot open output");
  }

  AudioReader reader;
  if (!force_pcm) {
    /* Probe for WAV header. */
    FILE *probe = fopen(in_path, "rb");
    uint8_t magic[12];
    int is_wav = 0;
    if (probe) {
      size_t n = fread(magic, 1, sizeof(magic), probe);
      if (n == sizeof(magic) && memcmp(magic, "RIFF", 4) == 0 && memcmp(magic + 8, "WAVE", 4) == 0) {
        is_wav = 1;
      }
      fclose(probe);
    }
    if (is_wav) audio_reader_open_wav(&reader, in_path);
    else audio_reader_open_pcm(&reader, in_path);
  } else {
    audio_reader_open_pcm(&reader, in_path);
  }

  if (reader.sample_rate != 48000) {
    fprintf(stderr,
            "WARNING: input sample_rate=%u (expected 48000). Features assume 48 kHz; results may be misleading.\n",
            reader.sample_rate);
  }

  /* Initialize state for feature extraction */
  DenoiseState *st = pasdnet_create(NULL);
  if (!st) die("pasdnet_create failed");

  static const float a_hp[2] = {-1.99599f, 0.99600f};
  static const float b_hp[2] = {-2.f, 1.f};
  float hp_mem[2] = {0, 0};

  int16_t s16[FRAME_SIZE];
  float in_f[FRAME_SIZE];
  float x[FRAME_SIZE];
  kiss_fft_cpx X[FREQ_SIZE];
  kiss_fft_cpx P[FREQ_SIZE];
  float Ex[NB_BANDS];
  float Ep[NB_BANDS];
  float Exp[NB_BANDS];
  float features[NB_FEATURES];

  print_csv_header(out);

  int frame = 0;
  while (1) {
    size_t got = audio_reader_read_s16(&reader, s16, FRAME_SIZE);
    if (got == 0) break;
    for (size_t i = 0; i < FRAME_SIZE; i++) {
      float v = 0.f;
      if (i < got) v = (float)s16[i];
      in_f[i] = v;
    }

    /* Apply same HP filter as pasdnet_process_frame, but with our own state */
    rnn_biquad(x, hp_mem, in_f, b_hp, a_hp, FRAME_SIZE);

    int silence = rnn_compute_frame_features(st, X, P, Ex, Ep, Exp, features, x);

    /* Reconstruct pitch_index from feature[64] */
    float pitch_feat = features[2 * NB_BANDS];
    int pitch_index = (int)lroundf(100.f * pitch_feat + 300.f);

    fprintf(out, "%d,%d,%d", frame, pitch_index, silence);
    for (int i = 0; i < NB_FEATURES; i++) fprintf(out, ",%.8g", features[i]);
    for (int i = 0; i < NB_BANDS; i++) fprintf(out, ",%.8g", Ex[i]);
    for (int i = 0; i < NB_BANDS; i++) fprintf(out, ",%.8g", Exp[i]);
    fprintf(out, "\n");

    frame++;
    if (max_frames >= 0 && frame >= max_frames) break;

    /* If we got a partial tail frame, stop after dumping it once. */
    if (got < FRAME_SIZE) break;
  }

  pasdnet_destroy(st);
  audio_reader_close(&reader);
  if (out != stdout) fclose(out);

  return 0;
}
