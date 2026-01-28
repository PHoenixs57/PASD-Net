#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include "pasdnet.h"
#include "common.h"
#include "denoise.h"
#include "arch.h"
#include "kiss_fft.h"
#include "src/_kiss_fft_guts.h"

/*
   What this file does (high level):
   - Read paired 16-bit PCM streams: clean speech and noisy speech.
   - Randomly crop a segment and apply data augmentation (filtering, loudness scaling,
     reverberation, clipping, quantization, etc.).
   - Produce a noisy waveform xn and its clean target x.
   - For each frame: extract features, compute the ideal per-band gain g, and the VAD
     target vad_target.
   - Write (features, g, vad_target) sequentially to the output file as training data.
 */

/*
  lowpass / band_lp are used to randomly select the maximum frequency to train on.
  For bands above that cutoff, the label g is set to -1 (ignored / not trained).
 */
int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;

/*
  Number of frames per SEQUENCE (RNNoise-style framing is 10 ms per frame).
  Here 1000 frames ≈ 10 seconds of audio.
 */
#define SEQUENCE_LENGTH 1000
#define SEQUENCE_SAMPLES (SEQUENCE_LENGTH*FRAME_SIZE)

/*
  RIR parameters: used for frequency-domain convolution to simulate room reverberation.
 */
#define RIR_FFT_SIZE 65536
#define RIR_MAX_DURATION (RIR_FFT_SIZE/2)
#define FILENAME_MAX_SIZE 1000

/*
  rir_list stores multiple room impulse responses (RIRs).
  - rir: full reverberation
  - early: early reflections only
 */
struct rir_list {
  int nb_rirs;           /* The number of RIRs */
  kiss_fft_state *fft;   /* Shared FFT Plan */
  kiss_fft_cpx **rir;    /* Complete frequency-domain representation of RIR */
  kiss_fft_cpx **early;  /* Frequency domain representation of the early part */
};

/*
  Read one RIR file in the time domain and FFT it to the frequency domain.
  When early != 0, keep only a short decaying tail (samples 480~720) to approximate
  "early reflections".
 */
kiss_fft_cpx *load_rir(const char *rir_file, kiss_fft_state *fft, int early) {
  kiss_fft_cpx *x, *X;
  float rir[RIR_MAX_DURATION];
  int len;
  int i;
  FILE *f;
  f = fopen(rir_file, "rb");
  if (f==NULL) {
    fprintf(stderr, "cannot open %s: %s\n", rir_file, strerror(errno));
    exit(1);
  }
  x = (kiss_fft_cpx*)calloc(fft->nfft, sizeof(*x));
  X = (kiss_fft_cpx*)calloc(fft->nfft, sizeof(*X));
  len = fread(rir, sizeof(*rir), RIR_MAX_DURATION, f);
  if (early) {
    for (i=0;i<240;i++) {
      rir[480+i] *= (1 - i/240.f);
    }
    RNN_CLEAR(&rir[240+480], RIR_MAX_DURATION-240-480);
  }
  for (i=0;i<len;i++) x[i].r = rir[i];
  rnn_fft_c(fft, x, X);
  free(x);
  fclose(f);
  return X;
}

/*
  Read a list of RIR filenames from a text file and load each via load_rir(),
  producing a set of room responses to sample from randomly.
 */
void load_rir_list(const char *list_file, struct rir_list *rirs) {
  int allocated;
  char rir_filename[FILENAME_MAX_SIZE];
  FILE *f;
  f = fopen(list_file, "rb");
  if (f==NULL) {
    fprintf(stderr, "cannot open %s: %s\n", list_file, strerror(errno));
    exit(1);
  }
  rirs->nb_rirs = 0;
  allocated = 2;
  rirs->fft = rnn_fft_alloc_twiddles(RIR_FFT_SIZE, NULL, NULL, NULL, 0);
  rirs->rir = malloc(allocated*sizeof(rirs->rir[0]));
  rirs->early = malloc(allocated*sizeof(rirs->early[0]));
  while (fgets(rir_filename, FILENAME_MAX_SIZE, f) != NULL) {
    /* Chop trailing newline. */
    rir_filename[strcspn(rir_filename, "\n")] = 0;
    if (rirs->nb_rirs+1 > allocated) {
      allocated *= 2;
      rirs->rir = realloc(rirs->rir, allocated*sizeof(rirs->rir[0]));
      rirs->early = realloc(rirs->early, allocated*sizeof(rirs->early[0]));
    }
    rirs->rir[rirs->nb_rirs] = load_rir(rir_filename, rirs->fft, 0);
    rirs->early[rirs->nb_rirs] = load_rir(rir_filename, rirs->fft, 1);
    rirs->nb_rirs++;
  }
  fclose(f);
}

/*
  Block-wise frequency-domain convolution of the full sequence with the selected RIR:
  - early != 0: use the early-reflections version
  - early == 0: use the full RIR
 */
void rir_filter_sequence(const struct rir_list *rirs, float *audio, int rir_id, int early) {
  int i;
  kiss_fft_cpx x[RIR_FFT_SIZE] = {{0,0}};
  kiss_fft_cpx y[RIR_FFT_SIZE] = {{0,0}};
  kiss_fft_cpx X[RIR_FFT_SIZE] = {{0,0}};
  const kiss_fft_cpx *Y;
  if (early) Y = rirs->early[rir_id];
  else Y = rirs->rir[rir_id];
  i=0;
  while (i<SEQUENCE_SAMPLES) {
    int j;
    RNN_COPY(&x[0], &x[RIR_FFT_SIZE/2], RIR_FFT_SIZE/2);
    for (j=0;j<IMIN(SEQUENCE_SAMPLES-i, RIR_FFT_SIZE/2);j++) x[RIR_FFT_SIZE/2+j].r = audio[i+j];
    for (;j<RIR_FFT_SIZE/2;j++) x[RIR_FFT_SIZE/2+j].r = 0;
    rnn_fft_c(rirs->fft, x, X);
    for (j=0;j<RIR_FFT_SIZE;j++) {
      kiss_fft_cpx tmp;
      C_MUL(tmp, X[j], Y[j]);
      X[j].r = tmp.r*RIR_FFT_SIZE/2;
      X[j].i = tmp.i*RIR_FFT_SIZE/2;
    }
    rnn_ifft_c(rirs->fft, X, y);
    for (j=0;j<IMIN(SEQUENCE_SAMPLES-i, RIR_FFT_SIZE/2);j++) audio[i+j] = y[RIR_FFT_SIZE/2+j].r;
    i += RIR_FFT_SIZE/2;
  }
}

/* Simple LCG RNG, controlled separately from rand(). */
static unsigned rand_lcg(unsigned *seed) {
  *seed = 1664525**seed + 1013904223;
  return *seed;
}

/* Uniform random in [-0.5, 0.5). */
static float uni_rand() {
  return rand()/(double)RAND_MAX-.5;
}

/* Uniform random float in [0, f). */
static float randf(float f) {
  return f*rand()/(double)RAND_MAX;
}

/*
  Randomly generate the feedback (a) coefficients for a 2nd-order filter.
  Sometimes returns zeros (no filtering); otherwise returns a conjugate-pole pair to
  "color" the signal.
 */
static void rand_filt(float *a) {
  if (rand()%3!=0) {
    a[0] = a[1] = 0;
  }
  else if (uni_rand()>0) {
    float r, theta;
    r = rand()/(double)RAND_MAX;
    r = .7*r*r;
    theta = rand()/(double)RAND_MAX;
    theta = M_PI*theta*theta;
    a[0] = -2*r*cos(theta);
    a[1] = r*r;
  } else {
    float r0,r1;
    r0 = 1.4*uni_rand();
    r1 = 1.4*uni_rand();
    a[0] = -r0-r1;
    a[1] = r0*r1;
  }
}

/* Randomly generate a filter pair (a/b): feedback and feedforward for an IIR. */
static void rand_resp(float *a, float *b) {
  rand_filt(a);
  rand_filt(b);
}

/*
  Global buffers:
  - speech16 / noisy16: raw 16-bit PCM read from files
  - x: processed clean speech
  - xn: processed noisy speech
 */
short speech16[SEQUENCE_LENGTH*FRAME_SIZE];
short noisy16[SEQUENCE_LENGTH*FRAME_SIZE];
float x[SEQUENCE_LENGTH*FRAME_SIZE];
float xn[SEQUENCE_LENGTH*FRAME_SIZE];

#define P00 0.99f
#define P01 0.01f
#define P10 0.01f
#define P11 0.99f
#define LOGIT_SCALE 0.5f

/*
  Given per-frame energy E, estimate speech energy Esig and noise energy Enoise.
  Then run Viterbi on a 2-state HMM (P00~P11) to produce a smoothed VAD sequence.
  Output vad[frame] is 0/1 indicating whether the frame is considered speech.
 */
static void viterbi_vad(const float *E, int *vad) {
  int i;
  float Enoise, Esig;
  int back[SEQUENCE_LENGTH][2];
  float curr;
  Enoise = Esig = 1e-30;
  for (i=0;i<SEQUENCE_LENGTH;i++) {
    Esig += E[i]*E[i];
  }
  Esig = sqrt(Esig/SEQUENCE_LENGTH);
  for (i=0;i<SEQUENCE_LENGTH;i++) {
    Enoise += 1.f/(1e-8*Esig*Esig + E[i]*E[i]);
  }
  Enoise = 1.f/sqrt(Enoise/SEQUENCE_LENGTH);
  curr = 0.5;
  for (i=0;i<SEQUENCE_LENGTH;i++) {
    float p0, pspeech, pnoise;
    float prior;
    p0 = (log(1e-15+E[i]) - log(Enoise))/(.01 + log(Esig) - log(Enoise));
    p0 = MIN16(.9f, MAX16(.1f, p0));
    p0 = 1.f/(1.f + pow((1.f-p0)/p0, LOGIT_SCALE));
    if (curr*P11 > (1-curr)*P01) {
      back[i][1] = 1;
      prior = curr*P11;
    } else {
      back[i][1] = 0;
      prior = (1-curr)*P01;
    }
    pspeech = prior*p0;

    if ((1-curr)*P00 > curr*P10) {
      back[i][0] = 0;
      prior = (1-curr)*P00;
    } else {
      back[i][0] = 1;
      prior = curr*P10;
    }
    pnoise = prior*(1-p0);
    curr = pspeech / (pspeech + pnoise);
    /*printf("%f ", curr);*/
  }
  vad[SEQUENCE_LENGTH-1] = curr > .5;
  for (i=SEQUENCE_LENGTH-2;i>=0;i--) {
    if (vad[i+1]) {
      vad[i] = back[i+1][1];
    } else {
      vad[i] = back[i+1][0];
    }
  }
  for (i=0;i<SEQUENCE_LENGTH-1;i++) {
    if (vad[i+1]) vad[i] = 1;
  }
  for (i=SEQUENCE_LENGTH-1;i>=1;i--) {
    if (vad[i-1]) vad[i] = 1;
  }
}

/*
  Use the VAD result to fade clean speech x to zero during non-speech regions.
  This makes silence in the target truly silent and avoids leaking background into
  the training target.
 */
static void clear_vad(float *x, int *vad) {
  int i;
  int active = vad[0];
  for (i=0;i<SEQUENCE_LENGTH;i++) {
    if (!active) {
      if (i<SEQUENCE_LENGTH-1 && vad[i+1]) {
        int j;
        for (j=0;j<FRAME_SIZE;j++) x[i*FRAME_SIZE+j] *= j/(float)FRAME_SIZE;
        active = 1;
      } else {
        RNN_CLEAR(&x[i*FRAME_SIZE], FRAME_SIZE);
      }
    } else {
      if (i>=1 && vad[i]==0 && vad[i-1]==0) {
        int j;
        for (j=0;j<FRAME_SIZE;j++) x[i*FRAME_SIZE+j] *= 1.f - j/(float)FRAME_SIZE;
        active = 0;
      }
    }
  }
}

/*
  Compute a weighted RMS after a fixed filter.
  This correlates better with perceived loudness than raw energy and is used for
  automatic gain scaling.
 */
static float weighted_rms(float *x) {
  int i;
  float tmp[SEQUENCE_SAMPLES];
  float weighting_b[2] = {-2.f, 1.f};
  float weighting_a[2] = {-1.89f, .895f};
  float mem[2] = {0};
  float mse = 1e-15f;
  rnn_biquad(tmp, mem, x, weighting_b, weighting_a, SEQUENCE_SAMPLES);
  for (i=0;i<SEQUENCE_SAMPLES;i++) mse += tmp[i]*tmp[i];
  return 0.9506*sqrt(mse/SEQUENCE_SAMPLES);
}

/*
   Program entry point.

   Usage:
     dump_features [-rir_list list] speech.pcm noisy.pcm output.bin count

   Generate 'count' sequences (each ~10 seconds) and write per-frame
   (features, g, vad_target) to output.bin.
 */
int main(int argc, char **argv) {
  int i, j;
  int count=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float speech_gain = 1;
  FILE *f1, *f2, *fout;
  long speech_length, noisy_length;
  int maxCount;
  unsigned seed;
  DenoiseState *st;
  DenoiseState *noisy;
  char *argv0;
  char *rir_filename = NULL;
  struct rir_list rirs;
  /* Initialize RNG seed from PID so each run differs slightly. */
  seed = getpid();
  srand(seed);
  /* st: analysis state for clean speech; noisy: analysis state for noisy speech/features. */
  st = pasdnet_create(NULL);
  noisy = pasdnet_create(NULL);
  argv0 = argv[0];
  while (argc>6) {
    if (strcmp(argv[1], "-rir_list")==0) {
      rir_filename = argv[2];
      argv+=2;
      argc-=2;
    }
  }
  /*
     Argument check: besides optional -rir_list, 4 positional args are required.
     Inputs are paired clean speech and the corresponding original noisy speech.
   */
  if (argc!=5) {
    fprintf(stderr, "usage: %s [-rir_list list] <speech> <noisy> <output> <count>\n", argv0);
    return 1;
  }
  /* Open input clean speech, input noisy speech, and output file. */
  f1 = fopen(argv[1], "rb");
  f2 = fopen(argv[2], "rb");
  fout = fopen(argv[3], "wb");

  fseek(f1, 0, SEEK_END);
  speech_length = ftell(f1);
  fseek(f1, 0, SEEK_SET);
  
  fseek(f2, 0, SEEK_END);
  noisy_length = ftell(f2);
  fseek(f2, 0, SEEK_SET);

  maxCount = atoi(argv[4]);
  /* If an RIR list is provided, load all RIRs up front for random sampling. */
  if (rir_filename) load_rir_list(rir_filename, &rirs);
  for (count=0;count<maxCount;count++) {
    int rir_id;
    int vad[SEQUENCE_LENGTH];
    long speech_pos;
    int start_pos=0;
    float E[SEQUENCE_LENGTH] = {0};
    float mem[2]={0};
    int frame;
    int silence;
    kiss_fft_cpx X[FREQ_SIZE], Y[FREQ_SIZE], P[WINDOW_SIZE];
    float Ex[NB_BANDS], Ey[NB_BANDS], Ep[NB_BANDS];
    float Exp[NB_BANDS];
    float features[NB_FEATURES];
    float g[NB_BANDS];
    float speech_rms;
    /* Print progress to stderr every 1000 sequences. */
    if ((count%1000)==0) fprintf(stderr, "%d\r", count);
    {
      long max_pos;
      max_pos = IMIN(speech_length, noisy_length) - (long)sizeof(speech16);
      if (max_pos < 0) max_pos = 0;
      speech_pos = (rand_lcg(&seed)*2.3283e-10)*max_pos;
      if (speech_pos > max_pos) speech_pos = max_pos;
    }
    speech_pos -= speech_pos&1;
    fseek(f1, speech_pos, SEEK_SET);
    fseek(f2, speech_pos, SEEK_SET);
    fread(speech16, sizeof(speech16), 1, f1);
    fread(noisy16, sizeof(noisy16), 1, f2);
    /*
       start_pos controls how many initial samples force VAD=0.
       With some probability, we force an initial silence region to increase diversity.
     */
    if (rand()%4) start_pos = 0;
    else start_pos = -(int)(1000*log(rand()/(float)RAND_MAX));
    start_pos = IMIN(start_pos, SEQUENCE_LENGTH*FRAME_SIZE);

    /*
       Randomly choose an overall level (dB) and scale clean and noisy together.
       This preserves the pairing relationship.
     */
    speech_gain = pow(10., (-45+randf(45.f)+randf(10.f))/20.);
    rand_resp(a_sig, b_sig);
    /*
       Randomly choose a maximum frequency (lowpass), roughly from ~3 kHz upward.
       Bands above this cutoff are masked out in the training labels.
     */
    lowpass = FREQ_SIZE * 3000./24000. * pow(50., rand()/(double)RAND_MAX);
    for (i=0;i<NB_BANDS;i++) {
      if (eband20ms[i] > lowpass) {
        band_lp = i;
        break;
      }
    }

    /*
       Compute per-frame energy E[frame] from the 16-bit clean speech,
       and convert short -> float into x / xn.
     */
    for (frame=0;frame<SEQUENCE_LENGTH;frame++) {
      E[frame] = 0;   
      for(j=0;j<FRAME_SIZE;j++) {
        float s = speech16[frame*FRAME_SIZE+j];
        E[frame] += s*s;
        x[frame*FRAME_SIZE+j] = speech16[frame*FRAME_SIZE+j];
        xn[frame*FRAME_SIZE+j] = noisy16[frame*FRAME_SIZE+j];
      }
    }
    /* Compute a smoothed VAD track from the energy trajectory. */
    viterbi_vad(E, vad);

    RNN_CLEAR(mem, 2);
    rnn_biquad(x, mem, x, b_hp, a_hp, SEQUENCE_LENGTH*FRAME_SIZE);
    RNN_CLEAR(mem, 2);
    rnn_biquad(x, mem, x, b_sig, a_sig, SEQUENCE_LENGTH*FRAME_SIZE);
    RNN_CLEAR(mem, 2);
    rnn_biquad(xn, mem, xn, b_hp, a_hp, SEQUENCE_LENGTH*FRAME_SIZE);
    RNN_CLEAR(mem, 2);
    rnn_biquad(xn, mem, xn, b_sig, a_sig, SEQUENCE_LENGTH*FRAME_SIZE);

    /* Compute weighted RMS for loudness normalization (based on clean speech). */
    speech_rms = weighted_rms(x);

    /*
       Force VAD=0 for the first start_pos samples,
       then apply VAD-based fading to make non-speech clean regions silent.
     */
    RNN_CLEAR(vad, start_pos/FRAME_SIZE);
    clear_vad(x, vad);

    /* Adjust final gain based on RMS to avoid too loud/quiet (scale x and xn together). */
    speech_gain *= 3000.f/(1+speech_rms);
    for (j=0;j<SEQUENCE_SAMPLES;j++) {
      x[j] *= speech_gain;
      xn[j] *= speech_gain;
    }
    /* Randomly decide whether to add room reverberation for this sequence. */
    if (rir_filename && rand()%2==0) {
      rir_id = rand()%rirs.nb_rirs;
      rir_filter_sequence(&rirs, x, rir_id, 1);
      rir_filter_sequence(&rirs, xn, rir_id, 0);
    }
    if (rand()%4==0) {
      /* Simulate front-end saturation/clipping: limit xn to [-32767, 32767]. */
      for (j=0;j<SEQUENCE_SAMPLES;j++) {
        xn[j] = MIN16(32767.f, MAX16(-32767.f, xn[j]));
      }
    }
    if (rand()%2==0) {
      /* Simulate 16-bit quantization: round to nearest integer. */
      for (j=0;j<SEQUENCE_SAMPLES;j++) {
        xn[j] = floor(.5f + xn[j]);
      }
    }
    /*
       For each frame:
       - Compute clean-band energies Ey from clean speech x.
       - Extract features from noisy speech xn and compute Ex / Ep / Exp.
       - Compute target gain g and vad_target, then write to output.
     */
    for (frame=0;frame<SEQUENCE_LENGTH;frame++) {
      float vad_target;
      /*
         Analyze band energies on clean speech;
         extract features and band energies/correlations on noisy speech.
       */
      rnn_frame_analysis(st, Y, Ey, &x[frame*FRAME_SIZE]);
      silence = rnn_compute_frame_features(noisy, X, P, Ex, Ep, Exp, features, &xn[frame*FRAME_SIZE]);
      /*rnn_pitch_filter(X, P, Ex, Ep, Exp, g);*/
      vad_target = vad[frame];
      for (i=0;i<NB_BANDS;i++) {
        /* Ideal gain: want |Y| ≈ g * |X|, so use sqrt(energy ratio). */
        g[i] = sqrt((Ey[i]+1e-3)/(Ex[i]+1e-3));
        if (g[i] > 1) g[i] = 1;  /* No amplification, only attenuation. */
        /* Do not train this band under these conditions; use -1 as an ignore marker. */
        if (silence || i > band_lp) g[i] = -1;  /* Silent frame or above random lowpass. */
        if (Ey[i] < 5e-2 && Ex[i] < 5e-2) g[i] = -1;  /* Both close to noise floor. */
      }
#if 0
      {
        short tmp[FRAME_SIZE];
        for (j=0;j<FRAME_SIZE;j++) tmp[j] = MIN16(32767, MAX16(-32767, xn[frame*FRAME_SIZE+j]));
        fwrite(tmp, FRAME_SIZE, 2, fout);
      }
#endif
#if 1
      /* Write in order: features, per-band target gains, and frame VAD label. */
      fwrite(features, sizeof(float), NB_FEATURES, fout);
      fwrite(g, sizeof(float), NB_BANDS, fout);
      fwrite(&vad_target, sizeof(float), 1, fout);
#endif
    }
  }

  fclose(f1);
  fclose(f2);
  fclose(fout);
  return 0;
}
