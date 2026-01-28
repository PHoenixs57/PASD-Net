#ifndef RNN_H_
#define RNN_H_

#include "pasdnet.h"
#include "pasdnet_data.h"

#include "opus_types.h"

#define WEIGHTS_SCALE (1.f/256)

#define MAX_NEURONS 1024


typedef struct {
  float conv1_state[CONV1_STATE_SIZE];
  float conv2_state[CONV2_STATE_SIZE];
  float gru1_state[GRU1_STATE_SIZE];
  float gru2_state[GRU2_STATE_SIZE];
  float gru3_state[GRU3_STATE_SIZE];
} RNNState;
void compute_rnn(const PASDNet *model, RNNState *rnn, float *gains, float *vad, const float *input, int arch);

#endif /* RNN_H_ */
