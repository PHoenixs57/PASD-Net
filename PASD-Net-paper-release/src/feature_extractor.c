// feature_extractor.c
#include <stdlib.h>
#include "pasdnet.h"
#include "denoise.h"
#include "common.h"

#define FRAME_SIZE 480  // 根据PASDNet实际定义调整
#define SEQUENCE_LENGTH 2000
#define NB_FEATURES 65  // 特征维度
#define NB_BANDS 32     // 增益维度

// 特征提取函数
void extract_features(
    const short* speech,    // 语音数据 (SEQUENCE_LENGTH*FRAME_SIZE)
    const short* noise,     // 噪声数据
    const short* fgnoise,   // 前景噪声数据
    float* features,        // 输出特征 (SEQUENCE_LENGTH * NB_FEATURES)
    float* gain,            // 输出增益目标 (SEQUENCE_LENGTH * NB_BANDS)
    float* vad              // 输出VAD目标 (SEQUENCE_LENGTH)
) {
    // 初始化PASDNet状态
    DenoiseState *st = pasdnet_create(NULL);
    DenoiseState *noisy = pasdnet_create(NULL);
    
    // 临时缓冲区
    float x[SEQUENCE_LENGTH*FRAME_SIZE];
    float n[SEQUENCE_LENGTH*FRAME_SIZE];
    float fn[SEQUENCE_LENGTH*FRAME_SIZE];
    float xn[SEQUENCE_LENGTH*FRAME_SIZE];
    int vad_int[SEQUENCE_LENGTH];
    float E[SEQUENCE_LENGTH] = {0};

    // 转换音频格式并计算能量
    for (int frame = 0; frame < SEQUENCE_LENGTH; frame++) {
        E[frame] = 0;
        for (int j = 0; j < FRAME_SIZE; j++) {
            int idx = frame*FRAME_SIZE + j;
            x[idx] = speech[idx];
            n[idx] = noise[idx];
            fn[idx] = fgnoise[idx];
            E[frame] += x[idx] * x[idx];
        }
    }

    // 计算VAD（复用原逻辑）
    viterbi_vad(E, vad_int);
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
        vad[i] = vad_int[i];
    }

    // 预处理（滤波、增益调整等，复用原逻辑）
    preprocess_audio(x, n, fn, xn, vad_int, SEQUENCE_LENGTH*FRAME_SIZE);

    // 提取特征和目标
    for (int frame = 0; frame < SEQUENCE_LENGTH; frame++) {
        float X[FREQ_SIZE], P[WINDOW_SIZE];
        float Ex[NB_BANDS], Ep[NB_BANDS], Exp[NB_BANDS];
        float Y[FREQ_SIZE], Ey[NB_BANDS];
        float g[NB_BANDS];

        // 分析干净语音和带噪语音
        rnn_frame_analysis(st, Y, Ey, &x[frame*FRAME_SIZE]);
        rnn_compute_frame_features(noisy, X, P, Ex, Ep, Exp, 
                                 &features[frame*NB_FEATURES], 
                                 &xn[frame*FRAME_SIZE]);

        // 计算目标增益
        compute_target_gain(g, Ey, Ex, vad_int[frame], NB_BANDS);
        for (int i = 0; i < NB_BANDS; i++) {
            gain[frame*NB_BANDS + i] = g[i];
        }
    }

    pasdnet_destroy(st);
    pasdnet_destroy(noisy);
}

// 辅助函数（从dump_features.c迁移必要实现）
void viterbi_vad(const float *E, int *vad) {
    // 复用原viterbi_vad实现
}

void preprocess_audio(float *x, float *n, float *fn, float *xn, int *vad, int len) {
    // 复用原预处理逻辑（滤波、增益调整等）
}

void compute_target_gain(float *g, float *Ey, float *Ex, int vad, int nb_bands) {
    // 复用原增益计算逻辑
}