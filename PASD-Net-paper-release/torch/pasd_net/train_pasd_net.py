import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import os
import pasd_net
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('features', type=str, help='path to feature file in .f32 format')
parser.add_argument('output', type=str, help='path to output folder')

parser.add_argument('--suffix', type=str, help="model name suffix", default="")
parser.add_argument('--cuda-visible-devices', type=str, help="comma separates list of cuda visible device indices, default: CUDA_VISIBLE_DEVICES", default=None)


model_group = parser.add_argument_group(title="model parameters")
model_group.add_argument('--cond-size', type=int, help="first conditioning size, default: 128", default=128)
model_group.add_argument('--gru-size', type=int, help="first conditioning size, default: 384", default=384)

training_group = parser.add_argument_group(title="training parameters")
training_group.add_argument('--batch-size', type=int, help="batch size, default: 128", default=128)
training_group.add_argument('--lr', type=float, help='learning rate, default: 5e-4', default=5e-4)
training_group.add_argument('--epochs', type=int, help='number of training epochs, default: 50', default=50)
training_group.add_argument('--sequence-length', type=int, help='sequence length, default: 2000', default=2000)
training_group.add_argument('--lr-decay', type=float, help='learning rate decay factor, default: 5e-5', default=5e-5)
training_group.add_argument('--initial-checkpoint', type=str, help='initial checkpoint to start training from, default: None', default=None)
training_group.add_argument('--gamma', type=float, help='perceptual exponent (default 0.25)', default=0.25)
training_group.add_argument('--sparse', action='store_true')

args = parser.parse_args()



class PASDNetDataset(torch.utils.data.Dataset):
    def __init__(self,
                features_file,
                sequence_length=2000):

        self.sequence_length = sequence_length

        self.data = np.memmap(features_file, dtype='float32', mode='r')
        dim = 98

        self.nb_sequences = self.data.shape[0]//self.sequence_length//dim
        self.data = self.data[:self.nb_sequences*self.sequence_length*dim]

        self.data = np.reshape(self.data, (self.nb_sequences, self.sequence_length, dim))

    def __len__(self):
        return self.nb_sequences

    def __getitem__(self, index):
        return self.data[index, :, :65].copy(), self.data[index, :, 65:-1].copy(), self.data[index, :, -1:].copy()

def mask(g):
    return torch.clamp(g+1, max=1)

adam_betas = [0.8, 0.98]
adam_eps = 1e-8
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
sequence_length = args.sequence_length
lr_decay = args.lr_decay

cond_size  = args.cond_size
gru_size  = args.gru_size

checkpoint_dir = os.path.join(args.output, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = dict()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

checkpoint['model_args']    = ()
# ------------------------------------------------------------------------------
checkpoint['model_kwargs']  = {'cond_size': cond_size, 'gru_size': gru_size, 'use_attention': True}
# ------------------------------------------------------------------------------
model = pasd_net.PASDNet(*checkpoint['model_args'], **checkpoint['model_kwargs'])

if type(args.initial_checkpoint) != type(None):
    checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)

checkpoint['state_dict']    = model.state_dict()

dataset = PASDNetDataset(args.features)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)


optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps)

# 全局 step 计数，用于 warmup
global_step = 0
warmup_steps = 5000  # 可以根据数据量再调，大一点更稳

def lr_lambda(step):
    # 先 warmup，再按原来的 1 / (1 + decay * x) 衰减
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    # 原来的 lr_decay 调度逻辑
    return 1.0 / (1.0 + lr_decay * (step - warmup_steps))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

gamma = args.gamma

train_loss_history = []
train_gain_loss_history = []
train_vad_loss_history = []

if __name__ == '__main__':
    model.to(device)
    states = None
    for epoch in range(1, epochs + 1):

        running_gain_loss = 0
        running_vad_loss = 0
        running_loss = 0

        print(f"training epoch {epoch}...")
        with tqdm.tqdm(dataloader, unit='batch') as tepoch:
            for i, (features, gain, vad) in enumerate(tepoch):
                optimizer.zero_grad()
                features = features.to(device)
                gain = gain.to(device)
                vad = vad.to(device)

                pred_gain, pred_vad, states = model(features, states=states)
                states = [state.detach() for state in states]
                gain = gain[:,3:-1,:]
                vad = vad[:,3:-1,:]
                target_gain = torch.clamp(gain, min=0)
                target_gain = target_gain*(torch.tanh(8*target_gain)**2)

                e = pred_gain**gamma - target_gain**gamma
                gain_loss = torch.mean((1+5.*vad)*mask(gain)*(e**2))
                #vad_loss = torch.mean(torch.abs(2*vad-1)*(vad-pred_vad)**2)
                vad_loss = torch.mean(torch.abs(2*vad-1)*(-vad*torch.log(.01+pred_vad) - (1-vad)*torch.log(1.01-pred_vad)))
                loss = gain_loss + .001*vad_loss

                loss.backward()
                # 梯度裁剪，防止偶尔梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                if args.sparse:
                    model.sparsify()

                global_step += 1
                scheduler.step()

                running_gain_loss += gain_loss.detach().cpu().item()
                running_vad_loss += vad_loss.detach().cpu().item()
                running_loss += loss.detach().cpu().item()
                tepoch.set_postfix(loss=f"{running_loss/(i+1):8.5f}",
                                   gain_loss=f"{running_gain_loss/(i+1):8.5f}",
                                   vad_loss=f"{running_vad_loss/(i+1):8.5f}",
                                   )

        # 计算本轮平均损失
        avg_gain_loss = running_gain_loss / len(dataloader)
        avg_vad_loss = running_vad_loss / len(dataloader)
        avg_loss = running_loss / len(dataloader)

        train_loss_history.append(avg_loss)
        train_gain_loss_history.append(avg_gain_loss)
        train_vad_loss_history.append(avg_vad_loss)

        # save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'pasd_net{args.suffix}_{epoch}.pth')
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['loss'] = avg_loss
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, checkpoint_path)

    # --- 训练结束后画损失曲线 ---
    epochs_axis = range(1, epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, train_loss_history, label='total loss')
    plt.plot(epochs_axis, train_gain_loss_history, label='gain loss')
    plt.plot(epochs_axis, train_vad_loss_history, label='vad loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)

    # 保存到输出目录（和 checkpoints 同级）
    fig_path = os.path.join(args.output, f'train_loss_curves{args.suffix}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练损失曲线已保存到: {fig_path}")