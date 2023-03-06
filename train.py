import argparse
import json
import os

import git
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import Config
from wgvc import WhisperGuidedVC
from utils.realtime import RealtimeWavDataset
from utils.wrapper import TrainingWrapper

import speechset


class Trainer:
    """TacoSpawn trainer.
    """
    LOG_IDX = 0
    EVAL_MAX_SEC = 2
    EVAL_INTVAL = 4

    def __init__(self,
                 model: WhisperGuidedVC,
                 dataset: speechset.speeches.SpeechSet,
                 config: Config,
                 device: torch.device):
        """Initializer.
        Args:
            model: whisper-guided VC models.
            dataset, testset: dataset.
            config: unified configurations.
            device: target computing device.
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device

        def identity(x): return x
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.train.batch,
            shuffle=config.train.shuffle,
            collate_fn=identity,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory)

        # training wrapper
        self.wrapper = TrainingWrapper(model, config, device)

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            config.train.learning_rate,
            (config.train.beta1, config.train.beta2))

        self.train_log = SummaryWriter(
            os.path.join(config.train.log, config.train.name, 'train'))
        self.test_log = SummaryWriter(
            os.path.join(config.train.log, config.train.name, 'test'))

        self.ckpt_path = os.path.join(
            config.train.ckpt, config.train.name, config.train.name)

        self.melspec = speechset.utils.MelSTFT(speechset.Config())
        self.cmap = np.array(plt.get_cmap('viridis').colors)

    def train(self, epoch: int = 0):
        """Train wavegrad.
        Args:
            epoch: starting step.
        """
        self.model.train()
        step = epoch * len(self.loader)
        for epoch in tqdm.trange(epoch, self.config.train.epoch):
            with tqdm.tqdm(total=len(self.loader), leave=False) as pbar:
                for it, bunch in enumerate(self.loader):
                    # [B], [B], [B, T]
                    sid, speeches, lengths = self.dataset.collate(bunch)
                    # [B]
                    sid = torch.tensor(sid, device=self.device)
                    # [B, seglen]
                    segment = self.wrapper.random_segment(speeches, lengths)
                    # compute loss
                    loss, losses, aux = self.wrapper.compute_loss(sid, segment)
                    # update
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    step += 1
                    pbar.update()
                    pbar.set_postfix({'loss': loss.item(), 'step': step})

                    for key, val in losses.items():
                        self.train_log.add_scalar(f'loss/{key}', val, step)

                    with torch.no_grad():
                        grad_norm = np.mean([
                            torch.norm(p.grad).item()
                            for p in self.model.parameters() if p.grad is not None])
                        param_norm = np.mean([
                            torch.norm(p).item()
                            for p in self.model.parameters() if p.dtype == torch.float32])

                    self.train_log.add_scalar('common/grad-norm', grad_norm, step)
                    self.train_log.add_scalar('common/param-norm', param_norm, step)
                    self.train_log.add_scalar(
                        'common/learning-rate', self.optim.param_groups[0]['lr'], step)

                    if it % (len(self.loader) // 50) == 0:
                        self.train_log.add_image(
                            # [3, M, T]
                            'train/gt', self.mel_img(segment[Trainer.LOG_IDX].cpu().numpy()), step)
                        self.train_log.add_image(
                            # [3, M, T]
                            'train/q(z_{t}|z_{0})', self.mel_img(aux['base'][Trainer.LOG_IDX]), step)
                        self.train_log.add_image(
                            # [3, M, T]
                            'train/p(z_{0}|z_{t})', self.mel_img(aux['denoised'][Trainer.LOG_IDX]), step)

            self.model.save(f'{self.ckpt_path}_{epoch}.ckpt', self.optim)

            losses = {key: [] for key in losses}
            with torch.no_grad():
                maxlen = int(Trainer.EVAL_MAX_SEC * self.config.model.sr)
                # clamp
                timesteps = min(lengths[Trainer.LOG_IDX].item(), maxlen)
                # [T], gt plot
                speech = speeches[Trainer.LOG_IDX, :timesteps]
                self.test_plot('test/gt', speech.cpu().numpy(), step)

                # inference
                self.model.eval()
                # [1, T x H]
                signal, [_, *ir] = self.model(
                    speech[None], sid[Trainer.LOG_IDX, None], use_tqdm=True)
                self.model.train()

                self.test_plot('test/synth', signal.squeeze(dim=0).cpu().numpy(), step)

                # intermediate representation
                intval = len(ir) // Trainer.EVAL_INTVAL
                for i, signal in enumerate(ir[::-1]):
                    if i % intval == 0:
                        self.test_plot(
                            f'gt-aux/p(z_{{{i}}}|z_{{{i + 1}}}))', signal.squeeze(0), step)

    def mel_img(self, signal: np.ndarray) -> np.ndarray:
        """Generate mel-spectrogram images.
        Args:
            signal: [float32; [T]], speech signal.
        Returns:
            [float32; [3, M, T // hop]], mel-spectrogram in viridis color map.
        """
        # [T, M]
        mel = self.melspec(signal)
        # minmax norm in range(0, 1)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-7)
        # in range(0, 255)
        mel = (mel * 255).astype(np.uint8)
        # [T, M, 3]
        mel = self.cmap[mel]
        # [3, M, T], make origin lower
        mel = np.flip(mel, axis=1).transpose(2, 1, 0)
        return mel

    def test_plot(self, name: str, signal: np.ndarray, step: int):
        """Plot signals.
        Args:
            name: plot name.
            signal: [np.float32; [T]], audio signal.
            step: current training steps.
        """
        self.test_log.add_image(
            name, self.mel_img(signal), step)
        self.test_log.add_audio(
            name, signal[None], step, sample_rate=self.config.model.sr)


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config', default=None)
    parser.add_argument('--load-epoch', default=0, type=int)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--auto-rename', default=False, action='store_true')
    args = parser.parse_args()

    # seed setting
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # configurations
    config = Config()
    if args.config is not None:
        print('[*] load config: ' + args.config)
        with open(args.config) as f:
            config = Config.load(json.load(f))

    if args.name is not None:
        config.train.name = args.name

    log_path = os.path.join(config.train.log, config.train.name)
    # auto renaming
    if args.auto_rename and os.path.exists(log_path):
        config.train.name = next(
            f'{config.train.name}_{i}' for i in range(1024)
            if not os.path.exists(f'{log_path}_{i}'))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ckpt_path = os.path.join(config.train.ckpt, config.train.name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    sr = config.model.sr
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    # prepare datasets
    readers = speechset.datasets.ConcatReader([
        speechset.datasets.VCTK('./datasets/VCTK-Corpus', sr),
        speechset.datasets.LibriTTS('./datasets/LibriTTS/train-clean-100', sr),
        speechset.datasets.LibriTTS('./datasets/LibriTTS/train-clean-360', sr)])
    config.model.num_spk = len(readers.speakers())
    print(f'[*] speakers: {config.model.num_spk}')

    trainset = speechset.utils.IDWrapper(RealtimeWavDataset(readers, device))

    # model definition
    model = WhisperGuidedVC(config.model)
    model.to(device)

    trainer = Trainer(model, trainset, config, device)

    # loading
    if args.load_epoch > 0:
        # find checkpoint
        ckpt_path = os.path.join(
            config.train.ckpt,
            config.train.name,
            f'{config.train.name}_{args.load_epoch}.ckpt')
        # load checkpoint
        ckpt = torch.load(ckpt_path)
        model.load_(ckpt, trainer.optim)
        print('[*] load checkpoint: ' + ckpt_path)
        # since epoch starts with 0
        args.load_epoch += 1

    # git configuration
    repo = git.Repo()
    config.train.hash = repo.head.object.hexsha
    with open(os.path.join(config.train.ckpt, config.train.name + '.json'), 'w') as f:
        json.dump(config.dump(), f)

    # start train
    trainer.train(args.load_epoch)
