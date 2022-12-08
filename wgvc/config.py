class Config:
    """Configuration for StyleDDPM-VC.
    """
    def __init__(self):
        # out-defined
        self.num_spk = None
        self.sr = 22050

        # diffusion steps
        self.steps = 64

        # schedules
        self.sched_start = 1e-4
        self.sched_end = 0.05

        # embedder
        self.pe = 128
        self.embeddings = 512
        self.mappings = 2

        # block
        self.channels = 64
        self.kernels = 3
        self.dilations = 2

        # wavenet
        self.cycles = 3
        self.layers = 10

        # whisper
        self.whisper_name = 'openai/whisper-base'
        # classifier-guidance, with norm-based scaling
        # , default 0.3 on guided-tts 2
        self.norm_scale = 0.3

        # upsampler
        self.upkernels = 5
        # prod to 160, STFT hop-length of `openai/whisper-base`
        self.upscales = [5, 4, 4, 2]
        self.leak = 0.2

        # speaker embedding
        self.spk = 512

        # classifier-free guidance, speaker
        # , default 1.0 on guided-tts 2
        # , default 0.3 on classifier-free guidance
        self.w = 0.3

        # prior temperature
        # , default 1.5 on guided-tts 2
        self.tau = 1.5
