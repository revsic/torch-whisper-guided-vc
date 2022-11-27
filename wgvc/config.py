class Config:
    """Configuration for StyleDDPM-VC.
    """
    def __init__(self):
        # out-defined
        self.num_spk = None
        self.sr = 22050

        # diffusion steps
        self.steps = 512

        # schedules
        self.internals = 1024
        self.logit_max = 10
        self.logit_min = -10

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

        # speaker embedding
        self.spk = 512

        # classifier-free guidance
        self.w = 0.3
