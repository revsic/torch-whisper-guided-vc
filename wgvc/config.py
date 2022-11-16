class Config:
    """Configuration for StyleDDPM-VC.
    """
    def __init__(self):
        # out-defined
        self.mel = 80
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

        # unet
        self.channels = 128
        self.kernels = 3
        self.stages = 4
        self.blocks = 2

        # context
        self.context = 1024
        self.w2v_name = 'facebook/wav2vec2-large-xlsr-53'
        self.w2v_lin = 12

        # speaker embedding
        self.spk = 512

        # classifier-free guidance
        self.w = 0.3
