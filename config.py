from speechset.config import Config as DataConfig
from wgvc.config import Config as ModelConfig


class TrainConfig:
    """Configuration for training loop.
    """
    def __init__(self, sr: int, hop: int):
        """Initializer.
        Args:
            sr: sample rate.
            hop: stft hop length.
        """
        # optimizer
        self.learning_rate = 1e-4
        self.beta1 = 0.5
        self.beta2 = 0.9

        # classifier-free guidance
        self.null_prob = 0.5

        # loader settings
        self.split = -100
        self.batch = 16
        self.shuffle = True
        self.num_workers = 4
        self.pin_memory = True

        # train iters
        self.epoch = 1000

        # segment length
        sec = 1.0
        self.seglen = int(sr * sec) // hop * hop

        # path config
        self.log = './log'
        self.ckpt = './ckpt'

        # model name
        self.name = 't1'

        # commit hash
        self.hash = 'unknown'


class Config:
    """Integrated configuration.
    """
    def __init__(self):
        self.data = DataConfig(batch=None)
        self.train = TrainConfig(self.data.sr, self.data.hop)
        self.model = ModelConfig()

    def validate(self):
        assert (
            self.data.sr == self.model.sr
            and self.data.win_fn == 'hann'), \
                'inconsistent data and model settings'

    def dump(self):
        """Dump configurations into serializable dictionary.
        """
        return {k: vars(v) for k, v in vars(self).items()}

    @staticmethod
    def load(dump_):
        """Load dumped configurations into new configuration.
        """
        conf = Config()
        for k, v in dump_.items():
            if hasattr(conf, k):
                obj = getattr(conf, k)
                load_state(obj, v)
        return conf


def load_state(obj, dump_):
    """Load dictionary items to attributes.
    """
    for k, v in dump_.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj
