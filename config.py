from speechset.config import Config as DataConfig
from wgvc.config import Config as ModelConfig


class TrainConfig:
    """Configuration for training loop.
    """
    def __init__(self, sr: int):
        """Initializer.
        Args:
            sr: sample rate.
        """
        # optimizer
        self.learning_rate = 1e-4
        self.beta1 = 0.5
        self.beta2 = 0.9

        # augmentation
        self.smin = 68
        self.smax = 92
        self.std = 0.1

        # loader settings
        self.batch = 24
        self.shuffle = True
        self.num_workers = 4
        self.pin_memory = True

        # train iters
        self.epoch = 1000

        # classifier-free guidance
        null_prob = 0.5
        self.null_size = int(self.batch * null_prob)

        # segment length
        sec = 1.0
        self.seglen = int(sr * sec)

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
        self.model = ModelConfig()
        self.train = TrainConfig(self.model.sr)

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
