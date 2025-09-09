from configs.base_config import get_config as get_base_config
from configs.dataconfigs import get_config as get_data_config


def get_config():
    config = get_base_config()

    # training
    training = config.training
    training.batch_size = 32
    training.n_steps = 100000
    training.log_freq = 50
    training.eval_freq = 100

    # data
    config.data = get_data_config("abcd")

    # model
    model = config.model
    model.estimate_noise = True
    model.ndims = 32
    model.time_embedding_size = 32
    model.layers = 5
    model.dropout = 0.0
    model.act = "gelu"
    model.sigma_max= 904
    model.attention_heads = 4
    model.hidden_size = 32
    model.attention_dim_head = model.hidden_size // model.attention_heads
    model.name = 'tab-transformer'

    # optimization
    optim = config.optim
    optim.lr = 1e-3
    optim.weight_decay = 0.0
    optim.scheduler = "none"

    return config
