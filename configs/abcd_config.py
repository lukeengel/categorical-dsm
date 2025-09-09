from configs.base_config import get_config as get_base_config
from configs.dataconfigs import get_config as get_data_config


def get_config():
    config = get_base_config()

    # training
    training = config.training
    training.batch_size = 512
    training.n_steps = 100000
    training.log_freq = 100
    training.eval_freq = 100

    # data
    config.data = get_data_config("abcd")

    # model
    model = config.model
    model.estimate_noise = True
    model.ndims = 512
    model.time_embedding_size = 256
    model.layers = 15
    model.dropout = 0.2
    model.act = "gelu"
    model.sigma_max= 9
    model.sigma_min = 0.1
    model.embedding_type = "fourier"
    #model.attention_heads = 8
    #model.hidden_size = 1024
    #model.attention_dim_head = model.hidden_size // model.attention_heads
    model.name = 'tab-resnet'

    # optimization
    optim = config.optim
    optim.lr = 0.0001
    optim.weight_decay = 1e-7
    optim.scheduler = "cosine"

    return config
