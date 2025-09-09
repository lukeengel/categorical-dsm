import ml_collections
import torch
import math
from configs.base_config import get_config as get_base_config
from configs.dataconfigs import get_config as get_data_config

def get_config():
    config = get_base_config()

    # Training configuration
    training = config.training
    training.batch_size = 512  # Default value; overridden by sweeps
    training.n_steps = 100000  # Default value
    training.log_freq = 50
    training.eval_freq = 100
    training.checkpoint_freq = 10000
    training.snapshot_freq = 10000

    # Data configuration
    config.data = get_data_config("abcd")

    # Model configuration
    model = config.model
    model.estimate_noise = True
    model.ndims = 582
    model.time_embedding_size = 256
    model.layers = 15
    model.dropout = 0.3
    model.act = "gelu"
    model.sigma_max= 9
    model.sigma_min = 0.1
    model.embedding_type = "positional"
    #model.attention_heads = 8
    #model.hidden_size = 1024
    #model.attention_dim_head = model.hidden_size // model.attention_heads
    model.name = 'tab-resnet'


    optim = config.optim
    optim.lr = 3e-4
    optim.weight_decay = 1e-5
    optim.scheduler = "cosine"

    # Configuration for Hyperparameter sweeps
    config.sweep = sweep = ml_collections.ConfigDict()

    # Define sweep parameters
    param_dict = dict(
         #Batch size (larger values)
        #training_batch_size={
         #   "distribution": "int_uniform",
          #  "min": 64,
           # "max": 2048,
        #},
       
        # Learning rate (Adam optimizer)
        #model_optim_lr={
         #  "distribution": "log_uniform_values",
          #  "min": 1e-5,
           # "max": 1e-1,
        #},

        # Weight decay
        #optim_weight_decay={
         #   "distribution": "log_uniform_values",
          #  "min": 1e-6,
           # "max": 1e-2,
        #},

        model_act={
            "values": [
                "gelu",
                "swish",
            ]
        },
	model_embedding_type={"values": ["fourier", "positional"]},
	model_dropout={"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},

        # Model dimensionality (larger range for ndims)
        model_ndims={
            "distribution": "int_uniform",
            "min": 128,
            "max": 2048,
        },

        # Number of layers
        model_layers={
            "distribution": "int_uniform",
            "min": 5,
            "max": 40,
        },
    )

    sweep.parameters = param_dict
    sweep.method = "random"  # Random search over hyperparameter space

    return config



