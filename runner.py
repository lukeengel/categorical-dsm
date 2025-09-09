import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from torchinfo import summary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


import wandb
from dataloader import get_dataset
from models.score_base import TabScoreModel #, VisionScoreModel
from ood_detection_helper import auxiliary_model_analysis, ood_metrics
from sklearn.model_selection import KFold

mpl.rc("figure", figsize=(10, 4), dpi=100)
sns.set_theme()

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_loss_history = []  # Store validation loss history

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the current validation loss
        current_val_loss = trainer.callback_metrics[self.monitor].item()
        self.val_loss_history.append(current_val_loss)

        print(f"Current Val Loss: {current_val_loss:.4f}")

        # Call the original EarlyStopping logic
        super().on_validation_epoch_end(trainer, pl_module)


def train(config, workdir, sweep_id=None):

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Select model class based on config
    model = TabScoreModel(config) if "tab" in config.model.name else VisionScoreModel(config)


    train_loader, val_loader, test_loader, train_subject_ids, val_subject_ids, test_subject_ids = get_dataset(config)

    #TODO: Add hyperprameters to workdir name


    # Checkpoint that saves periodically to allow for resuming later
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{workdir}/checkpoints-meta/",
        save_last=True,  # Saves a copy as `last.ckpt`
        every_n_train_steps=config.training.checkpoint_freq,
    )

    snapshot_callback = ModelCheckpoint(
        dirpath=f"{workdir}/checkpoints/",
        monitor="val_loss",
        filename=(f"{sweep_id}-" if sweep_id else "") + "{step}-{val_loss:.4f}",
        save_top_k=5,
        save_last=False,
        every_n_train_steps=config.training.snapshot_freq,
    )

    # Early stopping callback 
    early_stopping_callback = CustomEarlyStopping(
    #print(f"Current Val Loss: {val_loss:.4f}"),
    monitor='val_loss',  # Metric to monitor
    patience=10, # Number of epochs with no improvement after which training will be stopped 
    min_delta = 5,
    verbose=True,  # Print messages when early stopping is triggered 
    mode='min', ) # Minimize the monitored metric 


    callback_list = [checkpoint_callback, snapshot_callback, early_stopping_callback]

    if "tab" in config.model.name:
        logging.info(ModelSummary(model, max_depth=3))
    else:
        summary(
            model,
            depth=3,
            input_data=[
                torch.zeros(
                    1,
                    config.data.categorical_channels + config.data.continuous_channels,
                    config.data.image_size,
                    config.data.image_size,
                ),
                torch.zeros(
                    1,
                ),
            ],
        )
    
    wandb_logger = WandbLogger(log_model=False, save_dir="wandb")
    wandb_logger.watch(model, log_freq=config.training.snapshot_freq, log="all")
    tb_logger = TensorBoardLogger(
        save_dir=f"{workdir}/tensorboard_logs/", name="", default_hp_metric=False
    )

    ckpt_path = None
    if config.training.resume:
        ckpt_path = f"{workdir}/checkpoints-meta/last.ckpt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError
    
    trainer = pl.Trainer(
        # precision=16,
        accelerator=str(config.device),
        default_root_dir=workdir,
        # max_epochs=config.training.n_epochs,
        max_steps=config.training.n_steps,
        gradient_clip_val=config.optim.grad_clip,
        val_check_interval=config.training.eval_freq,
        log_every_n_steps=config.training.log_freq,
        callbacks=callback_list,
        fast_dev_run=5 if config.devtest else 0,
        enable_model_summary=False,
        check_val_every_n_epoch=None,
        logger=[tb_logger, wandb_logger],
        # num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    eval(config, workdir, sweep_id, ckpt_num=-1)

class ConfigObject:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, ConfigObject(value) if isinstance(value, dict) else value)

    def to_dict(self):
        return {
            key: value.to_dict() if isinstance(value, ConfigObject) else value
            for key, value in self.__dict__.items()
        }


def eval(config, workdir, sweep_id=None, ckpt_num=-1):

    assert config.msma.checkpoint in ["best", "last"]
    denoise = config.msma.denoise
    ckpt_dir = "checkpoints" if config.msma.checkpoint == "best" else "checkpoints-meta"
    ckpt_dir = os.path.join(workdir, ckpt_dir)

    if sweep_id:
        ckpt_dir = os.path.join(ckpt_dir, sweep_id)

    ckpts = sorted(os.listdir(ckpt_dir))
    ckpt = ckpts[ckpt_num]
    step = ckpt.split("-")[0]
    fname = os.path.join(
        workdir, "score_norms", f"{step}-{'denoise' if denoise else ''}-score_norms.npz"
    )

    print(
        f"Evaluating {ckpt} with denoise = {denoise} and saving to {fname} if not already present."
    )

    if os.path.exists(fname):
        print(f"Loading from {fname}")
        with np.load(fname, allow_pickle=True) as npzfile:
            outdict = {k: npzfile[k].item() for k in npzfile.files}
    else:
        checkpoint_path = os.path.join(ckpt_dir, ckpt) 
        print(f"Using checkpoint: {checkpoint_path}") # Extract config directly from the checkpoint 
        checkpoint = torch.load(checkpoint_path, map_location='cpu') 
        if 'hyper_parameters' in checkpoint: 
            config_dict = checkpoint['hyper_parameters'] 
            print(f"Extracted config from checkpoint: {config}")
            config = ConfigObject(config_dict) 
        else: 
            print("Warning: No config found in checkpoint. Using provided config.")
        scorenet = TabScoreModel.load_from_checkpoint(
            checkpoint_path=os.path.join(ckpt_dir, ckpt), config=config
        ).cuda()
        scorenet.eval().requires_grad_(False)

        train_loader, val_loader, test_loader, train_subject_ids, val_subject_ids, test_subject_ids = get_dataset(config, train_mode=False)
        # (Optional) Print the head of test subject IDs to verify mapping.
        print("Test Subject IDs (head):", test_subject_ids[:5])

        outdict = {}
        with torch.cuda.device(0):
            for ds, loader in [
                ("train", train_loader),
                ("val", val_loader),
                ("test", test_loader),
            ]:
                score_norms = []
                labels = []
                for x_batch, y in loader:
                    s = (
                        scorenet.scorer(x_batch.cuda(), denoise_step=denoise)
                        .cpu()
                        .numpy()
                    )
                    score_norms.append(s)
                    labels.append(y.numpy())
                score_norms = np.concatenate(score_norms)
                labels = np.concatenate(labels)
                outdict[ds] = {"score_norms": score_norms, "labels": labels}

        # Attach the test subject IDs into the output dictionary.
        outdict["train"]["subject_ids"] = np.array(train_subject_ids)
        outdict["val"]["subject_ids"] = np.array(val_subject_ids)
        outdict["test"]["subject_ids"] = np.array(test_subject_ids)
        os.makedirs(os.path.join(workdir, "score_norms"), exist_ok=True)
        fname = os.path.join(
            workdir,
            "score_norms",
            f"{step}-{'denoise' if denoise else ''}-score_norms.npz",
        )

        with open(fname, "wb") as f:
            np.savez_compressed(f, **outdict)

    X_train = outdict["train"]["score_norms"]
    np.random.seed(42)
    np.random.shuffle(X_train)
    X_val = outdict["val"]["score_norms"]
    X_train = np.concatenate((X_train[: len(X_val)], X_val))
    test_labels = outdict["test"]["labels"]
    X_test = outdict["test"]["score_norms"][test_labels == 0]
    X_ano = outdict["test"]["score_norms"][test_labels == 1]
    results = auxiliary_model_analysis(
        X_train,
        X_test,
        [X_ano],
        components_range=range(5, 6, 1),
        labels=["Train", "Inlier", "Outlier"],
    )
    ood_metrics(
        -results["GMM"]["test_scores"],
        -results["GMM"]["ood_scores"][0],
        plot=True,
        verbose=True,
    )
    plt.suptitle(f"{config.data.dataset} - GMM", fontsize=18)
    plt.savefig(fname.replace("score_norms.npz", "gmm.png"), dpi=100)

    ood_metrics(
        results["KD"]["test_scores"],
        results["KD"]["ood_scores"][0],
        plot=True,
        verbose=True,
    )
    plt.suptitle(f"{config.data.dataset} - KD Tree", fontsize=18)
    plt.savefig(fname.replace("score_norms.npz", "kd.png"), dpi=100)

    logging.info(results["GMM"]["metrics"])


# Match outlier subject IDs to GMM OOD scores
    outlier_subject_ids = outdict["test"]["subject_ids"][test_labels == 1]
    gmm_ood_scores = results["GMM"]["ood_scores"][0]
    kd_ood_scores = results["KD"]["ood_scores"][0]

# Sanity check
    assert len(outlier_subject_ids) == len(gmm_ood_scores), "Mismatch between subjects and GMM scores!"

    def save_scores_to_csv(subject_ids, scores, output_path, score_label):
        df = pd.DataFrame({
            "subject_id": subject_ids,
            score_label: scores
        })
        df.to_csv(output_path, index=False)
        print(f"[INFO] Saved {score_label} to {output_path}")

    # Save into the outdict
    outdict["test"]["gmm_ood_scores"] = gmm_ood_scores
    outdict["test"]["kd_ood_scores"] = kd_ood_scores
    outdict["test"]["ood_subject_ids"] = outlier_subject_ids

    gmm_csv_path = fname.replace("score_norms.npz", "gmm_scores.csv")
    kd_csv_path = fname.replace("score_norms.npz", "kd_scores.csv")
    # Then call:
    save_scores_to_csv(outdict["test"]["ood_subject_ids"], outdict["test"]["gmm_ood_scores"], gmm_csv_path, "gmm_ood_score")
    save_scores_to_csv(outdict["test"]["ood_subject_ids"], outdict["test"]["kd_ood_scores"], kd_csv_path, "kd_ood_score")




