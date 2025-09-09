import logging

import pdb
import os
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, MinMaxScaler
from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    Subset,
    TensorDataset,
    random_split,
)
from torchvision.datasets import MNIST, FashionMNIST, Omniglot
from torchvision.transforms import Compose, InterpolationMode, Lambda, Resize, ToTensor
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)
import torchvision.transforms as T
from models.segmentation.presets import SegmentationTrain, SegmentationEval
from configs.dataconfigs import get_config
from models.mutils import onehot_to_logit
from PIL import Image
from tqdm.auto import tqdm

import models.segmentation.module_transforms as SegT

tabular_datasets = {
    # "bank": "bank-additional-ful-nominal.arff",
    # "chess": "chess_krkopt_zerovsall.arff",
    # "census": "census.pkl",
    # "probe": "kddcup99-corrected-probevsnormal-nominal-cleaned.arff",
    # "u2r": "kddcup99-corrected-u2rvsnormal-nominal-cleaned.arff",
    # "solar": "solar-flare_FvsAll-cleaned.arff",
    # "cmc": "cmc-nominal.arff",
    # "celeba": "list_attr_celeba_baldvsnonbald.arff",
    # "cars": "car_evaluation.csv",
    # "mushrooms": "mushrooms.csv",
    # "nursery": "nursery.csv",
    "abcd": "all_squeeky_id.csv",

}

def get_dataset(config, train_mode=True, return_with_loader=True, return_logits=True):
    generator = torch.Generator().manual_seed(config.seed)
    dataset_name = config.data.dataset.lower()
    rootdir = "/proj/NIRAL/studies/ABCD/PsychosisAtypicality/data/"
 
    if dataset_name in tabular_datasets:
        data, subject_ids = build_tabular_ds(dataset_name, return_logits=return_logits)

  

    # Subset inlier only
    logging.info(f"Splitting dataset with seed: {config.seed}")

    # Subset inlier only
    # Split 80,10,10 train, val, test
    # Combine test and outlier
    if dataset_name in tabular_datasets:
        inliers = data.tensors[1] == 0
        inlier_idxs = torch.argwhere(inliers).squeeze()
        outlier_idxs = torch.argwhere(~inliers).squeeze()
        logging.info(f"# Outliers: {len(outlier_idxs)}")
        inlier_ds = Subset(data, inlier_idxs)
        outlier_ds = Subset(data, outlier_idxs)
        # pdb.set_trace()
        train_ds, val_ds, test_ds = random_split(
            inlier_ds, [0.7, 0.2, 0.1], generator=generator
        )
        test_ds = ConcatDataset([test_ds, outlier_ds])
    else:
        train_ds, val_ds = random_split(data, [0.9, 0.1], generator=generator)
        test_ds = val_ds  # WONT BE USED
    #print(f'train_ds: {train_ds}')
    #train_min = train_ds.min(axis=1)
    #train_max = train_ds.max(axis=1)
    #train_ds -= train_min
    #val_ds -= train_min
    #test_ds -= train_min
    #train_ds /= train_max - train_min
    #val_ds /= train_max - train_min
    #test_ds /= train_max - train_min
    logging.info(f"Train, Val, Test: {len(train_ds)}, {len(val_ds)}, {len(test_ds)}")

    # A helper to extract the subject IDs corresponding to a dataset subset.
    def get_subset_subject_ids(subset, subject_ids):
        if subject_ids is None:
            return None
        # If the subset is a Subset, use its .indices attribute to map back to the original order.
        return [subject_ids[i] for i in subset.indices] if hasattr(subset, 'indices') else subject_ids

    def get_concat_dataset_indices(concat_dataset):
        all_indices = []
        for ds in concat_dataset.datasets:
            if hasattr(ds, 'indices'):
                all_indices.extend(ds.indices)
            else:
                # If ds is not a Subset, use range as fallback.
                all_indices.extend(list(range(len(ds))))
        return np.array(all_indices)


    train_subject_ids = get_subset_subject_ids(train_ds, subject_ids)
    val_subject_ids = get_subset_subject_ids(val_ds, subject_ids)
    test_indices = get_concat_dataset_indices(test_ds)
    test_subject_ids = np.array(subject_ids)[test_indices]
    
    # You can check the first sample:
    sample_features, sample_label = train_ds.dataset[0]
    print("Sample features:", sample_features)
    print("Sample label:", sample_label)

    # And then check the corresponding subject id:
    print("Subject ID for this sample:", train_subject_ids[:0])
    #print('Val indices:', val_ds.indices)
    # Print a sample mapping for verification.
    print("Train Subject IDs (head):", train_subject_ids[:5])
    print("Val Subject IDs (head):", val_subject_ids[:5])
    print("Test Subject IDs (head):", test_subject_ids[:5])

    # if train_mode and dataset_name in tabular_datasets:
    #     inlier_idxs = [idx for idx, (x, y) in enumerate(train_ds) if y == 0]
    #     train_ds = Subset(train_ds, inlier_idxs)
    print(config.training.batch_size)
    print(config.eval.batch_size)
    if return_with_loader:
    # Wrap DataLoader iterations with tqdm to show progress
        print("Loading Train Data...")
        train_ds = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
        shuffle=train_mode,
    )

        print("Loading Validation Data...")
        val_ds = DataLoader(
        val_ds,
        batch_size=config.eval.batch_size,
        num_workers=2,
        pin_memory=True,
    )

        print("Loading Test Data...")
        test_ds = DataLoader(
        test_ds,
        batch_size=config.eval.batch_size,
        num_workers=2,
        pin_memory=True,
    )

    return train_ds, val_ds, test_ds, train_subject_ids, val_subject_ids, test_subject_ids



def load_dataset(name):
    str_type = lambda x: str(x, "utf-8")

    # if name in ["adult"]:
    #     return pd.read_csv(f"data/{name}.csv").dropna()

    # AD_nominal
    # dtype = all categorical
    # Anomaly: AD

    # AID
    # dtype = all categorical
    # Anomaly: active
    basedir = "/proj/NIRAL/studies/ABCD/PsychosisAtypicality/data/"
    dataconfig = get_config(name)
    label_name = dataconfig.label_column

    if name == "abcd":
        df = pd.read_csv(f"{basedir}all_squeeky_id.csv", index_col=0)
        # Extract subject IDs from the src_subject_id column
        subject_ids = df['src_subject_id'].copy().values
        na_count = df.isna().sum().sum()
        print(f'na count {na_count}') 
        df.dropna(axis=1, how='all', inplace=True)
        print(df.head())
        print(df.shape)
        #print(df.tail())
        #df = pd.read_csv(f"{basedir}merged_data.csv").dropna()
        print(f'df: {df}')
        columns_to_change = ['demo_sex_v2','race_ethnicity']
#    columns_to_change = ['famhx_ss_fath_prob_vs_p', 'famhx_ss_moth_prob_vs_p', 'famhx_ss_fath_prob_nrv_p', 'famhx_ss_moth_prob_nrv_p','ksads_gad_raw_271_p', 'ksads_gad_raw_273_p', 
    #                 'ksads_pd_raw_176_p', #'ksads_pd_raw_178_p','ksads_sad_raw_209_p', 'ksads_sad_raw_211_p', 'ksads_gad_raw_271_t', 'ksads_gad_raw_273_t', 'ksads_sad_raw_209_t', 'ksads_sad_raw_211_t', 'group_last_final' ]

        df[columns_to_change] = df[columns_to_change].astype(str)
        print(df.dtypes)
        unique_sex = df['demo_sex_v2'].unique()
        print('demo_sex_v2', unique_sex)
        unique_race = df['race_ethnicity'].unique()
        df = df.drop(columns=['src_subject_id'])

        #df[columns_to_change] = df[columns_to_change].astype(str) + "years"
        #df = df.drop(columns=columns_to_change)

    # if name == "census":
    #     df = pd.read_pickle(basedir + tabular_datasets[name])
    # elif name in ["cars", "mushrooms", "nursery"]:
    #     df = pd.read_csv(f"data/{tabular_datasets[name]}")
        
    #     if name == "cars":
    #         labels = df[label_name]
    #         drop_mask = np.logical_or(labels == "acc", labels == "good")
    #         labels = labels[~drop_mask]
    #         df = df[~drop_mask]

    #         # df[label_name][labels == "unacc"] = "0"
    #         # df[label_name][labels == "vgood"] = "1"
        
    #     if name == "nursery":
    #         labels = df[label_name]
    #         drop_mask = np.logical_or(labels == "not_recom", labels == "very_recom")
    #         labels = labels[drop_mask]
    #         df = df[drop_mask]        print('race_ethnicity:', unique_sex)


    #         # df[label_name][labels == "not_recom"] = "0"
    #         # df[label_name][labels == "very_recom"] = "1"

    else:
        data, metadata = arff.loadarff(basedir + tabular_datasets[name])
        df = pd.DataFrame(data).applymap(str_type)

    X = df.drop(
        columns=label_name,
    )
    y = np.zeros(len(df[label_name]), dtype=np.float32)
    ano_idxs = df[label_name] == dataconfig.anomaly_label
    y[ano_idxs] = 1.0
    print(y)
    return X, y.squeeze(), subject_ids, dataconfig


def build_tabular_ds(name, return_logits=True):
    X, y, subject_ids, dataconfig = load_dataset(name)
    to_logit = lambda x: np.log(np.clip(x, a_min=1e-5, a_max=1.0))
    # to_logit = lambda x: np.log(np.clip(x*1e5, a_min=1e-5, a_max=1e5))


    categorical_columns_selector = selector(dtype_include=object)
    continuous_columns_selector = selector(dtype_include=[int, float])
    categorical_features = categorical_columns_selector(X)
    continuous_features = continuous_columns_selector(X)
    

    cat_processor = [OneHotEncoder(sparse_output=False)]
    if return_logits:
        cat_processor.append(FunctionTransformer(to_logit))
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), continuous_features),
            (
              "cat",
              make_pipeline(*cat_processor),
                categorical_features,
            ),
        ]
    )

    if name in ["probe", "mushrooms", "nursery"]:
        # Some categories only appear in outliers ...
        # so preprocessor needs to know them
        preprocessor.fit(X)
    else:
        # Only fit on inliers
        

        preprocessor.fit(X[y == 0])

    categories = [
        len(x)
        for x in preprocessor.named_transformers_["cat"]
        .named_steps["onehotencoder"]
        .categories_
    ]
    print(dataconfig.categories)
    print(categories)
    print(len(continuous_features))
    assert categories == dataconfig.categories
    assert len(continuous_features) == dataconfig.numerical_features
    # pdb.set_trace()
    X = preprocessor.transform(X)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    logging.info(f"Loaded dataset: {name}, Shape: {X.shape}")

    return TensorDataset(X, y), subject_ids


class CachedVOCSegmentation(torch.utils.data.Dataset):
    def __init__(self, root, image_set="train", download=False, transforms=None):
        self.rootdir = root
        self.image_set = image_set
        self.transforms = transforms

        self.cache = []
        self.voc = VOCSegmentation(
            root=root,
            download=download,
            image_set=image_set,
            transforms=None,
        )

        logging.info(f"Loading images from {image_set} set")
        for idx in tqdm(range(len(self.voc))):
            img, target = self.voc[idx]
            # img = Image.open(self.voc.images[idx]).convert("RGB")
            # target = Image.open(self.voc.masks[idx])
            # print(img.size, target.size)

            img = T.functional.pil_to_tensor(img)
            img = T.functional.convert_image_dtype(img, dtype=torch.float32)
            target = torch.as_tensor(np.array(target)[None, ...], dtype=torch.int64)
            # print(img.shape, target.shape)
            # break

            self.cache.append((img, target))

        logging.info(f"Loaded {len(self.cache)} images")

    def __getitem__(self, idx):
        return self.transforms(*self.cache[idx])

    def __len__(self):
        return len(self.cache)


class MultiSequential(torch.nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        for module in self:
            x = module(x)
        return x


class TrainTransform(torch.nn.Module):
    def __init__(
        self,
        *,
        out_size,
        base_size,
        crop_size,
        hflip_prob=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        to_logits=False,
    ):
        super().__init__()

        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)
        self.out_sz = out_size

        trans = [SegT.RandomResize(min_size, max_size)]
        # trans = []
        if hflip_prob > 0:
            trans.append(SegT.RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                SegT.RandomCrop(crop_size),
                SegT.RandomResize(self.out_sz, self.out_sz),
                SegT.Normalize(mean=mean, std=std),
            ]
        )
        logging.info("Compiling transforms...")
        self.transforms = MultiSequential(*trans)
        self.transforms = torch.jit.script(self.transforms)
        logging.info("Completed.")

        # self.to_onehot = partial(F.one_hot, num_classes=21)
        def build_one_hot_transform(to_logits=to_logits):
            if to_logits:

                @torch.jit.script
                def to_onehot(target):
                    target[target == 255] = 0
                    target = F.one_hot(target, num_classes=21).squeeze().float()
                    target = target.permute(2, 0, 1)
                    target = torch.log(torch.clamp(target, min=1e-5, max=1.0))
                    return target

            else:

                @torch.jit.script
                def to_onehot(target):
                    target[target == 255] = 0
                    target = F.one_hot(target, num_classes=21).squeeze().float()
                    target = target.permute(2, 0, 1)
                    return target

            return to_onehot

        self.to_onehot = build_one_hot_transform()

    def __call__(self, img, target):
        img, target = self.transforms((img, target))
        target = self.to_onehot(target)
        img = torch.cat((img, target), dim=0)
        return img, 0
