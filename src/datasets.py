"""Dataset related functions and classes"""

import os

import h5py
import numpy as np
import pandas as pd
import torch
from open_clip import get_tokenizer  # works on open-clip-torch>=2.23.0, timm>=0.9.8
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import CenterCrop, Compose, Normalize, ToTensor
from tqdm import tqdm
from transformers import BertTokenizer

# from configs.data_config import DataAugmentationConfig
from src import constants
from src.clip.clip import tokenize

# from src.transformations.cell import CellAugmentation
from src.transformations.cloome import _transform


class CellPainting(Dataset):
    """Customized dataset for cell painting images."""

    def __init__(
        self,
        sample_index_file: str,
        mole_struc: str = "morgan",
        context_length: int = 77,
        image_directory_path: str = os.path.join(constants.DATASET_DIR, "image_dir"),
        molecule_path: str = os.path.join(
            constants.DATASET_DIR,
            "caption_dir",
        ),
        transforms=None,
        group_views: bool = False,
        subset: float = 1.0,
        num_channels: int = 5,
        unique: bool = False,
        dataset: str = "bray2017",
    ):
        """Read samples from cellpainting dataset."""

        # Check if the path is a folder or a file (HDF5)

        if os.path.isdir(image_directory_path):
            self.is_hdf5 = False
            assert os.path.exists(
                image_directory_path
            ), f"Image directory {image_directory_path} does not exist."
        elif os.path.isfile(image_directory_path) and image_directory_path.endswith(".h5"):
            self.is_hdf5 = True
            self.h5_path = image_directory_path
            self.img_file = h5py.File(image_directory_path, "r")

            # with h5py.File(image_directory_path, "r") as f:
            try:
                self.img_ids = [name.decode("utf-8") for name in self.img_file["names"][:]]
            except KeyError:
                self.img_ids = [
                    name.decode("utf-8").replace(".npz", "")
                    for name in self.img_file["well_id"][:]
                ]

        else:
            raise ValueError(
                "image_directory_path must be either a valid directory or HDF5 file."
            )
        # Load sample index

        if type(sample_index_file) == list:
            dfs = []
            for file in sample_index_file:
                assert os.path.exists(file), f"Image path {file} does not exist."

                df = pd.read_csv(file, sep=",", header=0)
                dfs.append(df)

            sample_index = pd.concat(dfs, ignore_index=False)
        else:
            assert os.path.isfile(
                sample_index_file
            ), f"Sample index {sample_index_file} does not exist."
            sample_index = pd.read_csv(sample_index_file, sep=",", header=0)

        if unique:  # whether to return only unique treatment.
            sample_index["SAMPLE_KEY"] = (
                sample_index["SAMPLE_KEY"].str.rsplit("-", n=1).str[0]
            )
            sample_index = sample_index.drop_duplicates(subset="SAMPLE_KEY")
            sample_index = sample_index.reset_index(drop=True)

        sample_index.set_index(["SAMPLE_KEY"])
        sample_keys = sample_index["SAMPLE_KEY"].tolist()

        self.image_directory_path = image_directory_path
        self.sample_index = sample_index
        self.group_views = group_views
        self.transforms = transforms
        self.num_channels = num_channels
        self.context_length = context_length
        self.unique = unique

        # Load molecule file
        assert os.path.isfile(
            molecule_path
        ), f"Molecule file {molecule_path} does not exist."

        if mole_struc == "morgan":
            molecule_df = pd.read_hdf(molecule_path, key="df")
        else:
            molecule_df = pd.read_csv(molecule_path, index_col=["ID"])

        if unique:  # whether to return only unique treatment.
            if dataset == "rxrx3-core":
                molecule_df["new_index"] = molecule_df.index
            else:
                molecule_df["new_index"] = molecule_df.index.str.rsplit("-", n=1).str[0]

            molecule_df.set_index("new_index", inplace=True)
            molecule_df = molecule_df[~molecule_df.index.duplicated(keep="first")]

        if self.context_length == 256:
            self.tokenizer = get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
        elif self.context_length == 512:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        # else:
        #     raise ValueError(f"Context length {self.context_length} does not exist.")

        self.molecule_df = molecule_df
        self.mole_struc = mole_struc
        self.unique_molecule = np.unique(molecule_df.values)
        self.label_map = {
            molecule: idx for idx, molecule in enumerate(self.unique_molecule)
        }
        mol_keys = list(molecule_df.index.values)
        keys = list(set(sample_keys) & set(mol_keys))

        if len(keys) == 0:
            raise Exception("Empty dataset!")

        if subset != 1.0:
            keys = keys[: int(len(keys) * subset)]

        self.keys = keys
        self.n_samples = len(keys)
        self.sample_keys = list(keys)

    def __len__(self):
        """Return lenght of sample keys(PLATE-WELL_POSITION_ID)"""
        return len(self.keys)

    def __getitem__(self, idx):
        """Return mole and image pairs"""

        sample_key = self.keys[idx]
        mol_fp = self.molecule_df.loc[sample_key].values
        if self.mole_struc == "text":
            if self.context_length == 256:
                mol = self.tokenizer(mol_fp, context_length=self.context_length)
                mol = torch.squeeze(mol)
            elif self.context_length == 512:
                output = self.tokenizer(
                    mol_fp[0],
                    padding="max_length",
                    max_length=self.context_length,
                    return_tensors="pt",
                )
                mol = {
                    "input_ids": output["input_ids"].squeeze(0),
                    "attention_mask": output["attention_mask"].squeeze(0),
                }
            else:
                mol = tokenize(mol_fp, self.context_length, truncate=True).flatten()
        elif self.mole_struc == "plate":
            # obtain plate id
            match = str(sample_key).split("-")
            mol = match[0] + "-" + match[1]
        elif self.mole_struc == "embedding":
            mol = np.array(eval(self.molecule_df.loc[sample_key].embedding))
            mol = mol.astype(np.float32)
        elif self.mole_struc == "label":
            mol = mol_fp[0]
        else:
            mol = torch.from_numpy(mol_fp)

        img = self.get_image(sample_key)["input"]

        return (
            (
                img,
                {
                    "channels": np.asarray([c for c in range(img.shape[0])]),
                },
            ),
            mol,
        )

    def get_image(self, key):
        """Load cell painting images wrt key"""

        if self.group_views:
            X = self.load_view_group(key)
        else:
            if self.is_hdf5:
                if key in self.img_ids:
                    index = self.img_ids.index(key)
                    # img_file = self._load_hdf5()
                    X = self.img_file["embeddings"][index]
                else:
                    print(f"ERROR: Missing sample '{key}' in HDF5 file")
                    return dict(input=np.nan, ID=key)
            else:
                filepath = os.path.join(self.image_directory_path, "{}.npz".format(key))
                if os.path.exists(filepath):
                    X = self.load_view(filepath=filepath)
                    if len(X.shape) == 3:
                        X = np.squeeze(X)
                    index = int(np.where(self.sample_index["SAMPLE_KEY"] == key)[0])
                else:
                    print("ERROR: Missing sample '{}'".format(key))
                    return dict(input=np.nan, ID=key)

        if self.transforms:
            X = self.transforms(X)

        return dict(input=X, row_id=index, ID=key)

    def load_view(self, filepath):
        """Load all channels for one sample"""
        npz = np.load(filepath, allow_pickle=True)

        if "sample" in npz:
            image = npz["sample"].astype(np.float32)
            return image

        return None

    def load_view_group(self, groupkey):
        result = np.empty((1040, 2088 - 12, 5), dtype=np.uint8)
        viewgroup = self.sample_index.get_group(groupkey)
        for i, view in enumerate(viewgroup.sort_values("SITE", ascending=True).iterrows()):
            corner = (0 if int(i / 3) == 0 else 520, i % 3 * 692)
            filepath = os.path.join(
                self.data_directory, "{}.npz".format(view[1].SAMPLE_KEY)
            )
            v = self.load_view(filepath=filepath)[:, 4:, :]
            result[corner[0] : corner[0] + 520, corner[1] : corner[1] + 692, :] = v
        return result

    def _load_hdf5(self):
        if self.img_file is None:
            self.img_file = h5py.File(self.h5_path, "r", swmr=True)
        return self.img_file

    def get_sample_keys(self):
        return self.sample_keys.copy()


def get_cellpainting_dataset(args, num_processes, is_train=True, subset=1.0):
    """Helpler function to get cell painting dataloader"""

    if args.model_type == "classifier":
        mole_struc = "plate"
    elif args.model_type == "pubmed_emb_clip":
        mole_struc = "embedding"
    elif args.model_type in ["vit", "densenet"]:
        mole_struc = "label"
    elif args.model_type in ["cloome", "cloome_phenom1", "molphenix", "cloome_mpnn"]:
        mole_struc = "morgan"
    else:
        mole_struc = "text"

    if args.dataset == "bray2017":
        split_name = (
            f"datasplit{args.split}-train" if is_train else f"datasplit{args.split}-val"
        )
        sample_index_file = os.path.join(
            args.outdir, args.split_label_dir, f"{split_name}.csv"
        )
        preprocess_fn = _transform(
            args.image_resolution_train,
            args.image_resolution_val,
            is_train,
            "dataset",
            "crop",
        )
    elif args.dataset == "jumpcp":
        split_name = "jumpcp_training_label2" if is_train else "jumpcp_testing_label2"
        sample_index_file = os.path.join(f"path_to_jumpcp/{split_name}.csv")

        if args.model_type == "cell_clip":
            preprocess_fn = None
        else:
            preprocess_fn = _transform(
                args.image_resolution_train,
                args.image_resolution_val,
                is_train,
                "dataset",
                "crop",
            )
    elif args.dataset == "rxrx3-core":
        sample_index_file = os.path.join(
            constants.OUT_DIR, "path_to_rxrx3/rxrx3-core_label.csv"
        )
        crop_res = args.image_resolution_train if is_train else args.image_resolution_val
        preprocess_fn = Compose(
            [
                CenterCrop(crop_res),
                ToTensor(),
                Normalize(
                    (47.1314, 40.8138, 53.7692, 46.2656, 28.7243),
                    (24.1384, 23.6309, 28.1681, 23.4018, 28.7255),
                ),
            ]
        )

    img_dir = os.path.join(constants.DATASET_DIR, args.dataset, args.img_dir)

    if args.model_type == "long_clip":
        dataset = CellPainting(
            sample_index_file,
            mole_struc,
            context_length=248,
            transforms=preprocess_fn,
            subset=subset,
            dataset=args.dataset,
        )
    elif args.model_type in [
        "bert_clip",
        "pubmed_clip",
        "cloome",
        "clip_channelvit",
        "cell_clip_mae",
        "cloome_mpnn",
    ]:
        if args.model_type in [
            "bert_clip",
            "mil_cell_clip",
            "clip_channelvit",
            "cell_clip_mae",
        ]:
            context_length = 512
        else:
            context_length = 256

        dataset = CellPainting(
            sample_index_file,
            mole_struc,
            context_length=context_length,
            transforms=preprocess_fn,
            subset=subset,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            unique=args.unique,
            dataset=args.dataset,
        )
    elif args.model_type in [
        "mil_cell_clip",
        "cell_clip",
        "cell_sigclip",
        "pubmed_clip_phenom1",
    ]:
        if args.model_type in ["mil_cell_clip", "cell_clip"]:
            context_length = 512
        else:
            context_length = 256
        dataset = CellPainting(
            sample_index_file,
            mole_struc,
            context_length=context_length,
            subset=subset,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            unique=args.unique,
            dataset=args.dataset,
        )
    elif args.model_type in ["cloome_phenom1", "molphenix"]:
        dataset = CellPainting(
            sample_index_file,
            mole_struc,
            subset=subset,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            unique=args.unique,
            dataset=args.dataset,
        )
    elif args.model_type in ["mae", "vit"]:
        preprocess_fn = _transform(
            args.image_resolution_train, args.image_resolution_val, is_train, None, "crop"
        )

        dataset = CellPainting(
            sample_index_file,
            mole_struc,
            transforms=preprocess_fn,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            subset=subset,
            unique=args.unique,
            dataset=args.dataset,
        )
    else:
        dataset = CellPainting(
            sample_index_file,
            mole_struc,
            transforms=preprocess_fn,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            subset=subset,
            unique=args.unique,
            dataset=args.dataset,
        )

    num_workers = (
        4 * torch.cuda.device_count()
        if torch.get_num_threads() >= 4
        else torch.get_num_threads()
    )
    # Calculate the exact number of full batches per process for distributed training.

    if num_processes > 1:

        adjusted_batch_size = int(args.batch_size / num_processes) * num_processes
        max_length = int(len(dataset) // adjusted_batch_size)
        num_samples = max_length * adjusted_batch_size

        subset_dataset = Subset(dataset, indices=range(num_samples))
    else:
        subset_dataset = dataset
        num_samples = len(dataset)

    if args.model_type == "mil_cell_clip":
        dataloader = DataLoader(
            subset_dataset,
            batch_size=int(args.batch_size / num_processes),
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
            collate_fn=collate_fn,
        )
    else:
        dataloader = DataLoader(
            subset_dataset,
            batch_size=int(args.batch_size / num_processes),
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
        )

    dataloader.num_samples = num_samples
    dataloader.label_map = dataset.label_map

    return dataloader


def collate_fn(batch):
    """Custom collate function to handle variable-length image sequences."""
    image_tuples, molecules = zip(*batch)  # Unzip batch

    images = [torch.from_numpy(img_tuple[0]) for img_tuple in image_tuples]
    images = torch.stack(images)

    channel_info = [img_tuple[1]["channels"] for img_tuple in image_tuples]

    input_ids = torch.stack([mol["input_ids"] for mol in molecules])
    attention_mask = torch.stack([mol["attention_mask"] for mol in molecules])

    mol_batch = {
        "input_ids": input_ids,  # (B, context_length)
        "attention_mask": attention_mask,  # (B, context_length)
    }

    # molecules = torch.stack(molecules)

    return (
        (
            images,
            {"channels": channel_info},
        ),
        mol_batch,
    )


def get_data_subset(dataset, n_samples=1000):
    """Returns randomly subsampled dataset. Use for debug purposes."""
    idcs = np.arange(len(dataset))
    n_samples = min(n_samples, len(dataset))
    np.random.shuffle(idcs)  # shuffles inplace
    new_idcs = idcs[:n_samples]

    return Subset(dataset, new_idcs)


def get_mean_std(loader, outfile):
    """Compute mean and standard deviation of images in the dataset."""

    # Initialize tensors to accumulate sum and squared sum
    sum_images = 0.0
    sum_squared_images = 0.0
    num_pixels = 0

    for batch in tqdm(loader):
        (images, extra_tokens), chem = batch

        images = images.to(device="cuda" if torch.cuda.is_available() else "cpu")

        # Compute the number of pixels per batch
        batch_pixels = images.size(0) * images.size(1) * images.size(2)

        # Reshape images to (batch_size * height * width, num_channels)
        images = images.view(-1, images.size(-1))

        # Accumulate the sum and sum of squares
        sum_images += torch.sum(images, dim=0)
        sum_squared_images += torch.sum(images ** 2, dim=0)
        num_pixels += batch_pixels

    # Compute mean and standard deviation
    mean = sum_images / num_pixels
    std = torch.sqrt((sum_squared_images / num_pixels) - (mean ** 2))

    print(mean, std)
    # Save the results to a file
    with open(outfile, "w") as f:
        f.write(f"Mean: {mean.tolist()}\n")
        f.write(f"Std: {std.tolist()}\n")

    return mean, std
