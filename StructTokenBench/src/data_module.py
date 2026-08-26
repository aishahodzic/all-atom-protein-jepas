import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import *
from tokenizer import *
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from tqdm import tqdm

def get_tokenizer_device(tokenizer_device) -> torch.device:
    """Get CPU or local GPU
    """
    if tokenizer_device == "cuda":
        gpu_idx = 0
        if torch.distributed.is_initialized():
            gpu_idx = torch.distributed.get_rank()
        device = torch.device(f"{tokenizer_device}:{gpu_idx}")
    elif tokenizer_device == "cpu":
        device = torch.device(tokenizer_device)
    return device

class ProteinDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer_name: str, tokenizer_device: str, seed: int, 
        micro_batch_size: int, data_args, py_logger, test_only: bool, 
        precompute_tokens: bool, tokenizer_kwargs: dict,
    ):
        super().__init__()

        self.tokenizer_name = tokenizer_name
        self.tokenizer_device = tokenizer_device
        self.tokenizer_kwargs = tokenizer_kwargs
        self.seed = seed
        self.micro_batch_size = micro_batch_size
        self.data_args = data_args
        self.py_logger = py_logger
        self.test_only = test_only
        self.precompute_tokens = precompute_tokens

        if self.test_only:
            self.all_split_names = []
        else:
            self.all_split_names = ["validation"]
        self.all_split_names += eval(self.data_args.data_name).SPLIT_NAME["test"]
        # to store device: tokenizer map to prevent multiple tokenizers on the same device
        self.device_tokenizer_map = {} 

    def prepare_data(self):
        pass
        
    def setup(self, stage=None):
        pass

    def _setup_tokenizer(self):
        """Initialize the tokenizer on appropriate device. 
        """
        device = get_tokenizer_device(self.tokenizer_device)
        if self.tokenizer_name.startswith("Wrapped"):
            # all Wrapped tokenizers deal with loading logic inside __init__() when built up
            # assume only this type of tokenizer needs to be device aware
            tokenizer = eval(self.tokenizer_name)(device=device, **self.tokenizer_kwargs)
        else:
            raise NotImplementedError
        
        self.device_tokenizer_map[device] = tokenizer
        return tokenizer
    
    def get_tokenizer(self):
        """Get the tokenizer on appropriate device, will initialize
        one if it doesn't exist.
        """
        device = get_tokenizer_device(self.tokenizer_device)
        tokenizer = self.device_tokenizer_map.get(device, None)
        if tokenizer is None:
            tokenizer = self._setup_tokenizer() 
        return tokenizer

    def get_codebook_embedding(self,):
        tokenizer = self.get_tokenizer()
        return tokenizer.get_codebook_embedding()

    def setup_hf_dataset(self, split="train"):
        """Set up HF datasets that will be consumed by the dataloader
        """
        kwargs = dict(self.data_args)
        kwargs.update({
            "split": split,
            "py_logger": self.py_logger,
            "tokenizer": self.get_tokenizer(),
            "in_memory": False,
        })
        dataset = eval(self.data_args.data_name)(**kwargs)
        dataset.evaluation_requested_split = split
        # need to shard the dataset here:
        assert torch.distributed.is_initialized()
        process_global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        tokenizer = self.get_tokenizer()
        if (
            isinstance(tokenizer, WrappedLaProteinaTokenizer)
            and tokenizer.evaluation_protocol is not None
            and world_size != 1
        ):
            raise RuntimeError(
                f"{tokenizer.evaluation_protocol} currently requires world_size=1 "
                "so token-cache cohort counts and immutable manifests are exact"
            )
        dataset.shard(shard_idx=process_global_rank, num_shards=world_size)

        if self.precompute_tokens:
            # precompute and cache the token ids:
            self.py_logger.info(
                f"Precomputing tokenized ids on {process_global_rank} with world size {world_size}..."
            )
            # cache_all_tokenized() gathers ALL structures' results from its worker Pool at
            # once (not streamed), so under the "file_system" sharing strategy (set globally
            # in run_supervised_task.py for a *different* problem -- an ancdata crash on the
            # long-running training DataLoader after hours of iteration) every tensor it
            # returns is a separate mmap in this process. For a large split (e.g. biolip2's
            # 8240 structures, ~8 tensors each) that blew straight through the kernel's
            # per-process vm.max_map_count (65530 here) -- mmap failed with ENOMEM inside the
            # Pool's background result-handling thread, which silently died and left the main
            # process hung forever (still holding its GPU) instead of crashing loudly.
            # "file_descriptor" doesn't mmap per tensor (bounded instead by ulimit -n, in the
            # millions here), so it's the right strategy for this one-time bulk gather.
            #
            # Setting the strategy directly (torch.multiprocessing.set_sharing_strategy)
            # here only changes THIS process's own state -- it does nothing for the Pool's
            # workers, which dataset/base.py spawns via multiprocessing.get_context("spawn"):
            # spawned children re-execute run_supervised_task.py's top-level code from
            # scratch, including its own set_sharing_strategy() call, independently of
            # whatever this process is doing. The env var IS inherited by spawned children
            # and that top-level call now reads it (see run_supervised_task.py), so setting
            # both here is what actually reaches the workers, not just this process.
            _prev_env = os.environ.get("TORCH_SHARING_STRATEGY")
            _prev_strategy = torch.multiprocessing.get_sharing_strategy()
            os.environ["TORCH_SHARING_STRATEGY"] = "file_descriptor"
            torch.multiprocessing.set_sharing_strategy("file_descriptor")
            try:
                dataset.cache_all_tokenized()
            finally:
                torch.multiprocessing.set_sharing_strategy(_prev_strategy)
                if _prev_env is None:
                    os.environ.pop("TORCH_SHARING_STRATEGY", None)
                else:
                    os.environ["TORCH_SHARING_STRATEGY"] = _prev_env
        
        if dataset.data_name not in ["ConformationalSwitchDataset", "CASP14Dataset", "CAMEODataset"]:
            for i in tqdm(range(len(dataset.data))):
                assert len(dataset.data[i]["real_seqs"]) == len(dataset.data[i]["token_ids"])
        return dataset

    def train_dataloader(self):
        """This will be run every epoch."""
        if self.test_only:
            return None
        
        if not hasattr(self, "train_hf_dataset"):
            self.train_hf_dataset = self.setup_hf_dataset("train")
        
        train_dataset = self.train_hf_dataset
        loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=self.data_args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.data_args.prefetch_factor,
        )
        self.py_logger.info(f"Finished loading training data: {len(train_dataset)} samples")
        return loader

    def val_dataloader(self):
        """Prepare both val and test sets here"""
        loaders = []
        
        for split in self.all_split_names:
            if not hasattr(self, f"{split}_hf_dataset"):
                setattr(self, f"{split}_hf_dataset", self.setup_hf_dataset(split))

            dataset = getattr(self, f"{split}_hf_dataset")
            loader = DataLoader(
                dataset,
                batch_size=self.micro_batch_size,
                collate_fn=dataset.collate_fn,
                num_workers=self.data_args.num_workers,
                shuffle=False,
                pin_memory=True,
                prefetch_factor=self.data_args.prefetch_factor,
            )
            self.py_logger.info(f"Finished loading {split} data: {len(dataset)} samples")
            loaders.append(loader)
        return loaders

    def evaluation_cohort_reports(self):
        """Return reports for every split materialized during fit/validation."""

        reports = {}
        split_attributes = [("train", "train_hf_dataset")]
        split_attributes.extend(
            (split, f"{split}_hf_dataset") for split in self.all_split_names
        )
        for requested_split, attribute in split_attributes:
            dataset = getattr(self, attribute, None)
            if dataset is None:
                continue
            report = getattr(dataset, "cohort_report", None)
            if report is not None:
                reports[requested_split] = report
        return reports


class PretrainingDataModule(pl.LightningDataModule):

    def __init__(self, device: str, seed: int, 
        micro_batch_size: int, data_args, py_logger, test_only,
    ):
        super().__init__()

        self.device = device
        self.seed = seed
        self.micro_batch_size = micro_batch_size
        self.data_args = data_args
        self.py_logger = py_logger
        self.test_only = test_only

        self.all_split_names = []
        if not self.test_only:
            self.all_split_names += ["validation"]
        self.all_split_names += eval(self.data_args.data_name).SPLIT_NAME["test"]

        # to store device: tokenizer map to prevent multiple tokenizers on the same device
        self.device_tokenizer_map = {}
    
    def _setup_tokenizer(self):
        device = get_tokenizer_device(self.device)
        tokenizer = EsmSequenceTokenizer()
        self.device_tokenizer_map[device] = tokenizer
        return tokenizer

    def get_tokenizer(self):
        """Get the tokenizer on appropriate device, will initialize
        one if it doesn't exist.
        """
        device = get_tokenizer_device(self.device)
        tokenizer = self.device_tokenizer_map.get(device, None)
        if tokenizer is None:
            tokenizer = self._setup_tokenizer() 
        return tokenizer 
    
    def setup_hf_dataset(self, split="train"):
        """Set up HF datasets that will be consumed by the dataloader
        """
        kwargs = dict(self.data_args)
        kwargs.update({
            "split": split,
            "py_logger": self.py_logger,
            "seq_tokenizer": self.get_tokenizer(),
            "in_memory": False,
        })
        dataset = eval(self.data_args.data_name)(**kwargs)
        # need to shard the dataset here:
        if torch.distributed.is_initialized():
            process_global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            dataset.shard(shard_idx=process_global_rank, num_shards=world_size)
        return dataset
    
    def train_dataloader(self):
        """This will be run every epoch."""
        if self.test_only:
            return None

        if not hasattr(self, "train_hf_dataset"):
            self.train_hf_dataset = self.setup_hf_dataset("train")
        
        train_dataset = self.train_hf_dataset
        loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=self.data_args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.data_args.prefetch_factor,
        )
        self.py_logger.info(f"Finished loading training data: {len(train_dataset)} samples")
        return loader

    def val_dataloader(self):
        """Prepare both val and test sets here"""
        loaders = []
        
        for split in self.all_split_names:
            if not hasattr(self, f"{split}_hf_dataset"):
                setattr(self, f"{split}_hf_dataset", self.setup_hf_dataset(split))

            dataset = getattr(self, f"{split}_hf_dataset")
            loader = DataLoader(
                dataset,
                batch_size=self.micro_batch_size,
                collate_fn=dataset.collate_fn,
                num_workers=self.data_args.num_workers,
                shuffle=False,
                pin_memory=True,
                prefetch_factor=self.data_args.prefetch_factor,
            )
            self.py_logger.info(f"Finished loading {split} data: {len(dataset)} samples")
            loaders.append(loader)
        return loaders

    def setup(self, stage=None):
        pass
