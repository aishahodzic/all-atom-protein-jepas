import hashlib
import os
import time
import multiprocessing as mp
from tqdm import tqdm
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from Bio import PDB
from biotite.sequence import Alphabet, Sequence, GeneralSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix

from protein_chain import WrappedProteinChain
from tokenizer import *
import util

def convert_chain_id(pdb_path, chain_id):

    if pdb_path.endswith(".pdb"):
        parser = PDB.PDBParser(QUIET=True)
    else:
        parser = PDB.MMCIFParser(QUIET=True)
    
    structure = parser.get_structure("check", pdb_path)
    if chain_id in structure[0]:
        return chain_id, False

    atom_array = convert.get_structure(CIFFile.read(pdb_path), model=1, 
                        extra_fields=["b_factor"])
    new_atom_array = convert.get_structure(CIFFile.read(pdb_path), model=1, 
                        extra_fields=["b_factor"], use_author_fields=False)
    chain_id_mapping = [(x,y) for x,y in zip(atom_array.chain_id, new_atom_array.chain_id) if y == chain_id]
    
    assert len(set([x[0] for x in chain_id_mapping])) == 1
    
    new_chain_id = chain_id_mapping[0][0]
    return new_chain_id, True


_CHAIN_ID_CACHE_DIRNAME = "chain_id_cache"


def resolve_chain_id_cached(pdb_path, chain_id, pdb_data_dir):
    """convert_chain_id's result depends only on file content + the input
    chain_id -- nothing task/checkpoint-specific -- so cache it on disk,
    keyed by (pdb_id, chain_id), reused across every future run/task/checkpoint
    instead of re-parsing the same file with Bio.PDB/biotite every time (the
    parsed *graph* was already cached this way; chain-id resolution wasn't,
    so re-running eval for a second/third checkpoint was re-paying this cost
    for identical results).
    """
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    cache_dir = os.path.join(pdb_data_dir, _CHAIN_ID_CACHE_DIRNAME)
    safe_chain = "".join(c if c.isalnum() else "_" for c in chain_id) or "_"
    cache_file = os.path.join(cache_dir, f"{pdb_id}__{safe_chain}.txt")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return f.read().strip()
    resolved_chain_id, _ = convert_chain_id(pdb_path, chain_id)
    os.makedirs(cache_dir, exist_ok=True)
    # write-then-rename: concurrent workers resolving the same pair won't
    # collide mid-write (each writes its own tmp file first)
    tmp_file = f"{cache_file}.tmp{os.getpid()}"
    with open(tmp_file, "w") as f:
        f.write(resolved_chain_id)
    os.replace(tmp_file, cache_file)
    return resolved_chain_id


_worker_lookups = {}  # populated once per worker process by _init_worker, not re-sent per task


def _init_worker(resolution_lookup, experiment_type_lookup, seqres_lookup):
    """Pool(initializer=...): sets per-process globals once, instead of
    re-pickling+sending these large dicts on every starmap task."""
    _worker_lookups["resolution"] = resolution_lookup
    _worker_lookups["experiment_type"] = experiment_type_lookup
    _worker_lookups["seqres"] = seqres_lookup


def _resolve_and_parse_worker_star(args):
    """pool.imap() (unlike starmap) only takes a single-arg callable -- this
    just unpacks the tuple. See _resolve_and_parse_worker_star's call site
    for why imap (streamed) replaced starmap (blocking, all-at-once)."""
    return _resolve_and_parse_worker(*args)


def _detach_shared_tensors(obj):
    """Clones every torch.Tensor attribute of a received graph object to
    plain (non-shared) memory, so the shared-memory IPC handle it arrived
    on (fd under "file_descriptor", mmap'd file under "file_system") can be
    released right away instead of staying alive until the whole precompute
    finishes. See the imap loop below for why this matters."""
    if obj is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if isinstance(v, torch.Tensor):
                setattr(obj, k, v.clone())
    return obj


def _resolve_and_parse_worker(pdb_path, chain_id, is_atlas, pdb_data_dir,
                               min_length=None, max_length=None, exclude_ids=frozenset(),
                               remove_non_standard_residues=False, experiment_types=frozenset(),
                               best_resolution=None, worst_resolution=None):
    """Runs in a worker process: resolves the chain ID (resolve_chain_id_cached,
    disk-cached across tasks/checkpoints), checks PDBDataSelector's criteria
    from cheap metadata BEFORE parsing (so excluded structures skip the
    expensive parse entirely, matching PDBDataSelector's own
    filter-before-process order), and only then calls
    WrappedLaProteinaTokenizer._parse_graph (LaProteina's own pipeline,
    called directly) for structures that pass. Failures return
    (None, None, None, None); the caller falls back to the normal per-item
    path, which hits + reports the same error.

    Remaining args mirror PDBDataSelector, read from the checkpoint's own
    config (WrappedLaProteinaTokenizer.__init__).
    """
    try:
        if is_atlas:
            resolved_chain_id = " "
        else:
            resolved_chain_id = resolve_chain_id_cached(pdb_path, chain_id, pdb_data_dir)

        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]

        metadata_match = WrappedLaProteinaTokenizer.matches_training_distribution_metadata(
            pdb_id, resolved_chain_id, _worker_lookups.get("seqres"),
            min_length, max_length, exclude_ids, remove_non_standard_residues,
            experiment_types, best_resolution, worst_resolution,
            _worker_lookups.get("resolution"), _worker_lookups.get("experiment_type"))
        if metadata_match is False:
            return resolved_chain_id, None, False, False

        was_pretrained_hit = WrappedLaProteinaTokenizer._in_pretrained_cache(pdb_id, resolved_chain_id)
        graph = WrappedLaProteinaTokenizer._parse_graph(pdb_path, resolved_chain_id)

        if metadata_match is True:
            matches = True  # already confirmed via metadata
        else:  # metadata_match is None: pdb_id/chain not in pdb_seqres.txt (e.g. AtlasDataset)
            matches = WrappedLaProteinaTokenizer.matches_training_distribution_graph(
                graph, pdb_id, min_length, max_length, exclude_ids, remove_non_standard_residues,
                experiment_types, best_resolution, worst_resolution,
                _worker_lookups.get("resolution"), _worker_lookups.get("experiment_type"))
        return resolved_chain_id, graph, matches, was_pretrained_hit
    except Exception:
        return None, None, None, None


class BaseDataset(Dataset):

    NONE_RETURN_LOAD_STRUCTURE = {
        "pdb_id": None, 
        "chain_id": None,
        "residue_range": None,
        "pdb_chain": None,
    }

    def __init__(self, *args, **kwargs):
        """
        in kwargs:
            data_path: data storage directory prefix
            target_field: target label name
            split: "train", "valid", or "test"
            py_logger: python logger
            tokenizer: sequence tokenizer or structural tokenzier
            in_memory: False
        """
        self.data_path = kwargs["data_path"]
        self.target_field = kwargs["target_field"]
        self.truncation_length = kwargs["truncation_length"]
        self.filter_length = kwargs["filter_length"]
        self.split = kwargs["split"]
        self.py_logger = kwargs["py_logger"]
        self.structure_pad_token_id = kwargs["tokenizer"].pad_token_id
        self.multi_label = kwargs["multi_label"]
        self.is_global_or_local = kwargs["is_global_or_local"]
        self.PDB_DATA_DIR = kwargs["pdb_data_dir"]
        self.fast_dev_run = kwargs.get("fast_dev_run", False)
        self.data_name = kwargs["data_name"]

        self.use_continuous = kwargs["use_continuous"]
        # `use_sequence`` for BaseDataset is always set to True to pass sequence
        # information to models, while `use_sequence` for the model itself is 
        # False by default to disable using sequence during tokenization
        self.use_sequence = True

        # try to load pre-processed data
        target_split_file = self.get_target_file_name()
        
        if os.path.exists(target_split_file):
            self.data = torch.load(target_split_file, weights_only=False)
            self.py_logger.info(f"Loading from processed file {target_split_file},"
                                f"structured data of {len(self.data)} entries.")
        else:
            self.py_logger.info(f"Cannot load from processed file {target_split_file} "
                                f"for structured data")
            if dist.is_initialized():
                assert dist.get_world_size() == 1
            # process data entries from raw data, different for every datasets
            self.process_data_from_scratch(*args, **kwargs)

            # preprocess index mappings before loading PDB structures, different for every datasets
            self.prepare_structure_loading()
                
            self.load_all_structures()

            self.sanity_check()
            # save to disk
            self.save_structured_data()
            
        # Dataset sharding will be done in LightningDataModule

        # assign tokenizer if haven't been assign in `process_data_from_scratch`
        if not hasattr(self, "tokenizer"):
            self.tokenizer = kwargs["tokenizer"]

        self.patch_due_to_protokens()

        self.patch_for_TAPE_homo()

    def patch_due_to_protokens(self,):
        """filter because ProTokens cannot proceed proteins longer than 1024
        """
        len_limit = 1024
        new_data = []
        if self.data_name == "ConformationalSwitchDataset":
            for i in range(len(self.data)):
                if (len(self.data[i]["prot1_pdb_chain"].sequence) <= len_limit 
                    and len(self.data[i]["prot2_pdb_chain"].sequence) <= len_limit):
                    new_data.append(self.data[i])
        else:
            for i in range(len(self.data)):
                if len(self.data[i]["pdb_chain"].sequence) <= len_limit:
                    new_data.append(self.data[i])
            
        if len(new_data) != len(self.data):
            self.data = new_data
            self.py_logger.info(f"reduce sequence lengths because of ProTokens from {len(self.data)} to {len(new_data)}")

    def patch_for_TAPE_homo(self,):
        """
        Filter proteins causing error in TAPE RH, which are indexed at 11220 (out of 12071) and 11958 (out of 12070)
        Error Example: 
            Bio.PDB.PDBExceptions.PDBConstructionException: Blank altlocs in duplicate residue SER (' ', 22, ' ') of chain 'A'
        Error Explanation: https://biopython.org/wiki/Reading_large_PDB_files
        """
        if self.data_name == "TapeRemoteHomologyDataset" and self.split == "train":
            skip_index = 11220
            self.data = self.data[:skip_index] + self.data[skip_index + 1:]
            skip_index = 11958
            self.data = self.data[:skip_index] + self.data[skip_index + 1:]
    
            self.py_logger.info(f"reduce sequence lengths for TAPE Homo to {len(self.data)}")
    
    def get_target_file_name(self,):
        assert NotImplementedError

    def save_structured_data(self, ):
        file = self.get_target_file_name()
        torch.save(self.data, file)
        self.py_logger.info(f"Save the processed, structured data to disk: {file}")
    
    def prepare_structure_loading(self):
        assert NotImplementedError

    def collate_fn(self, batch):
        """passed to DataLoader as collate_fn argument"""
        batch = list(filter(lambda x: x is not None, batch))

        input_ids, labels, seqs = tuple(zip(*batch))
        disc_flag, cont_flag = [], []
        for key in ALL_TOKENIZER_TYPE["discretized"]:
            disc_flag.append(isinstance(self.tokenizer, eval(key)))
        for key in ALL_TOKENIZER_TYPE["continuous"]:
            cont_flag.append(isinstance(self.tokenizer, eval(key)))
        disc_flag, cont_flag = any(disc_flag), any(cont_flag)
        
        if cont_flag:
            length_list = [len(x) for x in input_ids]
        
        input_ids = util.pad_structures(input_ids, 
                        constant_value=self.structure_pad_token_id, 
                        truncation_length=self.truncation_length)
        # input_ids:
        ## discretized verson: [B, L] for structural ids
        ## continuous version: [B, L, hidden_dim] (e.g., for tokenizer like ProteinMPNN)
        
        if disc_flag:
            input_mask = input_ids == self.structure_pad_token_id
        elif cont_flag:
            input_mask = torch.ones((input_ids.shape[0], input_ids.shape[1]), dtype=torch.bool, device=input_ids.device)
            for i in range(len(input_mask)):
                input_mask[i][length_list[i]:] = False
            input_mask = ~input_mask
        else:
            raise NotImplementedError  
        # input_mask: always [B, L]

        try:
            labels = torch.LongTensor(labels)
            # labels: [B,] for global labels
        except:
            labels = util.pad_structures(labels, constant_value=-100,
                        truncation_length=self.truncation_length,
                        pad_length=input_ids.shape[1])
            assert labels.shape == input_ids.shape[:2]
            # labels: always [B, L] for local labels
        
        for seq in seqs:
            assert max(seq) < 26
        seqs = util.pad_structures(seqs, constant_value=26, # 26 alphabets
                        truncation_length=self.truncation_length,
                        pad_length=input_ids.shape[1])
        assert seqs.shape == input_ids.shape[:2]

        return {
            "input_list": (input_ids, input_mask, seqs),
            "targets": labels
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def get_pdb_chain(self, pdb_id, chain_id):
        try:
            file = os.path.join(self.PDB_DATA_DIR, f"mmcif_files/{pdb_id}.cif")
            protein_chain = WrappedProteinChain.from_cif(file, 
                                                chain_id=chain_id, id=pdb_id)
        except:
            self.py_logger.info(f"Cannot retrieve from local cluster, pdb_id: {pdb_id}, chain_id: {chain_id}")
            return None
        return protein_chain
    
    def _get_init_cnt_stats(self):
        return {}
    
    def load_structure(self, idx, cnt_stats):
        """
        Arguments:
            idx: index for self.data list
            cnt_stats: a dict to calculate statistics for unsable data entries
        Return:
            {
                "pdb_id": pdb_id, 
                "chain_id": chain_id,
                "residue_range": residue_range,
                "pdb_chain": pdb_chain, 
                "local_label": local_label # optional
            }
            # residue_range default as [""] to indicate the whole protein; 
            # e.g., ["6-100"] to indicate PDB residue_index ranging from 6 to 100
        """
        assert NotImplementedError
        
    def load_all_structures(self, ):
        """For each pdb_id in self.data[], load its pdb structures by
        calling self.load_structure()
        """
        process_global_rank = 0
        if torch.distributed.is_initialized():
            process_global_rank = torch.distributed.get_rank()
        self.py_logger.info(f"Loading total {len(self.data)} structures on "
                            f"device {process_global_rank}")
        
        cnt_stats = self._get_init_cnt_stats()
        if self.fast_dev_run:
            self.data = self.data[:16]
        for i in tqdm(range(len(self.data))):
            res = self.load_structure(i, cnt_stats)
            
            for k in res.keys():
                self.data[i][k] = res[k]
            assert "pdb_id" in res
            assert "chain_id" in res
            assert "residue_range" in res
            assert "pdb_chain" in res

        self.py_logger.info(f"Processing all structures results in count "
                            f"statistics: {cnt_stats}")
        
        bg_time = time.time()
        new_data = []
        for i in range(len(self.data)):
            if not self.data[i]["pdb_id"] is None:
                new_data.append(self.data[i])
        ed_time = time.time()
        print("Timing: ", (ed_time - bg_time))

        self.py_logger.info(f"After loading structure filtering, original {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")
        self.data = new_data
    
    def sanity_check(self):
        """Filter according to length
        """

        new_data = []
        for item in self.data:
            pdb_chain, residue_range = item["pdb_chain"], item["residue_range"]
            selected_indices = self._get_selected_indices(pdb_chain.residue_index, residue_range)
            if len(selected_indices) == 0:
                continue
            # filter proteins that are too long
            if len(selected_indices) > self.filter_length:
                continue
            new_data.append(item)
        self.data = new_data

        self.py_logger.info(f"After sanity check for selected residues, original {len(self.data)} "
                            f"entries are reduced to {len(new_data)} entries.")

    def _get_selected_indices(self, residue_index, residue_range):
        """
        Arguments:
            residue_range: residue range with format like ["5-10", "20-300"] (default [""])
        """
        rr = residue_range
        if len(rr) == 1 and rr[0] == "":
            return np.arange(len(residue_index))
        
        left = [eval(sep.split("-")[0]) for sep in rr]
        right = [eval(sep.split("-")[1]) for sep in rr]
        rr_indices = [x for l, r in zip(left, right) for x in list(range(l, r+1))]

        selected_indices = []
        for i, ridx in enumerate(residue_index):
            if ridx in rr_indices:
                selected_indices.append(i)

        return selected_indices # a list
    
    def retrieve_pdb_path(self, pdb_id, chain_id):
        # specifically defined for ATLAS, PretrainPDB, CASP14 and CAMEO
        file = os.path.join(self.PDB_DATA_DIR, f"mmcif_files/{pdb_id}.cif")
        return file
    
    def resolve_pdb_path_and_chain(self, index):
        """Resolves the on-disk pdb_path and (possibly author-field-converted)
        chain_id for self.data[index]. Check self._resolved_path_chain_cache
        first (populated ahead of time, in parallel, by cache_all_tokenized
        for WrappedLaProteinaTokenizer) so this doesn't redo that work
        sequentially per item; resolve_chain_id_cached below also persists to
        disk across runs/tasks/checkpoints either way.
        """
        cache = getattr(self, "_resolved_path_chain_cache", None)
        if cache is not None and index in cache:
            return cache[index]

        item = self.data[index]
        pdb_id, chain_id = item["pdb_id"], item["chain_id"]
        pdb_path = self.retrieve_pdb_path(pdb_id, chain_id)

        if self.data_name == "AtlasDataset":
            chain_id = " "
        else:
            # convert chain_id if necessary because some chain_id needs to
            # use use_author_field (specified in biotite).
            # except atlas, other datasets' pdb_path is independent of chain_id;
            # and for atlas, there is no need to transform chain_id
            chain_id = resolve_chain_id_cached(pdb_path, chain_id, self.PDB_DATA_DIR)
        return pdb_path, chain_id

    def _get_item_structural_tokens(self, index, skip_check=False):
        
        item = self.data[index]
        if not skip_check:
            if "token_ids" in item:
                if self.is_global_or_local == "local":
                    assert len(item["token_ids"]) == len(item[self.target_field])
                return item["token_ids"], item[self.target_field], item["real_seqs"]
    
        pdb_chain, residue_range = item["pdb_chain"], item["residue_range"]
        pdb_path, chain_id = self.resolve_pdb_path_and_chain(index)
        assigned_labels = item[self.target_field]
        assert pdb_chain is not None
        
        if self.is_global_or_local == "local":
            assert len(residue_range) == 1 and residue_range[0] == ""
        
            if self.data_name in "ProteinShakeBindingSiteDataset":
                label_residue_index = item["residue_index"]
            elif self.data_name in ["BioLIP2FunctionDataset", 
                "InterProFunctionDataset", "ProteinGLUEEpitopeRegionDataset", 
                "AtlasDataset"]:
                # all local labels already aligned to pdb_chain.residue_index
                label_residue_index = pdb_chain.residue_index
            else:
                raise NotImplementedError
            
            assert len(assigned_labels) == len(label_residue_index)


        # encode protein structure into token_ids
        if isinstance(self.tokenizer, WrappedESM3Tokenizer):
            # chain_id conversion is already automatically dealt with 
            # WrappedProteinChain, and produced pdb_chain
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_chain, self.use_continuous, self.use_sequence) # torch.Tensors
        elif isinstance(self.tokenizer, WrappedFoldSeekTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_continuous, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedProTokensTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_continuous, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedProteinMPNNTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedMIFTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedOurPretrainedTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_chain, self.use_continuous, self.use_sequence) # torch.Tensors
        elif isinstance(self.tokenizer, WrappedAIDOTokenizer):
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_continuous, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedCheapS1D64Tokenizer):
            # CheapS1D64 is continuous tokenizer
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_sequence)
        elif isinstance(self.tokenizer, WrappedLaProteinaTokenizer):
            # LaProteina is a continuous tokenizer, same 3-arg signature as CheapS1D64/MIF
            token_ids, residue_index, seqs = self.tokenizer.encode_structure(pdb_path, chain_id, self.use_sequence)
        
        else:
            raise NotImplementedError
        
        assert len(token_ids) == len(residue_index)
        # code compatability in case token_ids store continuous reprs
        token_ids = token_ids.detach()
        assert len(residue_index) == len(seqs)
        
        if self.is_global_or_local == "local":
            # align residue_index and label_residue_index, so that token_ids align with assigned_labels
            org_len = len(token_ids)
            align_indices_1 = [i for i, x in enumerate(label_residue_index) if x in residue_index]
            label_residue_index = np.array(label_residue_index)[align_indices_1].tolist()
            assigned_labels = np.array(assigned_labels)[align_indices_1].tolist()

            align_indices_2 = [i for i, x in enumerate(residue_index) if x in label_residue_index]
            residue_index, token_ids = residue_index[align_indices_2], token_ids[align_indices_2]
            seqs = [x for i,x in enumerate(seqs) if i in set(align_indices_2)]

            try:
                assert (residue_index == np.array(label_residue_index)).all()
            except:
                # deal with repeated residue indices and achieve exact match with alignment
                idx_list = list(set(residue_index.tolist() + label_residue_index))
                
                alphabet = Alphabet(idx_list)
                sim_score = np.diag(np.ones(len(idx_list), dtype=int))
                substitution_matrix = SubstitutionMatrix(alphabet, alphabet, sim_score)
                seq1 = GeneralSequence(alphabet, label_residue_index)
                seq2 = GeneralSequence(alphabet, residue_index.tolist())
                alignment = align_optimal(seq1, seq2, substitution_matrix)
                
                alignment = alignment[0].trace
                align_indices_1, align_indices_2 = [], []
                for i in range(len(alignment)):
                    if (alignment[i] != -1).all():
                        align_indices_1.append(alignment[i][0])
                        align_indices_2.append(alignment[i][1])

                label_residue_index = np.array(label_residue_index)[align_indices_1].tolist()
                assigned_labels = np.array(assigned_labels)[align_indices_1].tolist()
                residue_index, token_ids = residue_index[align_indices_2], token_ids[align_indices_2]
                seqs = [x for i,x in enumerate(seqs) if i in set(align_indices_2)]


            if org_len - len(token_ids) != 0:
                print(">> residue reduced by : ", org_len - len(token_ids))

        # select according to residue range constraints for some global tasks
        selected_indices = self._get_selected_indices(residue_index, residue_range)
        assert len(selected_indices) != 0
        
        token_ids = token_ids[selected_indices]
        seqs = np.array(seqs)[selected_indices].tolist()
        if self.is_global_or_local == "local":
            assigned_labels = np.array(assigned_labels)[selected_indices].tolist()

        # cache the tokens
        self.data[index]["token_ids"] = token_ids.to("cpu").detach().clone()
        self.data[index][self.target_field] = assigned_labels
        self.data[index]["real_seqs"] = seqs
        if self.is_global_or_local == "local":
            assert len(token_ids) == len(assigned_labels)
        return token_ids, assigned_labels, seqs # torch.Tensor, List

    def __getitem__(self, index: int):
        return self._get_item_structural_tokens(index)

    def additional_label_filtering_for_TAPE_homo(self, tokenizer_name):

        if self.data_name == "TapeRemoteHomologyDataset":
            """
            The original TAPE dataset consists of 1195 labels.
            Filter label class that has less than 50 protein samples in the 
            training dataset, reducing from 1195 labels to 45 labels
            """

            labels_to_filter = set([
                22, 36, 47, 51, 73, 77, 78, 84, 88, 90, 126, 153, 176, 295, 
                0, 3, 21, 39, 45, 59, 70, 97, 179,
                26, 49, 60, 81, 95, 113, 124, 133, 143, 178,
                13, 14, 18, 42, 52, 56, 61, 91, 132, 135, 180, 246
            ])
            labels_mapping = {x: i for i, x in enumerate(sorted(list(labels_to_filter)))}

            assert self.target_field == "fold_label"
            new_data = []
            for x in self.data:
                if x[self.target_field] in labels_to_filter:
                    x[self.target_field] = labels_mapping[x[self.target_field]]
                    new_data.append(x)
            self.data = new_data

        if self.data_name == "TapeRemoteHomologyDataset" and tokenizer_name == "protokens":
            # filter 1ldt.cif
            new_data = []
            for i in range(len(self.data)):
                if self.data[i]["pdb_id"] != "1ldt":
                    new_data.append(self.data[i])
            self.data = new_data
        
    def additional_preprocessing_for_TAPE_homo(self, tokenizer_name):
        """
        Some proteins are skipped, because for all their residues, at least 
        one backbone coordinates are NaN
        """
        if ((tokenizer_name == "proteinmpnn" or tokenizer_name == "mif") 
            and self.data_name == "TapeRemoteHomologyDataset"):
            
            if self.split == "train":
                skip_index = set([793, 796, 894, 1119, 1200, 1303, 1315, 1686, 1966, 2359, 
                            2583, 3302, 4239, 4406, 4769, 4904, 7669, 9642, 9903, 9933, 
                            9937, 10517, 11832, 11836, 11958])
                new_data = []
                for i in range(len(self.data)):
                    if i not in skip_index:
                        new_data.append(self.data[i])
                self.data = new_data

            if self.split == "valid":
                
                skip_index = set([499, 619, 630])
                new_data = []
                for i in range(len(self.data)):
                    if i not in skip_index:
                        new_data.append(self.data[i])
                self.data = new_data
            
            if self.split == "test_family_holdout":
                skip_index = set([41, 828, 1131])
                new_data = []
                for i in range(len(self.data)):
                    if i not in skip_index:
                        new_data.append(self.data[i])
                self.data = new_data
            
            if self.split == "test_superfamily_holdout":
                skip_index = set([97, 111, 115, 129, 350])
                new_data = []
                for i in range(len(self.data)):
                    if i not in skip_index:
                        new_data.append(self.data[i])
                self.data = new_data
            
    
    def cache_all_tokenized(self):
        """Precompute all tokenization results"""
        
        flag_list, name_list, type_list = [], [], []
        for tp in ALL_TOKENIZER_TYPE:
            for key in ALL_TOKENIZER_TYPE[tp]:
                flag_list.append(isinstance(self.tokenizer, eval(key)))
                name_list.append(key.replace("Wrapped", "").replace("Tokenizer", "").lower())
                type_list.append(tp)
        flag = any(flag_list)
        
        if flag:
            index = np.nonzero(flag_list)[0].item()
            tokenizer_name = name_list[index]
            if isinstance(self.tokenizer, WrappedOurPretrainedTokenizer):
                tokenizer_name += f"_{self.tokenizer.ckpt_name}"
            elif isinstance(self.tokenizer, WrappedLaProteinaTokenizer):
                # basename alone collides (every checkpoint is 100k.ckpt or
                # 100k-EMA.ckpt, differing only by parent dir) -- hash the
                # resolved path too so different checkpoints get different caches.
                ckpt_basename = os.path.splitext(os.path.basename(self.tokenizer.pretrained_ckpt_path))[0]
                path_hash = hashlib.sha256(
                    os.path.realpath(self.tokenizer.pretrained_ckpt_path).encode()
                ).hexdigest()[:10]
                ckpt_id = f"{ckpt_basename}_{path_hash}"
                tokenizer_name += f"_{self.tokenizer.checkpoint_type}_{ckpt_id}"

            # use continous reprs
            continuous_flag = self.use_continuous
            if type_list[index] == "continuous":
                # continous flag only for discretized tokenizers (i.e., VQ-VAE-based PSTs)
                # set to False to avoid redundancy for continuous tokenizers
                continuous_flag = False
            continuous_flag = "" if not continuous_flag else "_continuous"

            # use sequence ids
            sequence_flag = "" if not self.use_sequence else "_sequence"

            # cache file to avoid redundant tokenizing for the same tokenizer 
            # when tuning hyper-parameters 
            cache_file_name = self.get_target_file_name() + f"_{tokenizer_name}_tokenized{continuous_flag}{sequence_flag}"
            if os.path.exists(cache_file_name):
                try:
                    new_data = torch.load(cache_file_name, weights_only=False)
                except Exception as e:
                    # a prior crash mid-write (e.g. process killed during
                    # torch.save) can leave a truncated/corrupt cache file;
                    # rebuild instead of hard-failing every run afterwards.
                    self.py_logger.info(
                        f"[Warning] cached tokenized data at {cache_file_name} "
                        f"failed to load ({e}); treating as corrupt and re-caching")
                    os.remove(cache_file_name)
                    new_data = None
                if (new_data and self.data_name == "AtlasDataset"
                        and self.target_field not in new_data[0]):
                    # get_target_file_name() has no target_field in it, so
                    # FlexRMSF/FlexBFactor/FlexNEQ share one cache file per
                    # checkpoint. A cache another Flex task built lacks *this*
                    # task's target_field -- treat as a miss, not a hit, so it
                    # gets populated here (main process) rather than later
                    # inside a forked DataLoader worker (CUDA re-init crash).
                    self.py_logger.info(
                        f"[Warning] cached tokenized data at {cache_file_name} was "
                        f"built for a different target_field (missing "
                        f"'{self.target_field}'); treating as a miss and re-caching")
                    new_data = None
                if new_data is not None:
                    self.data = new_data
                    self.additional_label_filtering_for_TAPE_homo(tokenizer_name)
                    self.py_logger.info(f"Loading cahced tokenized data from {cache_file_name}")
                    return
            else:
                self.py_logger.info(f"Cannot load cahced tokenized data from {cache_file_name}, caching now")
        else:
            raise NotImplementedError
                
        
        # cache-hit path (above) already calls this; fresh-build path didn't,
        # so a freshly-saved cache kept TAPE's raw 1195 fold_label values
        # instead of the 45-class subset num_labels expects -> out-of-range
        # CrossEntropyLoss target -> "CUDA error: device-side assert triggered".
        self.additional_label_filtering_for_TAPE_homo(tokenizer_name)
        self.additional_preprocessing_for_TAPE_homo(tokenizer_name)

        if isinstance(self.tokenizer, WrappedLaProteinaTokenizer):
            # Per item, encode_structure has two expensive file parses:
            # convert_chain_id (~0.1-1.6s) and the graph parse (~1.2s). Both
            # run here, per item, in the same parallel worker call, instead of
            # sequentially in the main process. Results are cached by index/
            # (path, chain) so the sequential loop below and
            # resolve_pdb_path_and_chain() hit warm caches instead of redoing this.
            num_workers = min(8, os.cpu_count() or 1)
            is_atlas = self.data_name == "AtlasDataset"
            index_to_pair = {}  # index -> (pdb_path, raw_chain_id), cheap string joins only
            for index in range(len(self)):
                item = self.data[index]
                pdb_id, chain_id = item["pdb_id"], item["chain_id"]
                pdb_path = self.retrieve_pdb_path(pdb_id, chain_id)
                index_to_pair[index] = (pdb_path, chain_id)

            # dedupe: the same (pdb_path, chain_id) can recur across many rows
            # (e.g. one BioLIP2 structure with several annotated ligands/sites)
            # -- parse each unique structure once, not once per row.
            unique_pairs = list(dict.fromkeys(index_to_pair.values()))
            self.py_logger.info(
                f"Parallel-resolving chain IDs + parsing {len(unique_pairs)} unique "
                f"structures (from {len(index_to_pair)} entries) across {num_workers} "
                f"worker processes (LaProteina's own processed/ dir persists these "
                f"across tasks/experiments)...")
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                num_workers, initializer=_init_worker,
                initargs=(self.tokenizer.resolution_lookup, self.tokenizer.experiment_type_lookup,
                          self.tokenizer.seqres_lookup),
            ) as pool:
                # imap (streamed, one result at a time), NOT starmap (blocking,
                # gathers all len(unique_pairs) results before returning) --
                # each result carries a graph with several torch tensors, and
                # for a large split (e.g. biolip2's 8240 structures) having
                # all of them alive simultaneously as shared-memory IPC
                # objects blows through OS-level ceilings on EITHER sharing
                # strategy: "file_system" hits vm.max_map_count (one mmap per
                # tensor); "file_descriptor" hits the ancdata/SCM_MAX_FD limit
                # on fd-passing. Both crash a background thread that the main
                # process doesn't notice, leaving it hung forever still
                # holding its GPU allocation. Streaming + cloning each result
                # to plain (non-shared) memory immediately keeps at most
                # ~num_workers results' worth of shared-memory handles alive
                # at once, independent of how many structures there are.
                args_list = [
                    (p, c, is_atlas, self.PDB_DATA_DIR,
                     self.tokenizer.min_length, self.tokenizer.max_length,
                     self.tokenizer.exclude_ids, self.tokenizer.remove_non_standard_residues,
                     self.tokenizer.experiment_types, self.tokenizer.best_resolution,
                     self.tokenizer.worst_resolution)
                    for p, c in unique_pairs
                ]
                results = []
                for i, result in enumerate(pool.imap(_resolve_and_parse_worker_star, args_list, chunksize=1)):
                    resolved_chain_id, graph, matches, was_pretrained_hit = result
                    results.append((resolved_chain_id, _detach_shared_tensors(graph), matches, was_pretrained_hit))
                    if (i + 1) % 1000 == 0 or (i + 1) == len(args_list):
                        self.py_logger.info(f"  ...resolved+parsed {i + 1}/{len(args_list)}")
            pair_to_result = dict(zip(unique_pairs, results))

            # Two reasons an index can't go through as-is: resolution/parsing
            # failed (unresolved_indices; sequential loop below reports the
            # error), or it's outside this checkpoint's training distribution
            # (distribution_mismatch_indices, see matches_training_distribution).
            # Both must be absent from self.data AND _resolved_path_chain_cache,
            # or a stale cache hit masks the real error deeper in the pipeline.
            unresolved_indices = set()
            distribution_mismatch_indices = set()
            graph_cache = {}
            resolved_by_index = {}
            newly_processed_pairs = []  # (pdb_id, chain_id) not in LaProteina's own pretrained cache
            for index, pair in index_to_pair.items():
                resolved_chain_id, graph, matches_distribution, was_pretrained_hit = pair_to_result[pair]
                if resolved_chain_id is None:
                    unresolved_indices.add(index)
                    continue
                if not was_pretrained_hit and graph is not None:
                    # graph is None when matches_training_distribution_metadata
                    # confidently excluded it before ever calling _parse_graph --
                    # that's not "processed here", it's "never processed".
                    pdb_id = os.path.splitext(os.path.basename(pair[0]))[0]
                    newly_processed_pairs.append((pdb_id, resolved_chain_id))
                if not matches_distribution:
                    distribution_mismatch_indices.add(index)
                    continue
                pdb_path = pair[0]
                resolved_by_index[index] = (pdb_path, resolved_chain_id)
                if graph is not None:
                    graph_cache[(pdb_path, resolved_chain_id)] = graph

            if not hasattr(self.tokenizer, "_graph_cache"):
                self.tokenizer._graph_cache = {}
            self.tokenizer._graph_cache.update(graph_cache)

            if newly_processed_pairs:
                misses_file = os.path.join(self.PDB_DATA_DIR, "not_in_laproteina_pretrained_cache.txt")
                with open(misses_file, "a") as f:
                    for pdb_id, chain_id in newly_processed_pairs:
                        f.write(f"{pdb_id}_{chain_id}\n")
                self.py_logger.info(
                    f"{len(newly_processed_pairs)}/{len(unique_pairs)} structures were not in "
                    f"LaProteina's own pretrained processed/ cache and were processed here "
                    f"(logged to {misses_file}).")

            if unresolved_indices:
                self.py_logger.info(
                    f"Excluding {len(unresolved_indices)} entries that failed to "
                    f"resolve/parse (missing files or similar).")
            if distribution_mismatch_indices:
                self.py_logger.info(
                    f"Excluding {len(distribution_mismatch_indices)} entries outside "
                    f"this checkpoint's training distribution "
                    f"(min_length={self.tokenizer.min_length}, max_length={self.tokenizer.max_length}, "
                    f"exclude_ids, non-standard residues -- from the checkpoint's own config).")

            excluded_indices = unresolved_indices | distribution_mismatch_indices
            if excluded_indices:
                self.data = [item for i, item in enumerate(self.data) if i not in excluded_indices]
                self._resolved_path_chain_cache = {
                    new_index: resolved_by_index[old_index]
                    for new_index, old_index in enumerate(
                        i for i in range(len(index_to_pair)) if i not in excluded_indices
                    )
                }
            else:
                self._resolved_path_chain_cache = resolved_by_index

        # pre-checking
        for index in tqdm(range(len(self))):
            try:
                self[index]
            except:
                self.py_logger.info(f"[Error]: Something wrong for index {index} "
                                    f"when using {tokenizer_name}\n[Warning]: if "
                                    f"you're using your own PST, you can skip wrongly "
                                    f"indexed samples for your PST. But please be aware that "
                                    f"other PST benchmakred by the authors all used these samples")
                raise IndexError
        if flag:
            # atomic write: a crash mid-torch.save (we've hit this from
            # worker/spawn code-version mismatches during active edits)
            # otherwise leaves a truncated cache_file_name that the
            # torch.load() above then fails to open on the next run.
            tmp_name = f"{cache_file_name}.tmp{os.getpid()}"
            torch.save(self.data, tmp_name)
            os.replace(tmp_name, cache_file_name)

    def shard(self, shard_idx: int, num_shards: int):
        """Shard the dataset inplace by keeping the every `num_shards`"""
        self.py_logger.info(f"Loading shard {shard_idx} with world size {num_shards}")

        indices = range(len(self))[shard_idx::num_shards]
        self.data = [self.data[i] for i in indices]

        self.py_logger.info("Done sharded loading.")

    
    def splitting_dataset(self, fold_split_ratio=0.4, fold_valid_ratio=0.2, 
        superfamily_split_ratio=0.2, superfamily_valid_ratio=0.2, seed=42
    ):
        """
        Perform splitting:
        - step 1: for each fold, split superfamilies into two groups (60%, 40%) 
            for training and test, resulting in the fold-level datasets
        - step 2: among the fold-level training data, for each superfamily, 
            split the family into two groups (60%, 40%) for training and test, 
            resulting in the superfamily-level datasets
        - Step 3: from the test data above, randomly take out 20% proteins 
            to create a validation set
        """

        # for each fold, split superfamilies
        fold_list, superfamily_list = [], []
        for i in range(len(self.data)):
            fold_list.append(self.data[i]["fold_label"])
            superfamily_list.append(self.data[i]["superfamily_label"])
        fold_list, superfamily_list = np.array(fold_list), np.array(superfamily_list)

        fold_train_indices, fold_test_indices = [], []
        for fold_idx in set(fold_list):
            indices = (fold_list == fold_idx).nonzero()[0]
            superfamily_vocab = list(set(superfamily_list[indices]))
            if int(len(superfamily_vocab) * fold_split_ratio) > 0:
                sf_train, sf_test = train_test_split(superfamily_vocab, 
                                        test_size=fold_split_ratio, random_state=seed)
                sf_train = np.isin(superfamily_list[indices], sf_train)
                sf_test = np.isin(superfamily_list[indices], sf_test)
                fold_train_indices += (indices[sf_train]).tolist()
                fold_test_indices += (indices[sf_test]).tolist()
            else:
                fold_train_indices += indices.tolist()

        fold_test_indices, fold_valid_indices = train_test_split(fold_test_indices, 
                                    test_size=fold_valid_ratio, random_state=seed)

        # among the fold-level training data, for each superfamily, random split 
        fold_train_indices = np.array(fold_train_indices)
        sf_train_indices, sf_test_indices = [], []
        for sf_idx in set(superfamily_list[fold_train_indices].tolist()):
            indices = (superfamily_list[fold_train_indices] == sf_idx).nonzero()[0]
            if int(len(indices) * superfamily_split_ratio) > 0:
                train_indices, test_indices = train_test_split(indices, 
                                        test_size=superfamily_split_ratio, random_state=seed)
                sf_train_indices += fold_train_indices[train_indices].tolist()
                sf_test_indices += fold_train_indices[test_indices].tolist()
            else:
                sf_train_indices += fold_train_indices[indices].tolist()

        sf_test_indices, sf_valid_indices = train_test_split(sf_test_indices, 
                                    test_size=superfamily_valid_ratio, random_state=seed)
        
        train_indices = sf_train_indices
        valid_indices = fold_valid_indices + sf_valid_indices
        fold_test_indices = fold_test_indices
        superfamily_test_indices = sf_test_indices

        assert len(train_indices) == len(set(train_indices))
        assert len(valid_indices) == len(set(valid_indices))
        assert len(fold_test_indices) == len(set(fold_test_indices))
        assert len(superfamily_test_indices) == len(set(superfamily_test_indices))
        assert len(self.data) == (len(set(train_indices)) + len(set(valid_indices))
                            + len(set(fold_test_indices)) + len(set(superfamily_test_indices)))

        self.py_logger.info(f"After splitting, result in {len(train_indices)} training data, "
                            f"{len(valid_indices)} validation data, {len(fold_test_indices)} fold-level test data, "
                            f"{len(superfamily_test_indices)} superfamily-level test data")
        
        train_data = [self.data[idx] for idx in train_indices]
        valid_data = [self.data[idx] for idx in valid_indices]
        fold_test_data = [self.data[idx] for idx in fold_test_indices]
        superfamily_test_data = [self.data[idx] for idx in superfamily_test_indices]
        return train_data, valid_data, fold_test_data, superfamily_test_data

    

    

