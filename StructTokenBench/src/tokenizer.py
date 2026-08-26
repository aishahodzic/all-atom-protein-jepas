import os
import time
import json
import joblib
import pickle
import urllib.request
import functools

import torch
import torch.nn.functional as F
import torch.distributed as dist

import Bio
from Bio import PDB
from biotite.structure.io.pdbx import CIFFile, convert
import numpy as np
from Bio.PDB import Chain
from typing import Literal

# ----- ESM3 Loading ------- #
from esm.models.esm3 import ESM3
from esm.pretrained import (
    ESM3_structure_encoder_v0, 
    ESM3_structure_decoder_v0, 
    ESM3_structure_decoder_v0,
)
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.constants import esm3 as C
from esm.utils.structure.affine3d import build_affine3d_from_coordinates

# ----- MIF Loading ------- #
from sequence_models.pretrained import load_model_and_alphabet as mif_load_model_and_alphabet
from sequence_models.pdb_utils import parse_PDB as mif_parse_PDB
from sequence_models.pdb_utils import process_coords as mif_process_coords


# ----- FoldSeek Loading ------- #
try:
    import mini3di
    from mini3di.utils import last
except ModuleNotFoundError:
    print("[Warining]: mini3di not found")

# ----- ProTokens Loading ------- #
try:
    # load from baselines/ProToken
    from data_process.preprocess import save_pdb_from_aux, protoken_encoder_preprocess, protoken_decoder_preprocess, init_protoken_model
    from data_process.preprocess import protoken_encoder_input_features, protoken_decoder_input_features
    from data_process import residue_constants
except ModuleNotFoundError:
    print("[Warining]: ProToken not found")

# ----- AIDO Loading ------- #

try:
    from modelgenerator.structure_tokenizer.models import EquiformerEncoderLightning
    from modelgenerator.structure_tokenizer.datasets.protein_dataset import ProteinDataset
    from modelgenerator.structure_tokenizer.datasets.protein import Protein
except ModuleNotFoundError:
    print("[Warining]: AIDO.st not found")

# ----- Cheap Loading ------- #
try:
    # load from Cheap
    from cheap_proteins.src.cheap.pretrained import (
        CHEAP_shorten_1_dim_64
    )
except ModuleNotFoundError:
    print("[Warining]: cheap not found")
    pass


# ----- ProteinMPNN Loading ------- #
try:
    from baselines.protein_mpnn_utils import ProteinMPNN, tied_featurize, parse_PDB_biounits, parse_PDB
    from baselines.protein_mpnn_cif_parser import parse_cif_pdb_biounits
except ModuleNotFoundError:
    print("[Warining]: ProteinMPNN not found")


from vqvae_model import VQVAEModel
from evaluation_manifest import sha256_file
from laproteina_checkpoint import extract_jepa_encoder_state_dict
from laproteina_eval_policy import (
    evaluation_cache_tag_suffix,
    resolve_evaluation_max_length,
    resolve_laproteina_evaluation_policy,
)

# ----- La Proteina Loading ------- #

# local imports - LaProteina
import sys
sys.path.append("/home/projects/msc-thesis")  
try:
    from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
    from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer
except ModuleNotFoundError:
    print("[Warning]: proteinfoundation (msc-thesis) not found")


from openfold.np.residue_constants import resname_to_idx
from proteinfoundation.utils.dense_padding_data_loader import dense_padded_from_data_list
from graphein.ml.datasets.pdb_data import PDBManager
# LaProteina's own training-time data pipeline, called directly (see _parse_graph)
from proteinfoundation.datasets.transforms import CoordsToNanometers, CenterStructureTransform, ChainBreakPerResidueTransform
from proteinfoundation.datasets.pdb_data import PDBLightningDataModule, PDBDataset





SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

# Reside Name Mapping:
# {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "ASX": "B", "CYS": "C", "GLN": "Q", 
# "GLU": "E", "GLX": "Z", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", 
# "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", 
# "VAL": "V", "KCX": "K"}

# All tokenizers should be registered here
# discretized tokenizers can produce both discretized and continuous strutcural reprs.
# while continuous tokenizers can ONLY produce continuous structural reprs.
ALL_TOKENIZER_TYPE = {
    "discretized": [
        "WrappedESM3Tokenizer",
        "WrappedFoldSeekTokenizer",
        "WrappedProTokensTokenizer",
        "WrappedOurPretrainedTokenizer",
        "WrappedAIDOTokenizer",
    ],
    "continuous": [
        "WrappedMIFTokenizer",
        "WrappedProteinMPNNTokenizer",
        "WrappedCheapS1D64Tokenizer", 
        "WrappedLaProteinaTokenizer"
    ]
}


class WrappedFoldSeekTokenizer():

    FOLDSEEK_STRUC_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"
    # reference to https://github.com/althonos/mini3di/blob/faeff98f8c411224fadb73e3878ef6c8ceefa887/mini3di/encoder.py#L27

    def __init__(self, device:  torch.device | str = "cpu"):
        self.device = device

        self.token2id = {s:i for i, s in enumerate(self.FOLDSEEK_STRUC_VOCAB)}
        self.pad_token_id = len(self.FOLDSEEK_STRUC_VOCAB)

        # third party implementation: https://github.com/althonos/mini3di/
        self.tokenizer_encoder = mini3di.Encoder() # based on numpy, not torch tensors
    
    def get_num_tokens(self):
        return len(self.FOLDSEEK_STRUC_VOCAB) + 1 # additional PAD token

    def get_codebook_embedding(self,):        
        return torch.tensor(self.tokenizer_encoder._CENTROIDS) # (20, 2)

    def _encode_structure_mini3di(self, pdb_path, chain_id, use_continuous=False, use_sequence=False):
        if pdb_path.endswith(".pdb"):
            parser = PDB.PDBParser(QUIET=True)
        else:
            parser = PDB.MMCIFParser(QUIET=True)
        structure = parser.get_structure("test", pdb_path)
        chain = structure[0][chain_id]
        if not use_continuous:
            states = self.tokenizer_encoder.encode_chain(chain)
            seq_mini3di = self.tokenizer_encoder.build_sequence(states)
        else:
            seq_mini3di = self.hijack_continuous_reprs(chain)
        residues = [residue for residue in chain.get_residues() if "CA" in residue] # aligned with FoldSeek's internal implementation
        residue_index = np.array([_.get_id()[1] for _ in residues])

        seqs = [Bio.PDB.Polypeptide.three_to_index(_.resname) 
                if Bio.PDB.Polypeptide.is_aa(_.resname, standard=True) else 20 for _ in residues] # 20 for unknown AA

        return seq_mini3di, residue_index, seqs
    
    def hijack_continuous_reprs(self, 
        chain: Chain,
        ca_residue: bool = True,
        disordered_atom: Literal["best", "last"] = "best",
    ):
        """
        Adapted from https://github.com/althonos/mini3di/blob/5bc2fb0257e8d743326f74615ee2c1820c66e7c1/mini3di/encoder.py#L60
        """
        # extract residues
        if ca_residue:
            residues = [residue for residue in chain.get_residues() if "CA" in residue]
        else:
            residues = list(chain.get_residues())
        # extract atom coordinates
        r = len(residues)
        ca = np.array(np.nan, dtype=np.float32).repeat(3 * r).reshape(r, 3)
        cb = ca.copy()
        n = ca.copy()
        c = ca.copy()
        for i, residue in enumerate(residues):
            for atom in residue.get_atoms():
                if atom.is_disordered() and disordered_atom == "last":
                    atom = last(atom)
                if atom.get_name() == "CA":
                    ca[i, :] = atom.coord
                elif atom.get_name() == "N":
                    n[i, :] = atom.coord
                elif atom.get_name() == "C":
                    c[i, :] = atom.coord
                elif atom.get_name() == "CB" or atom.get_name() == "CB A":
                    cb[i, :] = atom.coord
        # encode coordinates
        descriptors = self.tokenizer_encoder.feature_encoder.encode_atoms(ca, cb, n, c)
        
        # the last layer is for discretization
        reprs = functools.reduce(lambda x, f: f(x), self.tokenizer_encoder.vae_encoder.layers[:-1], descriptors.data)

        return reprs

    def encode_structure(self, pdb_path, chain_id, use_continuous=False, use_sequence=False):
        assert use_sequence
        
        seq_mini3di, residue_index, seqs = self._encode_structure_mini3di(pdb_path, chain_id, use_continuous)
        if not use_continuous:
            structural_tokens = torch.LongTensor([self.token2id[x] for x in seq_mini3di])
        else:
            structural_tokens = torch.tensor(seq_mini3di, device=self.device)

        # sometimes, seq_mini3di does match the length of pdb_chain from ESM3
        # because mini3di filter residues without CA

        return structural_tokens, residue_index, seqs
    
    
class WrappedProTokensTokenizer():

    """
    Adapted from https://colab.research.google.com/drive/15bBbfa7WigruoME089cSfE242K1MvRGz
    """

    def __init__(self, device=None):
        self.device = device

        dir_name =  os.path.join(os.path.dirname(__file__), "baselines/ProToken")
        self.tokenizer_encoder_1 = init_protoken_model(512, dir_name)
        self.tokenizer_encoder_2 = init_protoken_model(1024, dir_name)

        codebook_file_name = os.path.join(dir_name, "ProToken_Code_Book.pkl")
        self.codebook_embedding = pickle.load(open(codebook_file_name, "rb"))
        self.pad_token_id = len(self.codebook_embedding)
    
    def get_codebook_embedding(self,):
        return torch.tensor(self.codebook_embedding) # (512, 32)
    
    def get_num_tokens(self, ):
        return len(self.codebook_embedding) + 1
    
    def encode_structure(self, pdb_path, chain_id, use_continuous=False, use_sequence=False):
        assert use_sequence
        encoder_inputs, encoder_aux, seq_len = protoken_encoder_preprocess(pdb_path, task_mode="single", chain_id=chain_id)

        if seq_len <= 512:
            encoder_results = self.tokenizer_encoder_1.encoder(*encoder_inputs)
        elif seq_len <= 1024:
            encoder_results = self.tokenizer_encoder_2.encoder(*encoder_inputs)
        else:
            raise NotImplementedError

        structural_tokens, residue_index, seqs = [], [], []
        for p in range(encoder_aux['seq_mask'].shape[0]):
            if (encoder_aux['seq_mask'][p] 
            and encoder_aux["aatype"][p] < len(residue_constants.restype_order)): # < 20, may be unknown residues like HOH
                structural_tokens.append(encoder_results["protoken_index"][p])
                residue_index.append(encoder_aux["residue_index"][p])
                seqs.append(encoder_aux["aatype"][p])
        
        # tf.Tensor -> np.array -> torch.Tensor
        structural_tokens = torch.tensor(np.asarray(structural_tokens), device=self.device)
        residue_index = np.asarray(residue_index)

        assert (structural_tokens < 512).all()
        return structural_tokens, residue_index, seqs

class WrappedAIDOTokenizer():
    def __init__(self, device=None):
        self.device = device
        self.tokenizer_encoder = EquiformerEncoderLightning("genbio-ai/AIDO.StructureEncoder").to(self.device)
        self.tokenizer_encoder.training = False
        self.codebook_embedding = self.tokenizer_encoder.encoder.codebook.data.cpu()
        self.pad_token_id = len(self.codebook_embedding)
    
    def get_codebook_embedding(self,):
        return self.codebook_embedding # (512, 384)
    
    def get_num_tokens(self, ):
        return len(self.codebook_embedding) + 1
    
    def encode_structure(self, pdb_path, chain_id, use_continuous=False, use_sequence=False):
        # parse the pdb_file into a Protein object
        if pdb_path.endswith(".pdb"):
            protein = Protein.from_pdb_file_path(pdb_path, chain_id)
        elif pdb_path.endswith(".cif"):
            # 2nd arg (entity_id) will be ignored to ensure the same CIF 
            # parsing logic within this codebase
            protein = Protein.from_cif_file_path(pdb_path, 1, chain_id)
        # Do not implement the cropping logic for simplicity 
        # max_nb_res=1024 by default
        # input_crop = ProteinDataset.protein_to_input_crop(protein)
        protein_input = protein.to_torch_input() # a dict of tensors
        protein_batch = ProteinDataset.collate_fn([protein_input])

        output = self.tokenizer_encoder.forward(
            protein_batch["atom_positions"].to(self.device),
            protein_batch["attention_mask"].to(self.device),
            protein_batch["residue_index"].to(self.device),
        )
        structural_tokens = output["idx"][0]
        residue_index = protein_batch["residue_index"][0]
        seqs = protein_input["aatype"]
        return structural_tokens, residue_index, seqs

class WrappedCheapBaseTokenizer():

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = device
        self.pad_token_id = 0 # for pipeline compatibility
    
    def get_num_tokens(self):
        return None

    def get_codebook_embedding(self,):
        return None
    
    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence=False):
        assert use_sequence

        pdb_dict_list = parse_PDB(
            pdb_path, input_chain_list=[chain_id], ca_only=False,
            parse_fn=parse_cif_pdb_biounits
        )
        sequences = [
            pdb_dict_list[0]["seq"]
        ]
        rep, mask = self.pipeline(sequences) # [bsz=1, L, 64], [bsz=1, L]
        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_dict_list[0]["seq"]] # total 20 standard AA in Bio

        return rep.squeeze(0), pdb_dict_list[0]["ridx"], seqs # [L, dim], [L]

class WrappedCheapS1D64Tokenizer(WrappedCheapBaseTokenizer):
    def __init__(self, device):
        self.pipeline = CHEAP_shorten_1_dim_64(return_pipeline=True)
        super().__init__(device)

class WrappedOurPretrainedTokenizer():

    def __init__(self, device: torch.device | str = "cpu", model_cfg=None, pretrained_ckpt_path=None, ckpt_name=None):
        self.device = device
        # load
        self.model = VQVAEModel(model_cfg=model_cfg)
        model_states = torch.load(pretrained_ckpt_path, map_location=self.device)["module"]
        new_model_states = {}
        for k,v in model_states.items():
            assert k.startswith("model.")
            new_model_states[k[6:]] = v
        self.model.load_state_dict(new_model_states)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)
        
        self.seq_tokenizer = EsmSequenceTokenizer()

        self.ckpt_name = ckpt_name

        # reference: https://github.com/evolutionaryscale/esm/blob/39a3a6cb1e722347947dc375e3f8e2ba80ed8b59/esm/utils/constants/esm3.py#L18C12-L18C35
        self.pad_token_id = self.model.quantizer.codebook.weight.shape[0] + 3

    def get_num_tokens(self):
        return self.model.quantizer.codebook.weight.shape[0] + 5
    
    def get_codebook_embedding(self,):
        return self.model.quantizer.codebook.weight
    
    def encode_structure(self, pdb_chain, use_continuous=False, use_sequence=False):
        assert use_sequence
        
        coords, plddt, residue_index = pdb_chain.to_structure_encoder_inputs(self.device) # [1, L, 37, 3], [1, L], [1, L]

        attention_mask = coords[:, :, 0, 0] == torch.inf # [1, L]

        sequence = pdb_chain.sequence
        sequence = sequence.replace(C.MASK_STR_SHORT, "<mask>")
        
        seq_ids = self.seq_tokenizer.encode(sequence, add_special_tokens=False)
        seq_ids = torch.tensor(seq_ids, dtype=torch.int64, device=self.device)
        assert len(seq_ids) == len(coords[0])
        
        input_list = (coords, attention_mask, residue_index, seq_ids, pdb_chain)
        quantized_reprs, quantized_indices, reprs = self.model(input_list, use_as_tokenizer=True) # [1, L, dim], [1, L], [1, L, dim]

        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_chain.sequence] # total 20 standard AA in Bio

        if use_continuous:
            return reprs.squeeze(0), np.array(residue_index.squeeze(0).cpu()), seqs # [L, dim], [L]
        else:
            return quantized_indices.squeeze(0), np.array(residue_index.squeeze(0).cpu()), seqs # [L], [L]


class WrappedESM3Tokenizer():

    
    def __init__(self, device: torch.device | str = "cpu"):
        self.device = device
        
        self.tokenizer_encoder = ESM3_structure_encoder_v0(self.device)
        self.tokenizer_decoder = ESM3_structure_decoder_v0()
        
        # we don't need to attach the whole ESM3 model
        # only get the special token ids from the model
        attached_model = ESM3.from_pretrained("esm3_sm_open_v1")
        self.pad_token_id = attached_model.tokenizers.structure.pad_token_id
        self.bos_token_id = attached_model.tokenizers.structure.bos_token_id
        self.eos_token_id = attached_model.tokenizers.structure.eos_token_id
    
    def get_num_tokens(self):
        # reference from https://github.com/evolutionaryscale/esm/blob/17d48878a9cfad388fdf5ff4d3fe4ea0f0d24839/esm/models/esm3.py#L85
        return self.tokenizer_encoder.codebook.n_codes + 5 # 4096 + 5

    def get_codebook_embedding(self,):
        return self.tokenizer_encoder.codebook.embeddings # (4096, 128)

    def encode_structure(self, pdb_chain, use_continuous=False, use_sequence=False):
        assert use_sequence
        """Reference from https://github.com/evolutionaryscale/esm/blob/95e3c5be8acda407414810ff3aa7d27dbb6e30d3/esm/utils/encoding.py#L60
        """

        coords, plddt, residue_index = pdb_chain.to_structure_encoder_inputs(self.device)
        #coords: (1, L, 37, 3), plddt: (1, L), residue_index: (1, L)
        
        if not use_continuous:
            _, structure_tokens = self.tokenizer_encoder.encode(coords, residue_index=residue_index) # _, (1, L)
        else:
            structure_tokens = self.hijack_continuous_reprs(coords, residue_index=residue_index)
        coords = torch.squeeze(coords, dim=0)  # (L, 37, 3)  # type: ignore
        plddt = torch.squeeze(plddt, dim=0)  # (L,)  # type: ignore
        structure_tokens = torch.squeeze(structure_tokens, dim=0)  # (L,)  # type: ignore

        assert len(pdb_chain.residue_index) == len(pdb_chain.sequence)

        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_chain.sequence] # total 20 standard AA in Bio
        return structure_tokens, np.array(pdb_chain.residue_index), seqs
    
    def hijack_continuous_reprs(self, coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
    ):
        """
        Adapted from https://github.com/evolutionaryscale/esm/blob/39a3a6cb1e722347947dc375e3f8e2ba80ed8b59/esm/models/vqvae.py#L301
        """
        coords = coords[..., :3, :]
        affine, affine_mask = build_affine3d_from_coordinates(coords=coords)

        if attention_mask is None:
            attention_mask = torch.ones_like(affine_mask, dtype=torch.bool)
        attention_mask = attention_mask.bool()

        if sequence_id is None:
            sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)

        z = self.tokenizer_encoder.encode_local_structure(
            coords=coords,
            affine=affine,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            affine_mask=affine_mask,
            residue_index=residue_index,
        )

        z = z.masked_fill(~affine_mask.unsqueeze(2), 0)
        z = self.tokenizer_encoder.pre_vq_proj(z)

        return z

class WrappedMIFTokenizer():

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = device
        self.model, self.mif_collater = mif_load_model_and_alphabet('mif')
        self.pad_token_id = 0 # for pipeline compatibility

        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)
    
    def get_num_tokens(self):
        return None

    def get_codebook_embedding(self,):
        return None

    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence=False):
        assert use_sequence

        pdb_dict_list = parse_PDB(
            pdb_path, input_chain_list=[chain_id], ca_only=False,
            parse_fn=parse_cif_pdb_biounits
        )
        coords = {
            'N': np.array(pdb_dict_list[0][f"coords_chain_{chain_id}"][f"N_chain_{chain_id}"]),
            'CA': np.array(pdb_dict_list[0][f"coords_chain_{chain_id}"][f"CA_chain_{chain_id}"]),
            'C': np.array(pdb_dict_list[0][f"coords_chain_{chain_id}"][f"C_chain_{chain_id}"])
        }
        dist, omega, theta, phi = mif_process_coords(coords)
        batch = [[pdb_dict_list[0][f"seq_chain_{chain_id}"], torch.tensor(dist, dtype=torch.float),
              torch.tensor(omega, dtype=torch.float),
              torch.tensor(theta, dtype=torch.float), torch.tensor(phi, dtype=torch.float)]]
        src, nodes, edges, connections, edge_mask = self.mif_collater(batch)
        rep = self.model(src, nodes, edges, connections, edge_mask)

        assert len(pdb_dict_list[0]["seq"]) == len(rep[0])
        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_dict_list[0]["seq"]] # total 20 standard AA in Bio

        return rep.squeeze(0), pdb_dict_list[0]["ridx"], seqs # [L, dim], [L]

class WrappedProteinMPNNTokenizer():
    HIDDEN_DIM = 128
    NUM_LAYERS = 3
    MODEL_CKPT_BASEURL = "https://github.com/dauparas/ProteinMPNN/raw/refs/heads/main/"
    CKPT_CACHE_DIR = os.path.join(SCRIPT_PATH, 'ProteinMPNN')
    CA_ONLY = False

    def __init__(
            self, 
            device: torch.device | str = "cpu", 
            checkpoint_path: str = "vanilla_model_weights/v_48_020.pt"):
        self.device = device
        # init model and load model weights
        local_checkpoint_path = self._download_model_checkpoint(checkpoint_path)
        self._load_protein_mpnn_model(local_checkpoint_path)

        self.pad_token_id = 0 # for pipeline compatibility

    def _download_model_checkpoint(self, checkpoint_path):
        """Download ProteinMPNN checkpoint from GitHub if not locally cached."""
        ckpt_url = self.MODEL_CKPT_BASEURL + checkpoint_path
        cached_checkpoint_path = os.path.join(self.CKPT_CACHE_DIR, checkpoint_path)
        os.makedirs(os.path.dirname(cached_checkpoint_path), exist_ok=True)
        if not os.path.isfile(cached_checkpoint_path):
            urllib.request.urlretrieve(ckpt_url, cached_checkpoint_path)
        return cached_checkpoint_path

    def _load_protein_mpnn_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = ProteinMPNN(
            ca_only=self.CA_ONLY, 
            num_letters=21, 
            node_features=self.HIDDEN_DIM, 
            edge_features=self.HIDDEN_DIM, 
            hidden_dim=self.HIDDEN_DIM, 
            num_encoder_layers=self.NUM_LAYERS, 
            num_decoder_layers=self.NUM_LAYERS, 
            augment_eps=0.0, 
            k_neighbors=checkpoint['num_edges'])
        model.load_state_dict(checkpoint['model_state_dict'])
        self.model = model.to(self.device)

    def get_num_tokens(self):
        return None

    def get_codebook_embedding(self,):
        return None

    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence=False):
        assert use_sequence
        
        parse_fn = parse_cif_pdb_biounits
        pdb_dict_list = parse_PDB(
            pdb_path, input_chain_list=[chain_id], ca_only=self.CA_ONLY,
            parse_fn=parse_fn
            )

        # this function comes from ProteinMPNN:
        X, _, mask, _, _, chain_encoding_all, _, _, _, _, _, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
            pdb_dict_list, 
            self.device, None, None, None, None, None, None, 
            ca_only=self.CA_ONLY)

        h_V, h_E = self.model.encode(X, mask, residue_idx, chain_encoding_all)
        assert len(pdb_dict_list[0]["ridx"]) == len(h_V[0])
        # h_V: [1, L, hidden_dim]
        # h_E: [1, L, n_edges, hidden_dim] 

        assert len(pdb_dict_list[0]["seq"]) == len(h_V[0])
        seqs = [Bio.PDB.Polypeptide.one_to_index(x) if x != "X" else 20 for x in pdb_dict_list[0]["seq"]] # total 20 standard AA in Bio
        return h_V.squeeze(0), pdb_dict_list[0]["ridx"], seqs # [L, 128], [L]


# pdb structure -> populate batch (from graph) -> encode -> latent 
# exact input/output as other tokenizers.
# input: pdb structure 
# output: z_latent, residue_index, seqs
class WrappedLaProteinaTokenizer():

    def __init__(self, device: torch.device | str = "cpu", pretrained_ckpt_path = None,
                 checkpoint_type: str = "ae", encoder_cfg_path: str = None,
                 force_remove_non_standard_residues: bool = None,
                 encoder_source: Literal["online", "target"] = "online",
                 evaluation_max_length: int = None,
                 evaluation_cache_tag: str = None,
                 evaluation_protocol: str = None):
        self.device = device
        self.pad_token_id = 0 # for pipeline compatibility
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.checkpoint_type = checkpoint_type
        self.encoder_cfg_path = encoder_cfg_path
        self.evaluation_policy = resolve_laproteina_evaluation_policy(
            evaluation_protocol,
            encoder_source=encoder_source,
            evaluation_max_length=evaluation_max_length,
            cache_tag=evaluation_cache_tag,
        )
        self.evaluation_protocol = self.evaluation_policy.name
        self.encoder_source = self.evaluation_policy.encoder_source
        self.encoder_state_prefix = None
        self.evaluation_max_length_override = (
            self.evaluation_policy.evaluation_max_length
        )
        self.apply_checkpoint_training_filters = (
            self.evaluation_policy.apply_checkpoint_training_filters
        )
        self.strict_evaluation_cohort = self.evaluation_policy.strict_cohort
        self._checkpoint_sha256 = None
        self.encoder_cfg_sha256 = None
        if self.evaluation_protocol is not None:
            # Establish byte identity before deserialization. It is rechecked
            # after loading and when the result manifest is minted, so staged
            # files cannot be replaced while an in-memory encoder is attributed
            # to different bytes. Keep this publication overhead opt-in.
            self._checkpoint_sha256 = sha256_file(pretrained_ckpt_path)
            self.encoder_cfg_sha256 = (
                sha256_file(encoder_cfg_path)
                if encoder_cfg_path is not None
                else None
            )
        # Validate once at construction; BaseDataset uses the resulting suffix
        # to force fresh embeddings after an explicit preprocessing revision.
        self.evaluation_cache_tag = self.evaluation_policy.cache_tag
        self.evaluation_cache_tag_suffix = evaluation_cache_tag_suffix(
            self.evaluation_cache_tag
        )

        if checkpoint_type == "ae":
            if self.encoder_source != "online":
                raise ValueError("encoder_source='target' is only valid for JEPA checkpoints")
            model = AutoEncoder.load_from_checkpoint(pretrained_ckpt_path, map_location=self.device, weights_only=False)
            self.encoder = model.encoder
            raw_ckpt = torch.load(pretrained_ckpt_path, map_location="cpu", weights_only=False)
            train_cfg = raw_ckpt.get("hyper_parameters", {}).get("cfg_ae", {})
        elif checkpoint_type == "jepa":
            cfg = json.load(open(encoder_cfg_path))
            ckpt = torch.load(pretrained_ckpt_path, map_location=self.device, weights_only=False)
            encoder_weights, self.encoder_state_prefix = extract_jepa_encoder_state_dict(
                ckpt["state_dict"], self.encoder_source
            )
            # latent_z_dim inferred from the checkpoint's own weights, not assumed
            # from encoder_cfg_path's default (8) -- lets dim-64 (or any future
            # dim) checkpoints load correctly. latent_decoder_mean_n_log_scale.1
            # is the encoder's final [mean, log_scale] projection, so its output
            # width is 2 * latent_z_dim (mirrors Evaluation/linear_probing/checkpoints.py's
            # _infer_latent_z_dim -- same fix, same reasoning).
            latent_z_dim = encoder_weights["latent_decoder_mean_n_log_scale.1.weight"].shape[0] // 2
            print(
                f"[tokenizer] {pretrained_ckpt_path}: encoder_source={self.encoder_source} "
                f"({self.encoder_state_prefix}), latent_z_dim={latent_z_dim}"
            )
            cfg["nn_ae"]["latent_z_dim"] = latent_z_dim
            cfg["nn_ae"]["encoder"]["latent_z_dim"] = latent_z_dim
            cfg["nn_ae"]["decoder"]["latent_z_dim"] = latent_z_dim
            self.encoder = EncoderTransformer(**cfg["nn_ae"])
            self.encoder.load_state_dict(encoder_weights)
            train_cfg = cfg
        else:
            raise ValueError(f"Unknown checkpoint_type: {checkpoint_type!r}, expected 'ae' or 'jepa'")

        if self.evaluation_protocol is not None:
            if sha256_file(pretrained_ckpt_path) != self._checkpoint_sha256:
                raise RuntimeError("Checkpoint bytes changed while the encoder was loading")
            if (
                encoder_cfg_path is not None
                and sha256_file(encoder_cfg_path) != self.encoder_cfg_sha256
            ):
                raise RuntimeError("Encoder config bytes changed while the encoder was loading")

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder = self.encoder.to(self.device)
        self.latent_z_dim = int(
            self.encoder.latent_decoder_mean_n_log_scale[1].weight.shape[0] // 2
        )

        # Read the checkpoint's selector for the historical/default integration.
        # The locked publication protocol records these values but deliberately
        # does not apply them to StructTokenBench's own cohort.
        dataselector = train_cfg.get("dataset", {}).get("datamodule", {}).get("dataselector", {})
        self.checkpoint_training_selector = {
            "min_length": dataselector.get("min_length"),
            "max_length": dataselector.get("max_length"),
            "exclude_ids": sorted(str(x).lower() for x in dataselector.get("exclude_ids", [])),
            "remove_non_standard_residues": dataselector.get(
                "remove_non_standard_residues", False
            ),
            "best_resolution": dataselector.get("best_resolution") or None,
            "worst_resolution": dataselector.get("worst_resolution"),
            "experiment_types": sorted(dataselector.get("experiment_types") or []),
        }
        self.min_length = dataselector.get("min_length")
        self.checkpoint_max_length = dataselector.get("max_length")
        self.max_length = resolve_evaluation_max_length(
            self.checkpoint_max_length, self.evaluation_max_length_override
        )
        if self.evaluation_max_length_override is not None:
            print(
                "[tokenizer] evaluation-only max_length override: "
                f"checkpoint={self.checkpoint_max_length}, effective={self.max_length}; "
                "all other checkpoint dataselector criteria remain enabled"
            )
        self.exclude_ids = set(x.lower() for x in dataselector.get("exclude_ids", []))
        self.remove_non_standard_residues = dataselector.get("remove_non_standard_residues", False)
        if force_remove_non_standard_residues is not None:
            # AE1's dataselector has no such key (defaults False); non-standard residues then
            # crash atom37_to_torsion_angles (x1_sidechain_angles). Override explicitly.
            self.remove_non_standard_residues = force_remove_non_standard_residues
        self.best_resolution = dataselector.get("best_resolution") or None  # 0.0 is a PDBDataSelector no-op too
        self.worst_resolution = dataselector.get("worst_resolution")
        self.experiment_types = set(dataselector.get("experiment_types") or [])

        if not self.apply_checkpoint_training_filters:
            if force_remove_non_standard_residues not in (None, False):
                raise ValueError(
                    f"{self.evaluation_protocol} forbids the LaProteina-only "
                    "force_remove_non_standard_residues filter"
                )
            self.min_length = None
            self.max_length = None
            self.exclude_ids = set()
            self.remove_non_standard_residues = False
            self.best_resolution = None
            self.worst_resolution = None
            self.experiment_types = set()
            print(
                f"[tokenizer] locked evaluation protocol={self.evaluation_protocol}: "
                "online/student encoder, checkpoint-training cohort filters disabled; "
                "StructTokenBench selected-residue filter_length=600 controls the cohort"
            )

        feats_seq = train_cfg.get("nn_ae", {}).get("encoder", {}).get("feats_seq", [])
        self.needs_chain_break_feature = "chain_break_per_res" in feats_seq

        # PDBManager._parse_resolution/_parse_experiment_type called directly
        # against the same PDB-wide index files it downloads itself
        # (resolu.idx, pdb_entry_type.txt), already present locally from this
        # checkpoint's own training setup -- not a full PDBManager() catalog
        # construction, just its own parsing functions.
        import pathlib
        pdb_train_dir = pathlib.Path(os.environ.get("DATA_PATH", "/home/data")) / "pdb_train"
        if self.apply_checkpoint_training_filters:
            import types
            lookup_ns = types.SimpleNamespace(
                root_dir=pdb_train_dir,
                resolution_filename="resolu.idx",
                pdb_entry_type_filename="pdb_entry_type.txt",
            )
            self.resolution_lookup = PDBManager._parse_resolution(lookup_ns)
            self.experiment_type_lookup = PDBManager._parse_experiment_type(lookup_ns)
        else:
            # These training-only indexes are not part of STB's cohort and are
            # intentionally unnecessary in the portable publication mode.
            self.resolution_lookup = {}
            self.experiment_type_lookup = {}

        # PDBManager.parse() sources length AND the non-standard-residue check
        # from pdb_seqres.txt's sequence records, not from parsing the actual
        # structure -- same file, same read_fasta call, replicating its exact
        # record extraction (see PDBManager.parse), so length/exclusion
        # decisions can be made before ever parsing a structure, matching
        # PDBDataSelector's own filter-before-process order. Also more
        # faithful than checking parsed residue count: pdb_seqres length is
        # the full experimental sequence, which LaProteina's own training
        # filtered on -- parsed graphs can have fewer resolved residues due
        # to missing density, a different number entirely.
        self.seqres_lookup = {}  # "{pdb}_{chain}" (lowercase pdb) -> (length, sequence)
        if self.apply_checkpoint_training_filters:
            from graphein.protein.utils import read_fasta
            fasta = read_fasta(str(pdb_train_dir / "pdb_seqres.txt"))
            for header, seq in fasta.items():
                params = header.split()
                pdb_chain_id = params[0]
                length = int(params[2].split(":")[1])
                # PDB IDs are case-insensitive by convention, but chain letters
                # are NOT (large complexes use lowercase to extend past 26
                # chains) -- only normalize the pdb part, matching the lookup in
                # matches_training_distribution_metadata (which does the same).
                pdb_part, _, chain_part = pdb_chain_id.partition("_")
                self.seqres_lookup[f"{pdb_part.lower()}_{chain_part}"] = (length, seq)

    @property
    def checkpoint_sha256(self):
        """Content identity used by publication caches and result manifests."""

        return self._checkpoint_sha256

    #continious latent
    def get_num_tokens(self):
        return None

    def get_codebook_embedding(self,):
        return None

    # mmcif_dir -> constructed PDBLightningDataModule, built once and reused
    # (each worker process gets its own copy)
    _DATAMODULE_CACHE = {}

    @staticmethod
    def _get_native_datamodule(mmcif_dir: str, format: str) -> "PDBLightningDataModule":
        """PDBLightningDataModule pointed at our existing mmcif dir via a
        symlinked raw/ (no copying), so LaProteina's own _load_and_process_pdb
        runs against files already on disk.

        No pre_transforms here: this checkpoint's own saved config has no
        pre_transforms key at all (confirmed directly from the checkpoint) --
        CoordsToNanometers/CenterStructureTransform/ChainBreakPerResidueTransform
        are its 'transforms', applied per PDBDataset.__getitem__ call (after
        the atom reorder), not baked into the one-time processed/ cache. See
        _get_eval_transform, applied explicitly in _parse_graph.

        root is hashed off mmcif_dir itself (not just its parent): Atlas gives
        every (pdb, chain) its own dir, so keying off the parent alone
        collapsed all of them onto one shared raw/ symlink. format comes from
        the caller (real extension on disk) instead of hardcoded "cif".

        The digest is computed off a node-normalized path, not the raw
        resolved one: hopper reads this same physical disk at /mnt/raid
        locally (run_eval.sh's PDB_DATA_ROOT override, for NFS-self-mount
        reliability -- see struct_token_bench_eval_noether.job) while every
        other node sees it at /mnt/hopper-raid over NFS. Hashing the raw
        path made those two aliases of the identical directory collide on
        different digests, so hopper silently built its own native-fallback
        cache (laproteina_native_root_bccb1d8e13) instead of sharing
        laproteina_native_root_5e731aadc7 with everyone else -- confirmed
        100% redundant (zero unique files) when we found it. Actual I/O
        below still goes through `resolved` (the fast local mount on
        whichever node is running); only the string fed to the hash is
        normalized, so every node writes into the same-named directory,
        which -- since /mnt/raid and /mnt/hopper-raid are the same disk --
        is the same directory on disk regardless of which alias got there.
        """
        if mmcif_dir in WrappedLaProteinaTokenizer._DATAMODULE_CACHE:
            return WrappedLaProteinaTokenizer._DATAMODULE_CACHE[mmcif_dir]

        import pathlib
        import hashlib
        import re
        resolved = pathlib.Path(mmcif_dir).resolve()
        canonical = re.sub(r"^/mnt/raid/", "/mnt/hopper-raid/", str(resolved))
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:10]
        root = resolved.parent / f"laproteina_native_root_{digest}"
        root.mkdir(parents=True, exist_ok=True)
        raw_link = root / "raw"
        if not raw_link.exists():
            # concurrent workers can race here; same target either way, so a
            # losing FileExistsError is spurious, not a real failure.
            try:
                raw_link.symlink_to(resolved)
            except FileExistsError:
                pass

        dm = PDBLightningDataModule(data_dir=str(root), format=format, store_het=False, store_bfactor=True)
        WrappedLaProteinaTokenizer._DATAMODULE_CACHE[mmcif_dir] = dm
        return dm

    # LaProteina's own training run already processed ~344k (pdb, chain) pairs
    # into this exact directory -- same _load_and_process_pdb, same raw PDB
    # source. Reuse it read-only before processing anything ourselves.
    _PRETRAINED_DATA_DIR = None

    @staticmethod
    def _get_pretrained_data_dir():
        if WrappedLaProteinaTokenizer._PRETRAINED_DATA_DIR is None:
            import pathlib
            WrappedLaProteinaTokenizer._PRETRAINED_DATA_DIR = (
                pathlib.Path(os.environ.get("DATA_PATH", "/home/data")) / "pdb_train"
            )
        return WrappedLaProteinaTokenizer._PRETRAINED_DATA_DIR

    # This checkpoint's actual 'transforms' (not 'pre_transforms' -- see
    # _get_native_datamodule): applied via PDBDataset(transform=...) at load
    # time, identically whether the underlying processed/ file came from
    # LaProteina's own pretrained cache or was processed here. GlobalRotationTransform
    # (random training augmentation) deliberately excluded -- no single
    # "correct" rotation to apply at eval time.
    _EVAL_TRANSFORM = None

    @staticmethod
    def _get_eval_transform():
        if WrappedLaProteinaTokenizer._EVAL_TRANSFORM is None:
            import torch_geometric.transforms as T
            WrappedLaProteinaTokenizer._EVAL_TRANSFORM = T.Compose([
                CoordsToNanometers(),
                CenterStructureTransform(),
                ChainBreakPerResidueTransform(chain_break_cutoff=WrappedLaProteinaTokenizer.CHAIN_BREAK_CUTOFF_ANGSTROM),
            ])
        return WrappedLaProteinaTokenizer._EVAL_TRANSFORM

    @staticmethod
    def _in_pretrained_cache(pdb_id: str, chain_id: str) -> bool:
        pretrained_dir = WrappedLaProteinaTokenizer._get_pretrained_data_dir()
        return (pretrained_dir / "processed" / f"{pdb_id}_{chain_id}.pt").exists()

    @staticmethod
    def _parse_graph(pdb_path: str, chain_id: str):
        """Calls LaProteina's own data pipeline directly: _load_and_process_pdb
        (parse) then PDBDataset.__getitem__ (PDB->OpenFold atom reorder +
        this checkpoint's own 'transforms') -- not a StructTokenBench
        reimplementation of either.

        Checks LaProteina's own training-time processed/ cache first
        (read-only, same raw PDB source and same _load_and_process_pdb, so a
        hit there is byte-identical to processing it ourselves); only
        processes via our own datamodule for pairs missing from it.
        """
        # base.py's chain_id=" " (Atlas) matches Bio.PDB's blank-chain
        # convention, correct for every other tokenizer. graphein (used below,
        # only by us) parses the same blank column as "" instead -- normalize
        # once, up front, so it's consistent everywhere below.
        if chain_id == " ":
            chain_id = ""
        # full basename incl. chain suffix, not just the 4-char PDB code --
        # correct: Atlas's raw file really is {pdb_id}_{chain_id}.pdb.
        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
        fname = f"{pdb_id}_{chain_id}.pt"
        transform = WrappedLaProteinaTokenizer._get_eval_transform()

        pretrained_dir = WrappedLaProteinaTokenizer._get_pretrained_data_dir()
        if (pretrained_dir / "processed" / fname).exists():
            ds = PDBDataset(
                pdb_codes=[pdb_id], chains=[chain_id], data_dir=str(pretrained_dir),
                transform=transform, format="cif", in_memory=False, file_names=[f"{pdb_id}_{chain_id}"],
            )
            return ds[0]

        mmcif_dir = os.path.dirname(pdb_path)
        # the real extension on disk -- .cif for the general corpus, .pdb for
        # Atlas's per-entry files (see _get_native_datamodule)
        fmt = os.path.splitext(pdb_path)[1].lstrip(".")
        dm = WrappedLaProteinaTokenizer._get_native_datamodule(mmcif_dir, fmt)
        if not (dm.processed_dir / fname).exists():
            result = dm._load_and_process_pdb((0, pdb_id, chain_id))
            if result is None:
                raise RuntimeError(
                    f"PDBLightningDataModule._load_and_process_pdb failed for "
                    f"{pdb_id}/{chain_id} (see its own logged warning for the cause)")

        ds = PDBDataset(
            pdb_codes=[pdb_id],
            chains=[chain_id],
            data_dir=dm.data_dir,
            transform=transform,
            format=dm.format,
            in_memory=False,
            file_names=[f"{pdb_id}_{chain_id}"],
        )
        return ds[0]

    CHAIN_BREAK_CUTOFF_ANGSTROM = 4.0  # ChainBreakPerResidueTransform's own default
    STANDARD_RESIDUES = set(resname_to_idx.keys()) - {"UNK"}

    # PDBManager.remove_non_standard_alphabet_sequences' own alphabet (the 20
    # standard amino acids, 1-letter codes -- matches pdb_seqres.txt's sequence
    # encoding, not the 3-letter codes STANDARD_RESIDUES uses for graph.residues).
    STANDARD_RESIDUES_1LETTER = set("ACDEFGHIKLMNPQRSTVWY")

    @staticmethod
    def matches_training_distribution_metadata(pdb_id, chain_id, seqres_lookup, min_length, max_length,
                                                 exclude_ids, remove_non_standard_residues,
                                                 experiment_types=frozenset(), best_resolution=None,
                                                 worst_resolution=None, resolution_lookup=None,
                                                 experiment_type_lookup=None):
        """Mirrors PDBDataSelector's criteria (values read from the checkpoint's
        own config, not hardcoded) using only cheap metadata (pdb_seqres.txt,
        resolu.idx, pdb_entry_type.txt) -- no structure parsing needed, so this
        can run BEFORE _parse_graph and skip it entirely for excluded
        structures, matching PDBDataSelector's own filter-before-process order.

        Returns True (confidently include), False (confidently exclude), or
        None (pdb_id/chain not found in pdb_seqres.txt -- e.g. AtlasDataset's
        placeholder chain_id=" " never will be -- caller should fall back to
        matches_training_distribution_graph once the structure is parsed,
        rather than treat "unknown to this lookup" as "excluded").
        """
        seqres_entry = seqres_lookup.get(f"{pdb_id.lower()}_{chain_id}")
        if seqres_entry is None:
            return None
        length, sequence = seqres_entry
        if min_length is not None and length < min_length:
            return False
        if max_length is not None and length > max_length:
            return False
        if pdb_id.lower() in exclude_ids:
            return False
        if remove_non_standard_residues and not set(sequence).issubset(
            WrappedLaProteinaTokenizer.STANDARD_RESIDUES_1LETTER
        ):
            return False
        if experiment_types:
            if experiment_type_lookup.get(pdb_id.lower()) not in experiment_types:
                return False
        if worst_resolution is not None or best_resolution is not None:
            resolution = resolution_lookup.get(pdb_id.lower())
            if resolution is None:
                return False
            if worst_resolution is not None and resolution > worst_resolution:
                return False
            if best_resolution is not None and resolution < best_resolution:
                return False
        return True

    @staticmethod
    def matches_training_distribution_graph(graph, pdb_id, min_length, max_length, exclude_ids,
                                              remove_non_standard_residues, experiment_types=frozenset(),
                                              best_resolution=None, worst_resolution=None,
                                              resolution_lookup=None, experiment_type_lookup=None):
        """Same criteria as matches_training_distribution_metadata, but using
        the parsed graph (resolved-residue count, 3-letter residue codes) --
        fallback for pdb_id/chain combos pdb_seqres.txt doesn't cover (e.g.
        AtlasDataset's placeholder chain_id). Only called after parsing, so
        offers no skip-the-parse benefit; use the metadata version first.

        exclude_ids/experiment_type_lookup/resolution_lookup are keyed by the
        bare PDB code (whole-entry properties), but pdb_id is "{pdb}_{chain}"
        for Atlas -- metadata_pdb_id strips the chain back off for just these
        3 lookups (no-op for the general corpus, no underscore there anyway).
        """
        n_residues = graph.coords.shape[0]
        if min_length is not None and n_residues < min_length:
            return False
        if max_length is not None and n_residues > max_length:
            return False
        metadata_pdb_id = pdb_id.split("_")[0].lower()
        if metadata_pdb_id in exclude_ids:
            return False
        if remove_non_standard_residues and any(
            r not in WrappedLaProteinaTokenizer.STANDARD_RESIDUES for r in graph.residues
        ):
            return False
        if experiment_types:
            if experiment_type_lookup.get(metadata_pdb_id) not in experiment_types:
                return False
        if worst_resolution is not None or best_resolution is not None:
            resolution = resolution_lookup.get(metadata_pdb_id)
            if resolution is None:
                return False
            if worst_resolution is not None and resolution > worst_resolution:
                return False
            if best_resolution is not None and resolution < best_resolution:
                return False
        return True

    def _encode_graph(self, graph):
        # converts graph into the padded [b, n, ...] tensor + mask format that Transformer-based EncoderTransformer expects
        # coords_nm, residue_type, coord_mask, and mask_dict are already populated
        # correctly (with the batch dim) by dense_padded_from_data_list
        batch = dense_padded_from_data_list([graph])
        batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0] # FeatureFactory expects this top-level key (see AutoEncoder.encode())
        if self.needs_chain_break_feature:
            # set by ChainBreakPerResidueTransform in _parse_graph; without it
            # FeatureFactory silently defaults to "no break" for every structure
            batch["chain_breaks_per_residue"] = graph.chain_breaks_per_residue.unsqueeze(0)
        batch = batch.to(self.device)

        self.encoder.eval()
        z_latent = self.encoder(batch)["mean"] # deterministic latent, comparable across AE and JEPA checkpoints

        residue_index = graph.residue_pdb_idx
        seqs = graph.residue_type

        return z_latent.squeeze(0), residue_index, seqs

    def encode_structure(self, pdb_path: str, chain_id: str, use_sequence=False):
        cache = getattr(self, "_graph_cache", None)
        graph = cache.pop((pdb_path, chain_id), None) if cache is not None else None
        if graph is None:
            graph = self._parse_graph(pdb_path, chain_id)
        return self._encode_graph(graph)
