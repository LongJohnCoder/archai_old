from enum import Enum
from typing import Mapping, Optional, List
import pathlib
import os
import torch

import yaml

from ..common.common import expdir_abspath


class RunMode(Enum):
    Search = 'search'
    EvalTrain = 'eval_train'
    EvalTest = 'eval_test'

class OpDesc:
    """Op description that is in each edge
    """
    def __init__(self, name:str, params:dict={}, in_len=1,
                 trainables:Optional[Mapping]=None)->None:
        self.name = name
        self.in_len = in_len
        self.params = params # parameters specific to op needed to construct it
        self.trainables = trainables

class EdgeDesc:
    """Edge description between two nodes in the cell
    """
    def __init__(self, op_desc:OpDesc, index:int, input_ids:List[int],
                 run_mode:RunMode)->None:
        assert op_desc.in_len == len(input_ids)
        self.op_desc = op_desc
        self.index = index
        self.input_ids = input_ids
        self.run_mode = run_mode

class NodeDesc:
    def __init__(self, edges:List[EdgeDesc]=[]) -> None:
        self.edges = edges

class AuxTowerDesc:
    def __init__(self, ch_in:int, n_classes:int) -> None:
        self.ch_in = ch_in
        self.n_classes = n_classes

class CellType(Enum):
    Regular = 'regular'
    Reduction  = 'reduction'

class ConvMacroParams:
    """Holds parameters that may be altered by macro architecture"""

    def __init__(self, ch_in:int, ch_out:int, affine:bool) -> None:
        self.ch_in, self.ch_out = ch_in, ch_out
        self.affine = affine

class CellDesc:
    def __init__(self, cell_type:CellType, index:int, nodes:List[NodeDesc],
            s0_op:OpDesc, s1_op:OpDesc,
            out_nodes:int, node_ch_out:int,
            alphas_from:int, run_mode:RunMode)->None:
        assert s0_op.params['conv'].ch_out == s1_op.params['conv'].ch_out
        assert s0_op.params['conv'].ch_out == node_ch_out

        self.cell_type = cell_type
        self.index = index
        self.nodes = nodes
        self.s0_op, self.s1_op = s0_op, s1_op
        self.out_nodes, self.node_ch_out = out_nodes, node_ch_out
        self.run_mode = run_mode
        self.alphas_from = alphas_from # cell index with which we share alphas

        self.cell_ch_out = out_nodes * node_ch_out
        self.conv_params = ConvMacroParams(node_ch_out,
                                           node_ch_out,
                                           run_mode!=RunMode.Search)

class ModelDesc:
    def __init__(self, stem0_op:OpDesc, stem1_op:OpDesc, pool_op:OpDesc,
                 ds_ch:int, n_classes:int, cell_descs:List[CellDesc],
                 aux_tower_descs:List[Optional[AuxTowerDesc]])->None:
        assert len(cell_descs) == len(aux_tower_descs)
        self.stem0_op, self.stem1_op, self.pool_op = stem0_op, stem1_op, pool_op
        self.ds_ch = ds_ch
        self.n_classes = n_classes
        self.cell_descs = cell_descs
        self.aux_tower_descs = aux_tower_descs

    def state_dict(self, clear=False)->dict:
        state_dict = {}
        for ci, cell_desc in enumerate(self.cell_descs):
            sd_cell = state_dict[ci] = {}
            for ni, node in enumerate(cell_desc.nodes):
                sd_node = sd_cell[ni] = {}
                for ei, edge_desc in enumerate(node.edges):
                    sd_node[ei] = edge_desc.op_desc.trainables
                    if clear:
                        edge_desc.op_desc.trainables = None
        return state_dict

    def load_state_dict(self, state_dict:dict)->None:
        for ci, cell_desc in enumerate(self.cell_descs):
            sd_cell = state_dict[ci]
            for ni, node in enumerate(cell_desc.nodes):
                sd_node = sd_cell[ni]
                for ei, edge_desc in enumerate(node.edges):
                    edge_desc.op_desc.trainables = sd_node[ei]

    def save(self, filename:str)->Optional[str]:
        yaml_filepath = expdir_abspath(filename)
        if yaml_filepath:
            if not yaml_filepath.endswith('.yaml'):
                yaml_filepath += '.yaml'

            # clear so PyTorch state is not saved in yaml
            state_dict = self.state_dict(clear=True)
            pt_filepath = ModelDesc._pt_filepath(yaml_filepath)
            torch.save(state_dict, pt_filepath)
            # save yaml
            pathlib.Path(yaml_filepath).write_text(yaml.dump(self))
            # restore state
            self.load_state_dict(state_dict)

        return yaml_filepath

    @staticmethod
    def _pt_filepath(desc_filepath:str)->str:
        return str(pathlib.Path(desc_filepath).with_suffix('.pth'))

    @staticmethod
    def load(yaml_filename:str)->'ModelDesc':
        yaml_filepath = expdir_abspath(yaml_filename)
        if not yaml_filepath or not os.path.exists(yaml_filepath):
            raise RuntimeError("Model description file is not found."
                "Typically this file should be generated from the search."
                "Please copy this file to '{}'".format(yaml_filepath))
        with open(yaml_filepath, 'r') as f:
            model_desc = yaml.load(f, Loader=yaml.Loader)

        # look for pth file that should have pytorch parameters state_dict
        pt_filepath = ModelDesc._pt_filepath(yaml_filepath)
        if os.path.exists(pt_filepath):
            state_dict = torch.load(pt_filepath, map_location=torch.device('cpu'))
            model_desc.load_state_dict(state_dict)
        # else no need to restore weights
        return model_desc
