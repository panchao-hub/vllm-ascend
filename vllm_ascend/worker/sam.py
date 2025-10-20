from typing import Optional, Union, List
from dataclasses import dataclass
from copy import deepcopy
import torch
import numpy as np
from numba import jit
from vllm.config import VllmConfig

class SAM:
    @dataclass
    class SAMState:
        next: dict[int, int]
        link: int
        length: int
        min_endpos: int

    def __init__(self, n_predicts: int = 3, device: str = "cpu"):
        self.n_predicts = n_predicts
        self.states: List[SAM.SAMState] = [SAM.SAMState(next={}, link=-1, length=0, min_endpos=0)]
        self.input_ids: List[int] = [-1]
        self.last = 0
        self.max_length = 0
        self.device = device

        # params need to be reset for each query
        self.cur_index = 0
        self.cur_length = 0

    def reset(self):
        self.states: List[SAM.SAMState] = [SAM.SAMState(next={}, link=-1, length=0, min_endpos=0)]
        self.input_ids: List[int] = [-1]
        self.last = 0
        self.max_length = 0
        self.cur_index = 0
        self.cur_length = 0
    
    def expand_state(self, state: SAMState):
        new_index = len(self.states)
        self.states.append(state)
        return new_index
    
    def add_state(self, token: int):
        self.max_length += 1
        cur = self.expand_state(
            SAM.SAMState(
                next={}, link=-1,
                length=self.max_length,
                min_endpos=self.max_length
            )
        )
        p = self.last
        while p != -1 and token not in self.states[p].next:
            self.states[p].next[token] = cur
            p = self.states[p].link
        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].next[token]
            if self.states[p].length + 1 == self.states[q].length:
                self.states[cur].link = q
            else:
                clone = self.expand_state(deepcopy(self.states[q]))
                self.states[clone].length = self.states[p].length + 1
                while p != -1 and self.states[p].next[token] == q:
                    self.states[p].next[token] = clone
                    p = self.states[p].link
                self.states[q].link = self.states[cur].link = clone
        self.last = cur
    
    def transfer_state(self, index: int, length: int, token: int):
        while index != 0 and token not in self.states[index].next:
            index = self.states[index].link
            length = self.states[index].length
        if token in self.states[index].next:
            index = self.states[index].next[token]
            length += 1
        else:
            index = length = 0
        return index, length

    def transfer_cur_state(self, token: int):
        self.cur_index, self.cur_length = self.transfer_state(self.cur_index, self.cur_length, token)
    
    def add_tokens(self, tokens: Union[List[int], np.ndarray]):
        for token in tokens:
            self.transfer_cur_state(token)
            self.add_state(token)
        self.input_ids.extend(tokens)
    
    def transfer_tokens(self, tokens: Union[List[int], np.ndarray]):
        for token in tokens:
            self.transfer_cur_state(token)
    
    def lookup(self, token: int):
        index, length = self.transfer_state(self.cur_index, self.cur_length, token)
        return index, length

    def to_anc(self, index: int):
        if index != 0:
            length_to_end = self.max_length - self.states[index].min_endpos
            while self.states[index].link != 0 and self.n_predicts > length_to_end:
                index = self.states[index].link
                length_to_end = self.max_length - self.states[index].min_endpos
        return index
    
    def gen_draft(self, index: int, start_token: int):
        index = self.to_anc(index)
        endpos = self.states[index].min_endpos
        pred_ids = []
        if endpos != 0:
            pred_ids = self.input_ids[endpos+1 : endpos+self.n_predicts+1]
        return pred_ids
    
    def propose(self, context_token_ids: np.ndarray):
        self.reset()
        self.add_tokens(context_token_ids[:-1])
        query_token = context_token_ids[-1]
        index_dyn, _ = self.lookup(query_token)
        seq = self.gen_draft(index_dyn, query_token)
        return np.array(seq)
    
class SAMProposer:
    def __init__(self, vllm_config):
        self.n_predicts = vllm_config.speculative_config.num_speculative_tokens
        self.all_proposers: dict[int, SAM] = {}
    
    def propose(self,
                request_id: int,
                old_token_ids,
                new_token_ids,
                num_sampled_ids):
        if self.all_proposers.get(request_id, None) is None:
            self.all_proposers[request_id] = SAM(n_predicts=self.n_predicts)
            self.all_proposers[request_id].add_tokens(old_token_ids)
        if num_sampled_ids>1:
            self.all_proposers[request_id].add_tokens(new_token_ids[:-1])
        query_token = new_token_ids[-1]
        index_dyn, _ = self.all_proposers[request_id].lookup(query_token)
        seq = self.all_proposers[request_id].gen_draft(index_dyn, query_token)
        self.all_proposers[request_id].add_tokens(new_token_ids[-1:])
        return np.array(seq)
    
    def load_model(self, *args, **kwargs):
        pass
