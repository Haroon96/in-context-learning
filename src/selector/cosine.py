from __future__ import annotations

import attr
import time
import numpy as np
from typing import Callable
from pydantic import BaseModel, Extra
from datasets import Dataset

from langchain.prompts.example_selector.base import BaseExampleSelector
from prompts.base import ExampleTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from selector.base import CommonSelectorArgs, SelectorUtilsMixin
from selector.greedy import decomposed_coverage_greedy
from tools.track import track

def best_comb_emb_coverage_v1(k, target_emb, cand_embs):
    init_skip_idxs = set()
    diffs = np.abs(cand_embs - target_emb[None, :])
    curr_comb = []
    curr_diff = np.array([np.inf for _ in range(target_emb.shape[0])])
    curr_obj = np.inf
    stats = dict(n_reset=0)
    while len(curr_comb) < k:
        best_idx = None
        best_diff = curr_diff
        best_obj = np.inf
        for idx, diff in enumerate(diffs):
            if idx in skip_idxs:
                continue
            cand_diff = np.minimum(diff, curr_diff)
            cand_obj = np.sum(cand_diff)
            if np.allclose(cand_obj, curr_obj):
                skip_idxs.add(idx)
            elif best_idx is None or cand_obj < best_obj:
                best_idx, best_diff, best_obj = idx, cand_diff, cand_obj
        if best_idx is None:
            skip_idxs = init_skip_idxs | set(curr_comb)
            stats['n_reset'] += 1
        else:
            curr_comb.append(best_idx)
            skip_idxs.add(best_idx)
            curr_diff, curr_obj = best_diff, best_obj
    return curr_comb, stats | dict(score=curr_obj)

@attr.s(auto_attribs=True)
class CosineCoverageSelectorArgs(CommonSelectorArgs):
    emb_lm: str = 'sentence-transformers/all-mpnet-base-v2'
    coverage: bool = True
    reorder: bool = False

    def get_name(self):
        name_parts = [self.emb_lm.split('/')[-1]]
        if self.coverage: name_parts.append('coverage')
        if self.reorder: name_parts.append(f'reorder')
        return '-'.join(name_parts)


class CosineCoverageSelector(BaseExampleSelector, SelectorUtilsMixin, BaseModel):
    args: CosineCoverageSelectorArgs
    example_template: ExampleTemplate
    demo_candidates: Dataset

    embedding: Embeddings = None
    cand_embs: np.ndarray = None

    query2idx: dict[str, int] = None
    shot_scores_l: np.ndarray | list[np.ndarray] | None = None
    shot_idxs_l: np.ndarray | list[np.ndarray] | None = None

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_example(self, example: dict[str, str]):
        ...

    @staticmethod
    def get_independent_shot_idxs(args, query_emb, cand_embs, return_scores=False):
        cand_scores = np.einsum('d,cd->c', query_emb, cand_embs)
        shot_idxs = np.argsort(cand_scores)[-args.n_shots:]
        if return_scores:
            return shot_idxs, cand_scores[shot_idxs]
        else:
            return shot_idxs

    @staticmethod
    def get_covering_shot_idxs(
        args, query_emb, cand_embs, cand_lens=None, max_len=-1, return_scores=False
    ):
        n_shots = args.n_shots
        cand_dimscores = np.einsum('d,cd->cd', query_emb, cand_embs)
        shot_idxs, _ = decomposed_coverage_greedy(
            n_shots, cand_dimscores, cand_lens=cand_lens, max_len=max_len)
        shot_scores = cand_dimscores[shot_idxs].sum(axis=-1)
        if args.reorder:
            order = np.argsort(shot_scores)
            shot_idxs = np.array(shot_idxs)[order]
            shot_scores = shot_scores[order]
        if return_scores:
            return np.array(shot_idxs), shot_scores
        else:
            return np.array(shot_idxs)

    @classmethod
    def get_shot_idxs(
        cls, args, query_emb, cand_embs, cand_lens=None, max_len=-1, return_scores=False
    ):
        if args.coverage:
            return cls.get_covering_shot_idxs(
                args, query_emb, cand_embs, cand_lens, max_len, return_scores=return_scores)
        else:
            return cls.get_independent_shot_idxs(
                args, query_emb, cand_embs, return_scores=return_scores)

    def select_examples(self, input_variables: dict[str, str], return_scores=False) -> list[dict]:
        query = self.example_template.format(**input_variables, test=True)
        if query not in self.query2idx:
            query_emb = np.array(self.embedding.embed_query(query))
            shot_idxs = self.get_shot_idxs(
                self.args, query_emb, self.cand_embs, return_scores=return_scores)
            if return_scores:
                shot_idxs, shot_scores = shot_idxs
        else:
            shot_idxs = self.shot_idxs_l[self.query2idx[query]]
            shot_scores = self.shot_scores_l[self.query2idx[query]]
        if return_scores:
            return self.demo_candidates.select(shot_idxs), shot_scores
        else:
            return self.demo_candidates.select(shot_idxs)

    @classmethod
    def from_examples(
        cls,
        args: CosineCoverageSelectorArgs,
        examples: list[dict],
        example_template,
        query_examples: list[dict],
        ex_len_fn: Callable = None,
        max_len: int = -1,
        subtract_gen_len: bool = False,
        device: str = 'cpu',
        progress_bar: bool = True,
        return_time: bool = False,
    ) -> CosineCoverageSelector:
        examples, query_examples, cand_strings, query_strings, query2idx = cls.common_setup_1(
            examples, query_examples, example_template)
        cand_lens, query_lens, completed_query_lens, max_len = cls.common_setup_2(
            examples, query_examples, ex_len_fn, max_len)

        embedding = HuggingFaceEmbeddings(model_name=args.emb_lm, device=device)
        cand_embs = np.array(embedding.embed_documents(cand_strings))
        beg = time.time()
        query_embs = np.array(embedding.embed_documents(query_strings))
        embed_time = time.time() - beg

        shot_idxs_l, shot_scores_l = [], []
        query_iter = range(len(query_examples))
        if progress_bar: query_iter = track(list(query_iter), description='Finding shots')
        beg = time.time()
        for idx in query_iter:
            if not subtract_gen_len:
                _max_len = max_len - query_lens[idx]
            else:
                _max_len = max_len - completed_query_lens[idx] - 4
            shot_idxs, shot_scores = cls.get_shot_idxs(
                args, query_embs[idx], cand_embs, cand_lens, _max_len, True)
            shot_idxs_l.append(shot_idxs)
            shot_scores_l.append(shot_scores)
        selector = cls(
            args=args,
            example_template=example_template,
            demo_candidates=examples,
            # embedding=embedding,
            # cand_embs=cand_embs,
            query2idx=query2idx,
            shot_scores_l=shot_scores_l,
            shot_idxs_l=shot_idxs_l,
        )
        sel_time = embed_time + time.time() - beg
        if return_time:
            return selector, sel_time
        else:
            return selector
