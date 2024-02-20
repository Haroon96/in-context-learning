from __future__ import annotations

import attr
import time
import numpy as np
import scipy.sparse as sp
from typing import Callable, Optional, Any
from pydantic import BaseModel, Extra

from rank_bm25 import BM25Okapi
from datasets import Dataset

from langchain.prompts.example_selector.base import BaseExampleSelector
from prompts.base import ExampleTemplate
from tools.structure.substructs import get_parser
from selector.base import StructuralSelectorArgs, SelectorUtilsMixin
from selector.greedy import decomposed_coverage_greedy
from tools.track import track

if False:
    def recall_stscores_v1(query_structs, cand_structs):
        n_cands = len(cand_structs)
        all_structs = set([t for s in cand_structs for t in s])
        struct2idx = {s: i for i, s in enumerate(all_structs)}
        cand_stscores = np.zeros((n_cands, len(all_structs)))
        for i, _cand_structs in enumerate(cand_structs):
            for s in query_structs & _cand_structs:
                cand_stscores[i, struct2idx[s]] = 1
        return cand_stscores

    def bm25_stscores(query_structs, cand_structs, score='bm25') -> list[dict[int, float]]:
        n_cands = len(cand_structs)
        all_structs = set([t for s in cand_structs for t in s])
        struct2idx = {s: i for i, s in enumerate(all_structs)}
        struct2idf = {s: np.log(n_cands / (1 + len(s))) for s in all_structs}
        cand_stscores = np.zeros((n_cands, len(all_structs)))
        for i, _cand_structs in enumerate(cand_structs):
            for s in query_structs & _cand_structs:
                cand_stscores[i, struct2idx[s]] = struct2idf[s]
        return cand_stscores

    def recall_stscores_v2(query_structs, cand_structs) -> list[dict[int, float]]:
        n_cands = len(cand_structs)
        cand_stscores = [dict() for _ in range(n_cands)]
        for i, s in enumerate(query_structs):
            for j, _cand_structs in enumerate(cand_structs):
                if s in _cand_structs:
                    # cand_stscores[j, i] = 1
                    cand_stscores[j][i] = 1 / len(query_structs)
        return cand_stscores

    def bm25_stscores_v1(query_structs, cand_structs, bm25: BM25Okapi) -> list[dict[int, float]]:
        n_cands = len(cand_structs)
        # cand_stscores = np.zeros((n_cands, len(query_structs)))
        cand_stscores = [dict() for _ in range(n_cands)]
        for i, s in enumerate(query_structs):
            q_freq = np.array([1 if s in _cand_structs else 0 for _cand_structs in cand_structs])
            doc_len = np.array([len(_cand_structs) for _cand_structs in cand_structs])

            for j, _cand_structs in enumerate(cand_structs):
                doc_len = len(_cand_structs)
                if s in _cand_structs:
                    q_freq = 1
                    q_freq = (bm25.delta + (q_freq * (bm25.k1 + 1)) / (bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl) + q_freq))
                    idf = bm25.idf.get(s) or 0
                    cand_stscores[j][i] = idf * q_freq
        return cand_stscores



def recall_stscores_v3(query_structs, cand_structs):
    cand_stscores = np.array([[1 / len(query_structs) if s in _cand_structs else 0
                               for s in query_structs]
                              for _cand_structs in cand_structs])
    return cand_stscores

def bm25_stscores_v2(query_structs, bm25):
    doc_len = np.array(bm25.doc_len)
    def cand_stscore(q):
        q_freq = np.array([(doc.get(q) or 0) for doc in bm25.doc_freqs])
        score = (bm25.idf.get(q) or 0) * (q_freq * (bm25.k1 + 1) /
                                        (q_freq + bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)))
        return score
    cand_stscores = np.array([cand_stscore(s) for s in query_structs]).T
    return cand_stscores

@attr.s(auto_attribs=True)
class StructuralCoverageSelectorArgs(StructuralSelectorArgs):
    metric: str = 'bm25'
    ordering: str | None = None
    coverage: bool = True
    add_cand_score: bool | None = False
    cand_score_discount: float | None = 1

    def get_name(self) -> str:
        name_parts = [f'{self.subst_size}_subst']
        if self.substruct != 'depst':
            name_parts.append(self.substruct)
        elif self.depparser != 'spacy':
            name_parts.append(self.depparser)
        name_parts.append(self.metric)
        if self.coverage: name_parts.append('coverage')
        if self.ordering: name_parts.append(f'{self.ordering}_order')
        if self.add_cand_score:
            if self.cand_score_discount != 1:
                name_parts.append(f'candscore_by{self.cand_score_discount}')
            else:
                name_parts.append('candscore')
        return '-'.join(name_parts)
Args = StructuralCoverageSelectorArgs

class StructuralCoverageSelector(BaseExampleSelector, SelectorUtilsMixin, BaseModel):
    args: Args
    example_template: ExampleTemplate
    demo_candidates: Dataset

    parser: Any = None
    cand_structs: list[list[Any]] = None
    bm25: Optional[BM25Okapi] = None

    query2idx: dict[str, int] = None
    shot_scores_l: np.ndarray | list[np.ndarray] | None = None
    shot_idxs_l: np.ndarray | list[np.ndarray] | None = None

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_example(self, example: dict[str, str]):
        ...

    @classmethod
    def get_independent_shot_idxs(
        cls, args: Args, query_structs, cand_structs=None, bm25=None,
        return_scores=False
    ):
        if args.metric == 'bm25':
            assert bm25 is not None
            cand_scores = bm25.get_scores(query_structs)
            assert np.allclose(cand_scores, bm25_stscores_v2(query_structs, bm25).sum(axis=-1))
        else:
            assert cand_structs is not None
            cand_stscores = recall_stscores_v3(query_structs, cand_structs)
            # cand_scores = [sum(s.values()) for s in cand_stscores]
            cand_scores = cand_stscores.sum(axis=-1)
        shot_idxs = np.argsort(cand_scores)[-args.n_shots:]
        if return_scores:
            return shot_idxs, cand_scores[shot_idxs]
        else:
            return shot_idxs

    @classmethod
    def get_covering_shot_idxs(
        cls, args: Args, query_structs, cand_structs=None, bm25=None,
        cand_lens=None, max_len=-1, return_scores=False, candidates=None
    ):
        n_shots = args.n_shots
        if args.metric == 'bm25':
            assert bm25 is not None
            cand_stscores = bm25_stscores_v2(query_structs, bm25)
        elif args.metric == 'recall':
            assert cand_structs is not None
            cand_stscores = recall_stscores_v3(query_structs, cand_structs)
        shot_idxs, stats = decomposed_coverage_greedy(
            n_shots, cand_stscores, args.add_cand_score, args.cand_score_discount, candidates, cand_lens=cand_lens, max_len=max_len)
        shot_scores = cand_stscores[shot_idxs].sum(axis=-1)
        if args.ordering:
            order = np.argsort(shot_scores)
            shot_idxs = np.array(shot_idxs)[order]
            shot_scores = shot_scores[order]
        if return_scores:
            return np.array(shot_idxs), shot_scores
        else:
            return np.array(shot_idxs)

    @classmethod
    def get_shot_idxes(
        cls, args: Args, query_structs, cand_structs=None, bm25=None,
        cand_lens=None, max_len=-1, return_scores=False, candidates=None
    ):
        if args.coverage:
            return cls.get_covering_shot_idxs(
                args, query_structs, cand_structs, bm25,
                cand_lens, max_len, return_scores, candidates)
        else:
            return cls.get_independent_shot_idxs(
                args, query_structs, cand_structs, bm25, return_scores)

    def select_examples(self, input_variables: dict[str, str], return_scores=False) -> list[dict]:
        query = self.example_template.format(**input_variables, test=True)

        if query not in self.query2idx:
            query_structs = self.get_substructs([query], self.args, self.parser)[0]
            shot_idxs = self.get_shot_idxes(
                self.args, query_structs, self.cand_structs, self.bm25, return_scores=return_scores, candidates=self.demo_candidates)
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
        args: Args,
        examples: list[dict],
        example_template,
        query_examples: list[dict] = None,
        ex_len_fn: Callable = None,
        max_len: int = -1,
        subtract_gen_len: bool = False,
        progress_bar: bool = True,
        return_time: bool = False,
    ) -> StructuralCoverageSelector:
        examples, query_examples, cand_strings, query_strings, query2idx = cls.common_setup_1(
            examples, query_examples, example_template)
        cand_lens, query_lens, completed_query_lens, max_len = cls.common_setup_2(
            examples, query_examples, ex_len_fn, max_len)

        parser = get_parser(args.depparser) if args.substruct == 'depst' else None
        cand_structs = cls.get_substructs(cand_strings, args, parser, verbose=True)
        beg = time.time()
        query_structs = cls.get_substructs(query_strings, args, parser, verbose=True)
        structs_time = time.time() - beg
        del parser

        bm25 = BM25Okapi(cand_structs) if args.metric == 'bm25' else None

        print('Finding shots...')
        shot_idxs_l, shot_scores_l = [], []
        query_iter = range(len(query_examples))
        if progress_bar: query_iter = track(list(query_iter), description='Finding shots')
        beg = time.time()
        for idx in query_iter:
            if not subtract_gen_len:
                _max_len = max_len - query_lens[idx]
            else:
                _max_len = max_len - completed_query_lens[idx] - 4

            shot_idxs, shot_scores = cls.get_shot_idxes(
                args, query_structs[idx], cand_structs, bm25,
                cand_lens, _max_len, return_scores=True)
            # shot_idxs_l[idx] = shot_idxs
            # shot_scores_l[idx] = shot_scores
            shot_idxs_l.append(shot_idxs)
            shot_scores_l.append(shot_scores)
        print(f'Average number of shots: {np.mean([len(shot_idxs) for shot_idxs in shot_idxs_l])}')
        selector = cls(
            args=args,
            example_template=example_template,
            demo_candidates=examples,
            # parser=parser,
            cand_structs=cand_structs,
            bm25=bm25,
            query2idx=query2idx,
            shot_scores_l=shot_scores_l,
            shot_idxs_l=shot_idxs_l,
        )
        sel_time = structs_time + time.time() - beg
        if return_time:
            return selector, sel_time
        else:
            return selector
