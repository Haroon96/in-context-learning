from __future__ import annotations

import sys
import attr
import torch
import time
import numpy as np
from pathlib import Path
from typing import Callable
from pydantic import BaseModel, Extra
from more_itertools import chunked
from datasets import Dataset
from transformers import AutoTokenizer

from langchain.prompts.example_selector.base import BaseExampleSelector
from prompts.base import ExampleTemplate
from selector.base import CommonSelectorArgs, SelectorUtilsMixin
from selector.greedy import decomposed_coverage_greedy
from selector.bertscore import *
from tools.track import track
from constants import Dataset as D

from selector.gist import Gister

if Path('../../gisting').exists():
    import sys
    if not '../../' in sys.path:
        sys.path.append('../..')
    from gisting.src.data.collator import get_prompt
else:
    print("Gist repository not found. Experiments involving gisting will fail.")


class AutoCompressor:
    def __init__(self, lm_name: str = 'princeton-nlp/AutoCompressor-2.7b-6k', device: str = 'cuda:0'):
        from AutoCompressors.auto_compressor import AutoCompressorModel
        self.device = device
        # Load a model pre-trained on 6k tokens in 4 compression steps
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name, use_fast=True)
        self.model = AutoCompressorModel.from_pretrained(lm_name).to(device).eval()

    def compress(self, sent):
        sent_tokens = self.tokenizer(sent, return_tensors="pt").input_ids.to(self.device)
        summary_vectors = self.model(sent_tokens, output_softprompt=True).softprompt[0]
        return summary_vectors

def gist_embed(sents, args: GistBertScoreSelectorArgs, compressor=None, device: str = 'cuda:0'):
    print(f'Gisting {len(sents)} prompts ...')

    compressor = compressor or args.get_compressor(device)
    embs_l, idfs_l = [], []
    for e in track(sents):
        if 'AutoCompressor' in args.emb_lm:
            emb = compressor.compress(e)
        else:
            ga = compressor.compress(e) # gist activations
            emb = ga.last_hidden_state[0] if args.layer == -1 else ga.hidden_states[0, :, args.layer]
        emb = emb.to(torch.float).cpu().numpy()
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        embs_l.append(emb)
        idfs_l.append(np.ones(emb.shape[0]))
    return embs_l, idfs_l

@attr.s(auto_attribs=True)
class GistBertScoreSelectorArgs(CommonSelectorArgs):
    dataset: D
    split: str | None = None
    emb_lm: str = '../../exp/run-gist-3tok-flan-t5-small-mtop/run-gist-3tok-flan-t5-small-mtop-run-42'
    lm_dtype: str = 'fp32'
    layer: int = -1
    metric: str = 'recall'
    idf: bool = False
    ordering: str | None = None
    coverage: bool = True
    add_cand_score: bool = False
    cand_score_discount: float | None = 1

    def get_tokenizer(self):
        return self.get_compressor().tokenizer

    def get_compressor(self, device: str = 'cuda:0'):
        if not hasattr(self, '_compressor'):
            if 'AutoCompressor' in self.emb_lm:
                self._compressor = AutoCompressor(self.emb_lm, device=device)
            else:
                self._compressor = Gister(
                    dataset=self.dataset,
                    model_name_or_path=self.emb_lm,
                    num_gist_tokens=self.num_gist_tokens,
                    precision=self.lm_dtype,
                    device=device,
                )
        return self._compressor
    @property
    def num_gist_tokens(self):
        if 'AutoCompressor' in self.emb_lm:
            return 50
        else:
            n_toks = [int(x[:-3]) for p in self.emb_lm.split('/') for x in p.split('-') if x.endswith('tok')]
            assert len(set(n_toks)) == 1
            return n_toks[0]

    def get_name(self) -> str:
        emb_lm_parts = self.emb_lm.split('/')
        if emb_lm_parts[-1] == '':
            emb_lm_parts = emb_lm_parts[:-1]
        if 'checkpoint' in emb_lm_parts[-1]:
            emb_lm_name = '-'.join(emb_lm_parts[-2:])
        else:
            emb_lm_name = emb_lm_parts[-1]
        name_parts = [emb_lm_name]
        if self.layer != -1: name_parts.append(f'layer{self.layer}')
        name_parts.append(self.metric)
        if self.idf: name_parts.append('idf')
        if self.coverage: name_parts.append('coverage')
        if self.ordering: name_parts.append(f'{self.ordering}_order')
        return '-'.join(name_parts)

    def get_friendly_name(self):
        name_parts = [self.emb_lm.split('/')[-1], self.metric]
        if self.idf: name_parts.append('idf')
        if self.coverage: name_parts.append('coverage')
        if self.ordering: name_parts.append(f'{self.ordering}_order')
        if self.add_cand_score:
            if self.cand_score_discount != 1:
                name_parts.append(f'candscore_by{self.cand_score_discount}')
            else:
                name_parts.append('candscore')
        return '+'.join(name_parts)

    def ex_to_string(self, instance) -> str:
        gist_str = "<GIST>" * self.num_gist_tokens
        prompt = get_prompt(self.dataset, self.split, instance, gist_str)
        return prompt

class GistBertScoreSelector(BaseExampleSelector, SelectorUtilsMixin, BaseModel):
    args: GistBertScoreSelectorArgs
    example_template: ExampleTemplate
    demo_candidates: Dataset
    query2idx: dict[str, int]
    shot_scores_l: np.ndarray | list[np.ndarray] | None = None
    shot_idxs_l: np.ndarray | list[np.ndarray] | None = None

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True


    def add_example(self, example: dict[str, str]):
        ...

    @staticmethod
    def get_cosine_covering_shot_idxs(
        args: GistBertScoreSelectorArgs,
        query_emb, cand_embs, cand_lens=None, max_len=-1, return_scores=False,
    ):
        n_shots = args.n_shots
        cand_dimscores = np.einsum('d,cd->cd', query_emb, cand_embs)
        shot_idxs, _ = decomposed_coverage_greedy(
            n_shots, cand_dimscores, cand_lens=cand_lens, max_len=max_len)
        shot_scores = cand_dimscores[shot_idxs].sum(axis=-1)
        if args.ordering:
            order = np.argsort(shot_scores)
            shot_idxs = np.array(shot_idxs)[order]
            shot_scores = shot_scores[order]
        if return_scores:
            return np.array(shot_idxs), shot_scores
        else:
            return np.array(shot_idxs)

    @staticmethod
    def get_bsr_covering_shot_idxs(
        args: GistBertScoreSelectorArgs,
        query_embs, query_idfs, cand_embs_l, cand_mask_l, cand_idfs_l,
        cand_lens=None, max_len=-1, return_scores=False,
    ):
        assert args.metric == 'recall'
        with torch.no_grad():
            n_shots = args.n_shots
            sims_l = []
            token_recalls_l = []
            for _cand_embs, _cand_mask, _cand_idfs in zip(cand_embs_l, cand_mask_l, cand_idfs_l):
                _cand_embs = _cand_embs.to(query_embs.device)
                _cand_mask = _cand_mask.to(query_embs.device)
                _sims = compute_sims(query_embs, None, _cand_embs, _cand_mask)
                _token_recalls = sims_to_token_recalls(_sims, query_idfs).cpu().numpy()
                sims_l.append(_sims)
                token_recalls_l.append(_token_recalls)
            token_recalls = np.concatenate(token_recalls_l, axis=0)
            shot_idxs, stats = decomposed_coverage_greedy(
                n_shots, token_recalls, args.add_cand_score, args.cand_score_discount,
                cand_lens=cand_lens, max_len=max_len)
            shot_scores = token_recalls[shot_idxs].sum(axis=-1)
            if args.ordering:
                if args.ordering != 'recall':
                    raise NotImplementedError
                    def lls(ll, idxes, batch_size):
                        sel = []
                        for i in idxes:
                            sel.append(ll[i//batch_size][i%batch_size])
                    shot_scores = sims_to_bertscore(
                        sims[shot_idxs], query_idfs, cand_idfs[shot_idxs],
                        metric=args.ordering)
                order = shot_scores.argsort()
                shot_idxs = np.array(shot_idxs)[order]
                shot_scores = shot_scores[order]
        if return_scores:
            return shot_idxs, shot_scores
        else:
            return shot_idxs

    @staticmethod
    def get_covering_shot_idxs(
        args: GistBertScoreSelectorArgs,
        query_embs, query_idfs, cand_embs_l, cand_mask_l, cand_idfs_l,
        cand_lens=None, max_len=-1, return_scores=False,
    ):
        if cand_embs_l[0].shape[1] > 1:
            return GistBertScoreSelector.get_bsr_covering_shot_idxs(
                args, query_embs, query_idfs, cand_embs_l, cand_mask_l, cand_idfs_l,
                cand_lens, max_len, return_scores)
        else:
            query_emb = query_embs.cpu().numpy()[0]
            cand_embs = torch.cat(cand_embs_l, axis=0)[:, 0, :].cpu().numpy()
            return GistBertScoreSelector.get_cosine_covering_shot_idxs(
                args, query_emb, cand_embs, cand_lens, max_len, return_scores)

    def select_examples(self, input_variables: dict[str, str], return_scores=False) -> list[dict]:
        # query = self.args.ex_to_string(input_variables, self.example_template)
        query = self.args.ex_to_string(input_variables)
        if query not in self.query2idx:
            if self.args.coverage:
                shot_idxs = self.get_covering_shot_idxs(
                    self.args,
                    self.query_embs[self.query2idx[query]],
                    self.query_mask[self.query2idx[query]],
                    self.query_idfs[self.query2idx[query]],
                    self.cand_embs, self.cand_mask, self.cand_idfs)
            else:
                scores = self.scores[self.query2idx[query]]
                shot_idxs = scores.argsort()[-self.args.n_shots:]
                shot_scores = scores[shot_idxs]
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
        args: GistBertScoreSelectorArgs,
        examples: list[dict],
        example_template,
        query_examples: list[dict] = None,
        ex_len_fn: Callable = None,
        max_len: int = -1,
        subtract_gen_len: bool = False,
        device: str = 'cpu',
        progress_bar: bool = True,
        return_time: bool = False,
    ) -> GistBertScoreSelector:
        import torch
        # examples, query_examples, cand_strings, query_strings, query2idx = cls.common_setup_1(
        #     examples, query_examples, example_template, args.ex_to_string)
        examples = cls.drop_duplicates(examples, example_template)
        query_examples = query_examples or []
        # cand_strings = [args.ex_to_string(ex, example_template) for ex in examples]
        # query_strings = [args.ex_to_string(ex, example_template) for ex in query_examples]
        cand_strings = [args.ex_to_string(ex) for ex in examples]
        query_strings = [args.ex_to_string(ex) for ex in query_examples]
        query2idx = {query: i for i, query in enumerate(query_strings)}

        cand_lens, query_lens, completed_query_lens, max_len = cls.common_setup_2(
            examples, query_examples, ex_len_fn, max_len)

        n_queries = len(query_examples)

        ls = lambda l, idxes: [l[i] for i in idxes]
        with torch.no_grad():
            compressor = args.get_compressor(device=device)
            cand_embs, cand_idfs = gist_embed(cand_strings, args, compressor, device)
            beg = time.time()
            query_embs, query_idfs = gist_embed(query_strings, args, compressor, device)
            embed_time = time.time() - beg
            del compressor
            torch.cuda.empty_cache()

            def make_chunks(lens, max_prod):
                i = 0
                chunks = [[]]
                while i < len(lens):
                    cand_chunk = chunks[-1] + [i]
                    if max([lens[j] for j in cand_chunk]) * len(cand_chunk) > max_prod:
                        chunks.append([])
                        continue
                    chunks[-1].append(i)
                    i += 1
                return chunks
            idx_chunks = make_chunks([emb.shape[0] for emb in cand_embs], max_prod=1000000)
            cand_embs_l, cand_mask_l, cand_idfs_l = [], [], []
            for c_idxes in track(idx_chunks):
                _cand_embs = ls(cand_embs, c_idxes)
                _cand_idfs = ls(cand_idfs, c_idxes)
                _cand_embs, _cand_mask, _cand_idfs = pad_embs_idfs(_cand_embs, _cand_idfs, device)
                cand_embs_l.append(_cand_embs)
                cand_mask_l.append(_cand_mask)
                cand_idfs_l.append(_cand_idfs)

        beg = time.time()
        if not args.coverage:
            with torch.no_grad():
                def get_batch_scores(q_idxes):
                    _query_embs = ls(query_embs, q_idxes)
                    _query_idfs = ls(query_idfs, q_idxes)
                    # TODO: try padding queries a priori
                    _query_embs, _query_mask, _query_idfs = pad_embs_idfs(_query_embs, _query_idfs, device)
                    scores_l = []
                    for _cand_embs, _cand_mask, _cand_idfs in zip(cand_embs_l, cand_mask_l, cand_idfs_l):
                        # _cand_embs = _cand_embs.to(device)
                        # _cand_mask = _cand_mask.to(device)
                        # _cand_idfs = _cand_idfs.to(device)
                        sims = compute_sims(_query_embs, _query_mask, _cand_embs, _cand_mask)
                        scores = sims_to_bertscore(
                            sims, _query_idfs, _cand_idfs, metric=args.metric)
                        scores_l.append(scores)
                        # torch.cuda.empty_cache()
                    scores = torch.cat(scores_l, axis=1)
                    return scores
                batch_size, batch_scores = 1, []
                query_iter = chunked(range(n_queries), batch_size)
                if progress_bar: query_iter = track(list(query_iter), description='Finding shots')
                for q_idxes in query_iter:
                    batch_scores.append(get_batch_scores(q_idxes).cpu())
                scores = torch.cat(batch_scores, axis=0)
                shot_scores_l = scores.sort(axis=-1).values[:, -args.n_shots:].numpy()
                shot_idxs_l = scores.argsort(axis=-1)[:, -args.n_shots:].numpy()
            torch.cuda.empty_cache()
            selector = cls(
               args=args,
                example_template=example_template,
                demo_candidates=examples,
                query2idx=query2idx,
                # scores=scores,
                shot_scores_l=shot_scores_l,
                shot_idxs_l=shot_idxs_l,
            )
        else:
            shot_idxs_l, shot_scores_l = [], []
            query_iter = range(n_queries)
            if progress_bar: query_iter = track(list(query_iter), description='Finding shots')
            for idx in query_iter:
                if not subtract_gen_len:
                    _max_len = max_len - query_lens[idx]
                else:
                    _max_len = max_len - completed_query_lens[idx] - 4
                _query_embs = torch.from_numpy(query_embs[idx]).to(device)
                _query_idfs = torch.from_numpy(query_idfs[idx]).to(device)
                shot_idxs, shot_scores = cls.get_covering_shot_idxs(
                    args, _query_embs, _query_idfs, cand_embs_l, cand_mask_l, cand_idfs_l, cand_lens, _max_len, return_scores=True)
                shot_idxs_l.append(np.array(shot_idxs))
                shot_scores_l.append(np.array(shot_scores))
            print(f'Average number of shots: {np.mean([len(shot_idxs) for shot_idxs in shot_idxs_l])}')

            torch.cuda.empty_cache()
            selector = cls(
                args=args,
                example_template=example_template,
                demo_candidates=examples,
                query2idx=query2idx,
                shot_scores_l=shot_scores_l,
                shot_idxs_l=shot_idxs_l
            )
        sel_time = embed_time + time.time() - beg
        if return_time:
            return selector, sel_time
        else:
            return selector
