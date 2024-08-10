from __future__ import annotations

import time
import numpy as np
from typing import Callable
from collections import defaultdict
from more_itertools import chunked
from bert_score.utils import get_tokenizer, get_model, model2layers

from selector.base import SelectorUtilsMixin
from selector.bertscore import *
from tools.track import track

class BalancedBertScoreSelector(BertScoreSelector):
   
    @classmethod
    def from_examples(
        cls,
        args: BertScoreSelectorArgs,
        examples: list[dict],
        example_template,
        query_examples: list[dict] = None,
        ex_len_fn: Callable = None,
        max_len: int = -1,
        subtract_gen_len: bool = False,
        device: str = 'cpu',
        progress_bar: bool = True,
        return_time: bool = False,
    ) -> BertScoreSelector:
        import torch
        examples, query_examples, cand_strings, query_strings, query2idx = cls.common_setup_1(
            examples, query_examples, example_template)
        cand_lens, query_lens, completed_query_lens, max_len = cls.common_setup_2(
            examples, query_examples, ex_len_fn, max_len)   
        cand_labels = cls.drop_duplicates(examples, example_template)['label']
     
        n_queries = len(query_examples)
        cand_contexts, query_contexts = [''] * len(examples), [''] * len(query_examples)
        tokenizer = get_tokenizer(args.emb_lm, use_fast=False)
        model = get_model(args.emb_lm, model2layers[args.emb_lm]).eval()
        model = model.to(device)

        from bert_score.utils import get_idf_dict
        if not args.idf:
            idf_dict = defaultdict(lambda: 1.0)
            # set idf for [SEP] and [CLS] to 0
            idf_dict[tokenizer.sep_token_id] = 0
            idf_dict[tokenizer.cls_token_id] = 0
        else:
            idf_dict = get_idf_dict(cand_strings, tokenizer)

        ls = lambda l, idxes: [l[i] for i in idxes]

        # embed examples
        with torch.no_grad():
            cand_embs, cand_idfs = embed(cand_strings, model, tokenizer, idf_dict, device, cand_contexts)
            beg = time.time()
            query_embs, query_idfs = embed(query_strings, model, tokenizer, idf_dict, device, query_contexts)
            embed_time = time.time() - beg
            model = model.to('cpu')
            del model
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
            cand_embs_l, cand_mask_l, cand_idfs_l = [], [], []
            for c_idxes in track(make_chunks([emb.shape[0] for emb in cand_embs], max_prod=1000000)):
                _cand_embs = ls(cand_embs, c_idxes)
                _cand_idfs = ls(cand_idfs, c_idxes)
                _cand_embs, _cand_mask, _cand_idfs = pad_embs_idfs(_cand_embs, _cand_idfs, 'cpu')
                cand_embs_l.append(_cand_embs)
                cand_mask_l.append(_cand_mask)
                cand_idfs_l.append(_cand_idfs)

        # score candidates
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
                        _cand_embs = _cand_embs.to(device)
                        _cand_mask = _cand_mask.to(device)
                        _cand_idfs = _cand_idfs.to(device)
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
            n_shots = args.n_shots
            args.n_shots = len(cand_strings)
            shot_idxs_l, shot_scores_l = [], []
            query_iter = range(n_queries)
            if progress_bar: query_iter = track(list(query_iter), description='Finding shots')
            for idx in query_iter:
                if subtract_gen_len:
                    _max_len = max_len - completed_query_lens[idx] - 4
                else:
                    _max_len = max_len - query_lens[idx]

                _query_embs = torch.from_numpy(query_embs[idx]).to(device)
                _query_idfs = torch.from_numpy(query_idfs[idx]).to(device)
                shot_idxs, shot_scores = cls.get_covering_shot_idxs(
                    args, _query_embs, _query_idfs,
                    cand_embs_l, cand_mask_l, cand_idfs_l,
                    cand_lens, _max_len, return_scores=True
                )
                shot_idxs = shot_idxs[::-1]
                shot_scores = shot_scores[::-1]
                balanced_shot_idxs = []
                balanced_shot_scores = []
                max_qty = round(n_shots / len(set(cand_labels)))
                counter = {}
                for idx, score in zip(shot_idxs, shot_scores):
                    label = cand_labels[idx]
                    if counter.get(label, 0) >= max_qty:
                        continue
                    balanced_shot_idxs.append(idx)
                    balanced_shot_scores.append(score)
                    counter[label] = counter.get(label, 0) + 1
                    if len(balanced_shot_idxs) >= n_shots:
                        break
                shot_idxs_l.append(np.array(balanced_shot_idxs))
                shot_scores_l.append(np.array(balanced_shot_scores))
            print(f'Average number of shots: {np.mean([len(shot_idxs) for shot_idxs in shot_idxs_l])}')

            args.n_shots = n_shots

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
