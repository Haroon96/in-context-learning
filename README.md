# Example Selection for In-Context Learning

Framework for convenient In-context Learning (ICL) evaluations for different datasets, LLMs, and example selection methods. In particular, it is used to evaluate the in-context example selection methods proposed in the following papers:

- [Coverage-based Example Selection for In-Context Learning](https://arxiv.org/abs/2305.14907) - BERTScore-Recall (BSR), Set-BSR. Originally implemented in the [icl-coverage](https://github.com/Shivanshu-Gupta/icl-coverage) repository.
- [GistScore: Learning Better Representations for In-Context Example Selection with Gist Bottlenecks](https://arxiv.org/abs/2311.09606) - GistScore, Set-GistScore. See also the [gist-icl](https://github.com/Shivanshu-Gupta/gist-icl) repository.

Apart from the above, it also supports the following selectors: Random, BM25, SentenceBERT (Cosine). See [`constants`](src/constants.py) for a list of datasets and LLMs that have currently been evaluated.

## Setup

1. Download datasets unavailable in HuggingFace from [here][icl-datasets] and store them in `data/`.
2. Install requirements: `pip install -r requirements.txt`
3. Some third-party repos:
   1. `qdecomp_with_dependency_graphs`: required for DROP dataset.

      ```bash
      mkdir icl-demo-selection/src/third_party
      git clone git@github.com:matanhasson/qdecomp_with_dependency_graphs.git icl-demo-selection/src/third_party/
      ```
4. [Optional] LLM-specific setup:
   1.  For experiments with LlaMA models, set the path to the directory containing downloaded LlaMA weights in `langchain.llms.huggingface.get_model_cache_dir`.
   2. Experiments with some LLMs may require setting up HuggingFace auth token by running `huggingface-cli login`.
   3. Store the OpenAI key in `openai_keys.txt` in the root directory.

## Organization

The repository is organized as follows:

```plaintext
icl
├── data             (local datasets -- download from https://1drv.ms/u/s!AqJNiE6C-nXuoawBxh-3rfUsSf4-8A?e=3o1YDK)
├── results          (icl experiment results and logs)
├── src              (relevant source files described below)
└── openai_keys.txt    (any openai keys, one per line)
```

Important source files include:

- [`src/params.py`](src/params.py) defines experiment parameters
  - [`src/data_params.py`](src/data_params.py) defines the parameters for each dataset
- [`src/constants.py`](src/constants.py) defines some useful enums and constants
- [`src/driver.py`](src/driver.py) is the main file to run a single ICL experiment. Instead of directly running this file, use [`src/experiments.py`](src/experiments.py) -- it takes care of many default parameters and makes it easy to run multiple experiments.
  - [`src/eval.py`](src/eval.py) used within [`src/driver.py`](src/driver.py) to run the ICL evaluation
- [`src/experiments.py`](src/experiments.py) contains the code to run experiments, track experiment statuses and aggregate results. Instead, of directly it dumps the parameters for all the experiments to a file that is then used by [`src/run.py`](src/run.py). Run `python experiments.py --help` to see help.
  - [`src/exp_utils.py`](src/exp_utils.py) defines various default arguments
- [`src/run.py`](src/run.py) used to run one or more experiments sequentially or in parallel on one or more GPUs. It is the main file to run experiments.
- [`src/selector/`](src/selector/) contains the implementations for the various selectors
- [`src/prompts/`](src/prompts/) contains templates for single examples and few-shot prompts

## Workflows

### Running ICL Evaluations

[`src/experiments.py`](src/experiments.py) and [`src/run.py`](src/run.py) are the main files to run ICL evaluations. The following are some example workflows:

1. Generate the parameters for 8-shot ICL with all the datasets, Neo and LLaMA-7B LLMs, with LLMs selected using Cosine, BERTscore, and GistScore selectors, and dump them to `params/all.jsonl`. See `experiments.main` for detailed usage.

   ```bash
   python experiments.py --label "test" --seeds 0 \
   --datasets "QNLI;MNLI;RTE;SST2;YELP;MRPC;QQP;PAWS;COPA;PIQA;WINOGRANDE;WSC;CMSQA;COLA;COMMONGEN;E2ENLG;DART;SST5;AGNEWS;AESLC;SMCALFLOW_CS;BREAK;MTOP;COGS" \
   --selectors "cosine;bertscore;gist_bertscore" \
   --lms "llama-7B" \
   --n-shots 8 --baselines-exp \
   --paramsfile "params/all.jsonl" --run \
   --no-collate-results \
   --preview "logfiles"
   ```

2. Run the experiments in `params/all.jsonl` parallelly on gpus 0 and 1.

   ```bash
   python run.py --paramsfile "params/all.jsonl" --gpus "0,1"
   ```

NOTE: To run ICL evaluations with GistScore, see the [gist-icl](https://github.com/Shivanshu-Gupta/gist-icl) repo.

### Adding a new dataset

1. Update `constants.Dataset` and `constants.category2datasets`.
2. Add a parameters class for it in [`src/data_params.py`](src/data_params.py) similar to all the other datasets.
    1. If it requires a new metric, add it to [`prompts/base.py`](prompts/base.py)
    2. Test it using `data_params.test_dataset` or `data_params.test`.
3. For ICL evaluation, some of these might also be necessary (though rare):
   1. If it requires any default arguments, add them to `exp_utils.dataset_args_d`
   2. It has more than one `split`s, add them to `exp_utils.ds2splits`. If it has more than one `test_split`s, those will be recorded in `exp_utils.dataset_args_d` (similar to COGS).
   3. If it requires a new metric, add the name for that metric to the `metric_cols` lists in `experiments.make_tables`.

## Miscellaneous Tips

There are two different types of command lines in this repository:
1. [Typer](https://typer.tiangolo.com/) - this one is used for non-nested parameterization. Allows multiple commands in a single script run as `python <script> <command> <arguments>`. The `<command>` only needs to be specified if there are more than one commands (eg. `src/data_params.py`). The `<arguments>` are specified a bit differently so try running with `--help` to see them.
   1. `src/experiments.py`:
   2. `src/run.py`
   3. `src/data_params.py`
2. [Hydra](hydra.cc/) - this one is used for more nested parameterization.
   1. `src/driver.py`: parameters defined in (`src/params.py:AllParams`)

[icl-datasets]: https://1drv.ms/u/s!AqJNiE6C-nXuoawBxh-3rfUsSf4-8A?e=3o1YDK

## Citation

If you found this repository useful, please cite the following papers:

```bibtex
@inproceedings{gupta-etal-2023-coverage,
    title = "Coverage-based Example Selection for In-Context Learning",
    author = "Gupta, Shivanshu  and
      Gardner, Matt  and
      Singh, Sameer",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.930",
    doi = "10.18653/v1/2023.findings-emnlp.930",
    pages = "13924--13950",
}
@article{gupta2023gistscore,
   title={GistScore: Learning Better Representations for In-Context Example Selection with Gist Bottlenecks},
   author={Shivanshu Gupta and Clemens Rosenbaum and Ethan R. Elenberg},
   year={2023},
   eprint={2311.09606},
   archivePrefix={arXiv},
   primaryClass={cs.CL}
}
```
