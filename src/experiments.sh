# GPT-4
python experiments.py --label test --seeds 0 --datasets YTIDEOLOGY --selectors bertscore --lms gpt4 --n-shots 4 --baselines-exp --paramsfile params/all.jsonl --run --no-collate-results --preview logfiles

python experiments.py --label test --seeds 0 --datasets YTIDEOLOGY --selectors bertscore --lms gpt4 --n-shots 8 --baselines-exp --paramsfile params/all.jsonl --run --no-collate-results --preview logfiles

python experiments.py --label test --seeds 0 --datasets YTIDEOLOGY --selectors bertscore --lms gpt4 --n-shots 12 --baselines-exp --paramsfile params/all.jsonl --run --no-collate-results --preview logfiles


# LLAMA-7B
python experiments.py --label test --seeds 0 --datasets YTIDEOLOGY --selectors bertscore --lms llama-7B --n-shots 4 --baselines-exp --paramsfile params/all.jsonl --run --no-collate-results --preview logfiles

python experiments.py --label test --seeds 0 --datasets YTIDEOLOGY --selectors bertscore --lms llama-7B --n-shots 8 --baselines-exp --paramsfile params/all.jsonl --run --no-collate-results --preview logfiles

python experiments.py --label test --seeds 0 --datasets YTIDEOLOGY --selectors bertscore --lms llama-7B --n-shots 12 --baselines-exp --paramsfile params/all.jsonl --run --no-collate-results --preview logfiles


# Mistral
python experiments.py --label test --seeds 0 --datasets YTIDEOLOGY --selectors bertscore --lms mistral --n-shots 4 --baselines-exp --paramsfile params/all.jsonl --run --no-collate-results --preview logfiles

python experiments.py --label test --seeds 0 --datasets YTIDEOLOGY --selectors bertscore --lms mistral --n-shots 8 --baselines-exp --paramsfile params/all.jsonl --run --no-collate-results --preview logfiles

python experiments.py --label test --seeds 0 --datasets YTIDEOLOGY --selectors bertscore --lms mistral --n-shots 12 --baselines-exp --paramsfile params/all.jsonl --run --no-collate-results --preview logfiles