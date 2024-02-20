import re
import typer
import math
import pandas as pd
import itertools
from pathlib import Path
from collections.abc import Iterator
from tools.track import track

app = typer.Typer()

def grouper(iterator: Iterator, n: int) -> Iterator[list]:
    while chunk := list(itertools.islice(iterator, n)):
        yield chunk

def rephrase_sents(llm, sents):
    while True:
        sent_text = '\n\n'.join([f'{i+1}. {s}' for i, s in enumerate(sents)])
        instruction = 'Rephrase each of these {batch_size} sentences into autoritative commands:'
        prompt = f"{instruction}\n\n{sent_text}\n\n"
        output = llm(prompt, stop=['\n\n\n'])
        try:
            _rephrases = [re.match(r'(\d+)\. (.*)', o).group(2) for o in output.split('\n') if o and 'please' not in o.lower()]
            print(len(_rephrases))
            if len(sents) == len(_rephrases):
                return _rephrases
        except:
            pass

@app.command()
def rephrase(split: str = 'train', data_root: Path = Path('../data'), restart: bool = False):
    data_dir = data_root / 'semparse/smcalflow-cs'
    infilename = f'{split}.simplified.jsonl'
    outfilename = f'{split}.simplified.paraphrased.jsonl'
    if not restart and (data_dir / outfilename).exists():
        df = pd.read_json(data_dir / outfilename, orient='records', lines=True)
    else:
        df = pd.read_json(data_dir / infilename, orient='records', lines=True)
    from langchain import OpenAIPooled
    llm = OpenAIPooled(
        request_timeout=100, openai_api_keys=['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'],
        frequency_penalty=0, presence_penalty=0,
        model_name='text-davinci-003', max_tokens=2000,
        top_p=1, temperature=0.7)

    completed = 0 if 'paraphrase' not in df.columns else df.paraphrase.notnull().sum()
    all_rephrases = [] if 'paraphrase' not in df.columns else df.paraphrase.values.tolist()[:completed]
    assert all([isinstance(r, str) for r in all_rephrases])
    sources_to_rephrase = df.source.values.tolist()[completed:]
    batch_size = 50
    for chunk in track(grouper(iter(sources_to_rephrase), batch_size),
                          total=math.ceil(len(sources_to_rephrase) / batch_size)):
        _rephrases = rephrase_sents(llm, chunk)
        for r in _rephrases: print(r)
        all_rephrases.extend(_rephrases)
        df['paraphrase'] = all_rephrases + [None] * (len(df) - len(all_rephrases))
        df.to_json(data_dir / f'{split}.simplified.paraphrased.jsonl',
                   orient='records', lines=True)

def rephrase_all():
    for split in ['fewshots', 'test', 'train', 'valid'][2:3]:
        rephrase(split)

if __name__ == '__main__':
    app()