"""
Script to run experiments for parameters dumped in a jsonl file in parallel on multiple gpus. Sample run:
```
python run.py --paramsfile "params/params.jsonl" --gpus 0,1,2
```
"""

import jsonlines
import os
import queue
import torch
import time
from rich import print
from typer import Typer, Option
from pathlib import Path
from joblib import Parallel, delayed, parallel_backend

from params import AllParams
from tools.exp import get_ints

app = Typer()
q = queue.Queue()

def run_cmd(cmd, outfile: Path = None, tee_output=False, verbose=False, debug=False):
    import shlex
    import subprocess
    if verbose:
        print(cmd)
        print(f'Logging to: {outfile}')
    if debug: return

    args = shlex.split(cmd)
    os.makedirs(outfile.parent, exist_ok=True)
    if outfile:
        if tee_output:
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            tee = subprocess.Popen(['tee', outfile], stdin=process.stdout)
            process.stdout.close()
            tee.communicate()
        else:
            process = subprocess.Popen(args, stdout=outfile.open('w'), stderr=subprocess.STDOUT)
    else:
        process = subprocess.Popen(args)
    ret = process.wait()
    return ret

@app.command()
def run_exps_parallel(
    paramsfile: Path = Option('params.jsonl', help='Path to the params file.'),
    gpus: str = Option('0,1,2,3,4,5,6,7', help='Comma separated list of GPUs to use. This will index into CUDA_VISIBLE_DEVICES if set.'),
    start_idx: int = Option(0, help='Start from this index.'),
    debug: bool = Option(False, help='Run in debug mode.'),
    clear_logs: bool = Option(False, help='Clear logs.'),
):
    if not paramsfile.exists():
        print('Params file does not exist...')
        return

    with jsonlines.open(paramsfile, mode='r') as reader:
        params_l = [AllParams.from_dict(p) for p in reader][start_idx:]
        if clear_logs:
            for p in params_l:
                outfile = p.outfile if not p.exp.only_prompts else p.promptsoutfile
                logfile = p.logfile if not p.exp.only_prompts else p.promptslogfile
                if outfile.exists(): os.remove(outfile)
                if logfile.exists(): os.remove(logfile)


    gpus = get_ints(gpus, sep=',')
    # if 'CUDA_VISIBLE_DEVICES' in os.environ:
    #     print(os.environ['CUDA_VISIBLE_DEVICES'])
    #     gpus = [get_ints(os.environ['CUDA_VISIBLE_DEVICES'], ',')[i] for i in gpus]
    print(gpus)
    for gpu in gpus:
        q.put(gpu)

    n_jobs = len(params_l)
    n_concurrent = len(gpus)

    def run_wrapper(i, params: AllParams):
        outfile = params.outfile if not params.exp.only_prompts else params.promptsoutfile
        print(f'  > {i+1}/{n_jobs} {outfile}')
        gpu = q.get(block=True)
        params.exp.gpu = gpu
        cmd = params.cmd
        print(i + 1, cmd)
        run_cmd(cmd, outfile, tee_output=False, verbose=False, debug=debug)
        q.put(gpu)
        torch.cuda.empty_cache()
        print(f'  < {i+1}/{n_jobs} {outfile}')

    print(f'Running {len(params_l)} jobs...')
    start_time = paramsfile.stat().st_mtime
    while params_l:
        with Parallel(n_jobs=n_concurrent, require='sharedmem', verbose=True) as parallel:
            parallel(delayed(run_wrapper)(i, params)
                     for i, params in enumerate(params_l))
        if debug: break
        if not any([p.completed_after(start_time) for p in params_l]):
            print('all jobs failed')
            break
        params_l = [p for p in params_l if not p.completed_after(start_time)]
        if params_l:
            print(f'Rerunning {len(params_l)} failed jobs...')
        else:
            print('All jobs completed')

if __name__ == '__main__':
    app()
