from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from tqdm import tqdm
from pathlib import Path
import os
import json

TOKEN_MAP = {
    'pad': '[PAD]',
    'eos': '<|endoftext|>',
}

PHRASE_MAP = {
    'goal': '[GOAL]',
    'proofstep': '[PROOFSTEP]',
}


def _download_and_unpack(tarball_url, data_dir, overwrite):
    import subprocess
    if (not overwrite) and Path(data_dir).exists():
        return
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    archive_path = os.path.join(data_dir, "archive.tar.gz")
    subprocess.call(['wget', '-O', archive_path, tarball_url])
    subprocess.call(['tar', '-xzf', archive_path, '-C', data_dir])


def _load_ds(data_dir):
    ds = {}
    for split in ['train', 'val', 'test']:
        ds[split] = json.load(open(os.path.join(
            data_dir, 'leandojo_benchmark_4', 'random', f'{split}.json'), 'r')
        ) 
    return ds


def _save_splits(splits, data_dir, tag):
    print("Saving split to disk...")
    out_dir = os.path.join(data_dir, 'processed')
    for split, examples in tqdm(splits.items(), total=len(splits)):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_file = os.path.join(
            out_dir, '%s-%s.jsonl' % (tag, split)
        )
        with open(out_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example))
                f.write('\n')


def _print_stats(splits):
    for split, examples in splits.items():
        print("%s\t%d" % (split, len(examples)))


    

# --- Proofstep
def _fmt_proofstep(state_before, tactic):
    # [GOAL]{state_before}[PROOFSTEP]{tactic}<|endoftext|>
    inp = f"{PHRASE_MAP['goal']}{state_before}{PHRASE_MAP['proofstep']}"
    out = f"{tactic}{TOKEN_MAP['eos']}"
    return inp, out


def fmt_proofstep(split):
    examples = []
    for traced_theorem in split:
        for tactic_example in traced_theorem['traced_tactics']:
            inp, out = _fmt_proofstep(tactic_example['state_before'], tactic_example['tactic'])
            examples.append({
                'input': inp,
                'output': out,
            })
    return examples


def proofstep(data_dir):
    ds = _load_ds(data_dir)
    out_ds = {}
    for split in ds:
        out_ds[split] = fmt_proofstep(ds[split])
    
    _save_splits(
        splits=out_ds,
        data_dir=data_dir,
        tag='proofstep'
    )
    _print_stats(
        splits=out_ds
    )
    return out_ds
# ---


def main(args):
    proofstep(args.datadir)


def setup(args):
    # Download data
    _download_and_unpack(
        tarball_url='https://zenodo.org/record/8040110/files/leandojo_benchmark_4_v1.tar.gz',
        data_dir=args.data_dir,
        overwrite=args.overwrite
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--data-dir', type=str, default='./data/leandojo_benchmark_4')

    args = parser.parse_args()
    setup(args)
    main(args)
