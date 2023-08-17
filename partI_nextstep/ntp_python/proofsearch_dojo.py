# Utilities for interacting with Lean and proof search
import os
import transformers
from lean_dojo import *
import json
import torch
from datetime import datetime
import heapq
import transformers
import random
from typing import List, Tuple
from tqdm import tqdm, trange

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def generate(prompt, model, tokenizer, num_samples):
    input_ids = tokenizer.encode(
        prompt, return_tensors='pt', truncation=True, max_length=1024
    ).to(model.device)

    texts, scores = [], []
    with torch.no_grad():
        out = model.generate(
            input_ids, 
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_samples,
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=num_samples
        )
        output_tokens = out.sequences[:, input_ids.shape[1]:]
        texts.extend(tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True
        ))
        scores.extend(out.sequences_scores.view(-1).tolist())

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores


def _unique_sorted(texts, scores):
    texts_, scores_ = [], []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_


def _tactic_state(state):
    if isinstance(state, TacticState):
        ts = state.pp
    else:
        ts = state.unsolved_tactic_state
    return ts


def _prompt(ts):
    prompt = f"[GOAL]{ts}[PROOFSTEP]"
    return prompt


def best_first_search(theorem, model, tokenizer, max_iters, num_samples, timeout=600) -> dict:
    try:
        with Dojo(theorem, hard_timeout=60 + timeout) as (dojo, init_state):
            queue = [(0.0, [], init_state)]
            visited = set()
            for _ in trange(max_iters):
                if len(queue) == 0:
                    break

                total_score, steps, state = heapq.heappop(queue)
                ts = _tactic_state(state)
                visited.add(ts)

                step_cands, step_scores = generate(
                    _prompt(ts), model, tokenizer, num_samples
                )

                for step, score in zip(step_cands, step_scores):
                    result = dojo.run_tac(state, step)
                    if isinstance(result, ProofFinished):
                        return {
                            'theorem': theorem.full_name, 
                            'proof': steps + [step], 
                            'score': total_score - score,
                            'success': True,
                            'failure_reason': ''
                        }
                    elif isinstance(result, TacticState):
                        if _tactic_state(result) not in visited:
                            # Score is negative log probability summed across steps
                            new_score = (total_score - score)
                            heapq.heappush(
                                queue, (new_score, steps+[step], result)
                            )
    except (DojoInitError, DojoHardTimeoutError, DojoCrashError) as e:
        return {'theorem': theorem.full_name, 'success': False, 'failure_reason': str(e)}

    return {'theorem': theorem.full_name, 'success': False, 'failure_reason': 'SearchEnded'}


def _save(model_name, results, args_dict, dt):
    output_file = 'results__%s__%s.json' % (model_name.replace('/', '_'), dt)
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'args': args_dict
            } , f, indent=4)
        print(output_file)


def load_model(model_name):
    model = transformers.GPTNeoXForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name', 
        default='wellecks/llmstep-mathlib4-pythia2.8b', 
        choices=['wellecks/llmstep-mathlib4-pythia2.8b']
    )
    parser.add_argument('--dataset-path', default='data/val.json')
    parser.add_argument('--max-iters', type=int, default=100)
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--num-examples', type=int, default=200)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, args.vllm)

    URL = "https://github.com/leanprover-community/mathlib4"
    COMMIT = "5a919533f110b7d76410134a237ee374f24eaaad"
    repo = LeanGitRepo(URL, COMMIT)

    dt = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    with open(args.dataset_path) as f:
        data = json.load(f)

        random.seed(43)
        data = random.sample(data, args.num_examples)
        results = []
        for example in tqdm(data, total=len(data)):
            file_path = example['file_path']
            theorem_name = example['full_name']
            theorem = Theorem(repo, file_path, theorem_name)
            result = best_first_search(
                theorem, model, tokenizer, 
                max_iters=args.max_iters,
                num_samples=args.num_samples
            )
            print(result)
            print('\n-----\n')
            results.append(result)

            _save(args.model_name, results, args.__dict__, dt)
            print(len([x for x in results if x['success']])/len(results))