# Utilities for interacting with Lean and proof search

from pylean import LeanServer
import torch
import heapq
import concurrent
import transformers
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  


def is_done(state):
    return state['sorries'] == [] and state['messages'] == []


def get_goal(state):
    goal = None
    for msg in state['messages']:
        if msg['data'].startswith('unsolved goals\n'):
            goal = '\n'.join(msg['data'].split('\n')[1:])
        elif msg['severity'] == 'error':
            return None
    return goal


def get_errors(state):
    return state['messages']


def parse_step(step):
    step = step.replace('<|endoftext|>', '')
    return step


def format_code(header, statement, steps_so_far, next_step):
    return header + (statement.replace(" {}", "") + '\n' + '\n'.join(steps_so_far + [next_step]))
    

def run_code(code):
    lean = LeanServer()
    out = lean.run_code(code)
    lean.proc.close()
    del lean
    return out


def sequence_scores(out, prompt_length, model, tokenizer):
    # Returns each output sequence's log probability normalized by the number of tokens.
    # An output sequence is defined as the tokens after the prompt up to and including eos.
    text = tokenizer.batch_decode(out.sequences)
    input_ids = tokenizer(
        text, return_tensors="pt", padding='longest', truncation=True
    ).to(model.device)
    with torch.no_grad():
        out = model(**input_ids)
        probs = torch.log_softmax(out.logits, dim=-1).detach()
        probs = probs[:, :-1, :]
        input_ids_shifted = input_ids.input_ids[:, 1:]
        log_probs = torch.gather(probs, 2, input_ids_shifted[:, :, None]).squeeze(-1)
        log_probs = log_probs[:, prompt_length:]
        up_to_eos_mask = (input_ids_shifted[:,prompt_length:].eq(
            tokenizer.eos_token_id).cumsum(1).cumsum(1) <= 1).type(log_probs.dtype)
        normalized_sequence_scores = (log_probs * up_to_eos_mask).sum(1) / up_to_eos_mask.sum(1)
    return normalized_sequence_scores


def generate(prompt, model, tokenizer, temperatures, num_samples) -> Tuple[List[str], List[float]]:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    texts = []
    scores = []
    with torch.no_grad():
        # Does beam search at temp 0.0, otherwise temperature sampling.
        for temp in temperatures:
            decoding_params = dict(
                max_new_tokens=256,
                do_sample=temp > 0,
                temperature=temp,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=num_samples,
                return_dict_in_generate=True,
                output_scores=True,
            )
            if temp == 0.0:
                decoding_params['num_beams'] = num_samples
            out = model.generate(
                input_ids, **decoding_params
            )
            
            texts.extend(tokenizer.batch_decode(
                out.sequences[:,input_ids.shape[1]:],
                skip_special_tokens=True
            ))
            scores_ = sequence_scores(
                out=out, 
                prompt_length=input_ids.shape[1], 
                model=model, 
                tokenizer=tokenizer
            )
            scores.extend(scores_.view(-1).tolist())

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores


def _unique_sorted(texts, scores):
    texts_, scores_ = [], []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_


def _print_type_checked_candidates(results):
    print('--- type-checked candidates:\n\t' + '\n\t'.join(
        '(%.3f) %s' % (step_score, step) 
        for state, step, step_score in results if (
        get_goal(state) is not None or is_done(state))
    ))


def _print_current(theorem_statement, steps):
    print('--- current:\n\t%s\n\t%s' % (
        theorem_statement.replace('{}', ''),
        '\n\t'.join(steps)) 
    )


def best_first_search(model, tokenizer, header, statement, max_iters, temperatures, num_samples, verbose=False) -> dict:
    goal = get_goal(run_code(header + statement))
    if goal is None:
        return {
            'theorem_statement': statement, 
            'success': False, 
            'msg': run_code(header + statement)
        }

    # Score, steps-so-far, goal state
    queue = [(0.0, [], goal)]
    visited = set()
    while len(queue) > 0 and max_iters > 0:
        # Dequeue the tuple with minimum score
        score, steps, goal = heapq.heappop(queue)
        visited.add(goal)
        if verbose:
            _print_current(statement, steps)

        # Generate next-step candidates
        prompt = f"[GOAL]{goal}[PROOFSTEP]"
        step_cands, step_scores = generate(
            prompt, 
            model, 
            tokenizer, 
            temperatures=temperatures, 
            num_samples=num_samples
        )

        # Run type checking in parallel via futures. 
        with ThreadPoolExecutor(max_workers=16) as executor:
            # We need to save the step and score associated to each future.
            future2step = {}
            for step, step_score in zip(step_cands, step_scores):
                code = format_code(header, statement, steps, step)
                future = executor.submit(run_code, **dict(code=code))
                future2step[future] = (step, step_score)

            # Collect the type checking results as they complete.
            results = []
            for future in tqdm(concurrent.futures.as_completed(future2step.keys()), total=len(future2step)):
                result = future.result()
                results.append((result, *future2step[future]))

        if verbose:
            _print_type_checked_candidates(results)
        for state, step, step_score in results:
            # Stop if we have found a complete proof.
            if is_done(state):
                return {
                    'theorem_statement': statement, 
                    'proof': steps + [step], 
                    'state': state,
                    'score': score - step_score,
                    'success': True
                }
            goal_cand = get_goal(state)
            # Add new candidates to the queue.
            if goal_cand is not None and goal_cand not in visited:
                # Score is normalized negative log probability summed across steps
                new_score = (score - step_score)
                heapq.heappush(
                    queue, (new_score, steps+[step], goal_cand)
                )
        
        max_iters -= 1

    return {'theorem_statement': statement, 'success': False}


def _save(results):
    from datetime import datetime
    import json
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    output_file = 'results__%s.json' % (dt_string)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        print(output_file)


def load_model(model_name):
    model = transformers.GPTNeoXForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = load_model('wellecks/llmstep-mathlib4-pythia2.8b')

    evaluation_theorems = [
        """theorem thm1 (a b c : Nat) : a + b = c → a ≤ c := by {}""",
        """theorem thm2 (x y : ℝ) : x < y → 0 < y - x := by {}""",
        """theorem thm3 (n : Nat) : n ≥ 0 := by {}""",
        """theorem thm4 (x y z : ℝ) : x ≤ y → y ≤ z → x ≤ z := by {}""",
        """theorem thm5 (m n : Nat) (h : m.coprime n) : m.gcd n = 1 := by {}""",
        """theorem thm6: r ⊆ s → s ⊆ t → r ⊆ t := by {}""",
        """theorem thm7 (f : ℕ → ℕ) : Monotone f → ∀ n, f n ≤ f (n + 1) := by {}""",
        """theorem thm8 (c : ℝ) : Injective fun x => x + c := by {}""",
        """theorem thm9 (p q : Prop) : (p ∧ q) → ¬(¬p ∨ ¬q) := by {}""",
        """theorem thm10 (A B : Set ℕ) : A ⊆ B → ∀ n, n ∈ A → n ∈ B := by {}""",
        """theorem thm11 (injg : Injective g) (injf : Injective f) : Injective fun x => g (f x) := by {}""",
        """theorem thm12 (a b : ℕ) (h : a ≤ b) : a * (a + 1) ≤ b * (b + 1) := by {}""",
        """theorem thm13 (a b : ℕ) (h : a ≠ b) : a * 2 ≠ b * 2 := by {}""",
    ]
        
    # Shared header for the theorems above
    header = """import Mathlib.Data.Nat.Factorization.Basic
    import Mathlib.Data.Nat.Prime
    import Mathlib.Data.Real.Basic
    
    open BigOperators
    open Function
    variable {α : Type _} (r s t : Set α)
    
    """

    results = []
    for theorem in evaluation_theorems:
        result = best_first_search(
            model, tokenizer, header, theorem, 
            max_iters=32,
            temperatures=[0.5],
            num_samples=16
        )
        print(result)
        print('\n-----\n')
        results.append(result)

    print(len([x for x in results if x['success']])/len(results))
    _save(results)