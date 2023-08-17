from flask import Flask, request, jsonify

import argparse
import transformers
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--hf-model', type=str, default='wellecks/llmstep-mathlib4-pythia2.8b')
parser.add_argument('--port', type=int, default=5000)
parser.add_argument('--num-samples', type=int, default=5)
args = parser.parse_args()

print("Loading model...")
model = transformers.GPTNeoXForCausalLM.from_pretrained(args.hf_model)
if torch.cuda.is_available():
    model.cuda()
model.eval()

tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(args.hf_model) 
print("Done.")


def generate(prompt, num_samples):
    print(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    out = model.generate(
        input_ids,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_samples,
        return_dict_in_generate=True,
        num_beams=num_samples,
        output_scores=True
    )
    texts = tokenizer.batch_decode(
        out.sequences[:,input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    texts = _unique_sorted(texts, out.sequences_scores.tolist())
    return texts


def _unique_sorted(texts, scores):
    texts_, scores_ = [], []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_

app = Flask(__name__)

@app.route('/', methods=['POST'])
def process_request():
    data = request.get_json()

    tactic_state = data.get('tactic_state')

    prompt = """[GOAL]%s[PROOFSTEP]""" % (tactic_state)
    texts = generate(prompt, args.num_samples)

    response = {"suggestions": texts}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=args.port)
