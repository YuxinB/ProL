import transformers
import argparse
import tqdm
import re
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig
from colorama import Fore, Style
from transformers import AutoTokenizer
from transformers import pipeline


def get_prompt(n, bernoulli_p):

    txt1 = "Consider the following sequence of states generated by a Markov process with 2 states (0, 1):\n\n"

    markov_txt = []
    targets = []
    transition = [[1 - bernoulli_p, bernoulli_p],
                  [bernoulli_p, 1 - bernoulli_p]]
    state = 0
    for i in range(n+20):
        if i < n:
            markov_txt.append(str(state))
        else:
            targets.append(state)
        state = np.random.choice([0, 1], p=transition[state])

    markov_txt = "".join(markov_txt)

    txt2 = "\n\nThe next 20 most likely sequence of states are:\n\n"

    prompt = txt1 + markov_txt + txt2

    return prompt, targets


def gen_txt(model_id, bernoulli_p):

    device = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device,
            )

    torch.compile(model, mode="reduce-overhead", fullgraph=True)

    np.random.seed(0)
    seeds = 1
    all_errs = []
    for n in tqdm.tqdm(range(2, 100, 6)):
        cur_errs = []
        for seed in range(seeds):
            prompt, targets = get_prompt(n, bernoulli_p)

            inp_tokens = tokenizer([prompt], return_tensors="pt").to(device)
            ln = inp_tokens['input_ids'].shape[1]

            generated_ids = model.generate(
                **inp_tokens,
                do_sample=False,
                top_p=None,
                temperature=None,
                use_cache=True, # Use-kv cache
                max_new_tokens=20)[:, ln:]

            gen_tokens = tokenizer.batch_decode(generated_ids)[0]
            gen_tokens = gen_tokens.replace('\n', '\n\n')

            # get all numbers from the generated text
            numbers = re.findall(r'\d+', gen_tokens)

            maxidx = -1
            maxval = -1
            for idx, ne in enumerate(numbers):
                if len(ne) > maxval:
                    maxval = max(maxval, len(ne))
                    maxidx = idx

            errs = []
            for idx, ni in enumerate(numbers[maxidx]):
                if ni == "1" or ni == "0":
                    err = int(targets[idx] != int(ni)) * ((0.9) ** (idx))
                    errs.append(err)

            if len(errs) == 0:
                errs = [all_errs[-1][1]]
                print("flag", gen_tokens, numbers[maxidx])
            cur_errs.append((n, np.mean(errs)))
        all_errs.append(cur_errs)

    # store all_errs
    mname = model_id.split("/")[1]
    b_txt = str(int(bernoulli_p * 100))
    np.save("data/scenario3_%s_%s.npy" % (mname, b_txt), all_errs)

    all_errs = np.array(all_errs)
    mean = all_errs.mean(1)
    std = all_errs.std(1)

    print("-------------------------------------------")
    print(Style.DIM + prompt + Style.NORMAL + gen_tokens)



if __name__ == "__main__":

    # argparse command line for midx and bernoulli_p
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--p", type=float, default=0.9)

    args = parser.parse_args()

    midx = args.m
    bernoulli_p = args.p


    model_names = ["tiiuae/falcon-7b",
                   "mistralai/Mistral-7B-v0.1",
                   "meta-llama/Llama-2-7b-hf",
                   "google/gemma-7b",
                  ]

    mname = model_names[midx]
    print("Model: %s" % mname)

    gen_txt(mname, bernoulli_p)

