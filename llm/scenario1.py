import transformers
import tqdm
import re
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig
from colorama import Fore, Style
from transformers import AutoTokenizer
from transformers import pipeline

bernoulli_p = 0.9

def get_prompt(n):

    txt1 = "Consider the following sequence of outcomes generated from a single Bernoulli distribution\n\n"

    generate_coinflips = []
    for i in range(n):
        coinflip = "1" if np.random.uniform(0, 1) < bernoulli_p else "0"
        generate_coinflips.append(coinflip)
    generate_coinflips = "".join(generate_coinflips)

    txt2 = "\n\n the next 20 most likely outcomes are:\n\n"

    prompt = txt1 + generate_coinflips + txt2

    return prompt


def gen_txt(model_id):

    device = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
                model_id,
                #load_in_8bit=True,
                #load_in_4bit=True,
                torch_dtype=torch.float16,
                device_map=device,
            )

    torch.compile(model, mode="reduce-overhead", fullgraph=True)

    all_errs = []
    for n in tqdm.tqdm(range(2, 100, 5)):
        cur_errs = []
        for seed in range(100):
            prompt = get_prompt(n)

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

            errs = []
            for ni in numbers[-1]:
                if ni == "1" or ni == "0":
                    if ni == "1":
                        errs.append(1 - bernoulli_p)
                    else:
                        errs.append(bernoulli_p)

            cur_errs.append((n, np.mean(errs)))
        all_errs.append(cur_errs)

    # store all_errs
    mname = model_id.split("/")[1]
    b_txt = str(int(bernoulli_p * 100))

    np.save("data/scenario1_%s_%s.npy" % (mname, b_txt), all_errs)

    all_errs = np.array(all_errs)
    mean = all_errs.mean(1)
    std = all_errs.std(1)

    print("-------------------------------------------")
    print(Style.DIM + prompt + Style.NORMAL + gen_tokens)



if __name__ == "__main__":
    model_names = ["tiiuae/falcon-7b",
                   "mistralai/Mistral-7B-v0.1",
                   "meta-llama/Llama-2-7b-hf",
                   "google/gemma-7b",
                  ]
    midx = 2

    mname = model_names[midx]
    print("Model: %s" % mname)

    gen_txt(mname)

