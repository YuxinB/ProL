import transformers
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig
from colorama import Fore, Style
from transformers import AutoTokenizer
from transformers import pipeline

def get_prompt():
    txt = "Complet the sequence of the next 10 trials given the first 20 trials are as follows \n\n"
    trials = "01110110100111011100"

    return txt + trials


def gen_txt(model_id, prompt):

    device = "cuda:1"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
                model_id,
                #load_in_8bit=True,
                #load_in_4bit=True,
                torch_dtype=torch.float16,
                device_map=device,
            )

    torch.compile(model, mode="reduce-overhead", fullgraph=True)
    #with torch.device(device):
    #   model.setup_caches(max_batch_size=1, max_seq_length=1250)


    inp_tokens = tokenizer([prompt], return_tensors="pt").to(device)
    ln = inp_tokens['input_ids'].shape[1]

    generated_ids = model.generate(
        **inp_tokens,
        do_sample=False,
        top_p=None,
        temperature=None,
        use_cache=True, # Use-kv cache
        max_new_tokens=50)[:, ln:]

    # end.record()


    gen_tokens = tokenizer.batch_decode(generated_ids)[0]
    gen_tokens = gen_tokens.replace('\n', '\n\n')

    print("-------------------------------------------")
    print(Style.DIM + prompt + Style.NORMAL + gen_tokens)


if __name__ == "__main__":
    model_names = ["tiiuae/falcon-7b",
                   "mistralai/Mistral-7B-v0.1",
                   "meta-llama/Llama-2-7b-hf",
                   "google/gemma-7b",
                  ]
    midx = 2
    pidx = 1

    mname = model_names[midx]
    prompt = get_prompt()

    print("Model: %s" % mname)

    gen_txt(mname, prompt)

