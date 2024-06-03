import transformers
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig
from colorama import Fore, Style
from transformers import AutoTokenizer
from transformers import pipeline

def get_prompt():

    prefix = "Generate the outcomes of 10 Bernoulli trials with probability of generating 0 is 0.5, 1 is also 0.5:\n\n"

    prompts = []
    for i in range(1024):
        s = format(i, '010b')
        prompts.append(prefix + s)

    return prompts


def gen_txt(model_id):

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

    prompts = get_prompt()

    all_log_prob = []
    for pr in prompts:
        inp_tokens = tokenizer([pr], return_tensors="pt")

        rvs = pr.split("\n\n")[1]
        rv_tokens = tokenizer.encode(rvs)[2:]

        ln = len(pr), len(rvs)
        tok_ln = len(inp_tokens["input_ids"][0])
        rv_ln = len(rv_tokens)


        logits = model(inp_tokens['input_ids'].to(device), return_dict=True).logits

        logits = logits[0, tok_ln - rv_ln - 1:, :]
        logits = logits[:-1]

        ind0 = tokenizer.convert_tokens_to_ids(["0"])[0]
        ind1 = tokenizer.convert_tokens_to_ids(["1"])[0]

        # delete all other logits
        logits = logits[:, [ind0, ind1]]
        logits = torch.nn.functional.softmax(logits, dim=-1)

        log_prob_list = []
        for i, tok in enumerate(rv_tokens):
            st = tokenizer.decode(tok)

            if st == "0" or st == "1":
                log_prob = np.log(logits[i, int(st)].item())
            else:
                log_prob = 0

            log_prob_list.append(log_prob)
        all_log_prob.append(log_prob_list)
    all_log_prob = np.array(all_log_prob)
    import ipdb; ipdb.set_trace() 

    np.save("data/gen_probs.npy", all_log_prob)



if __name__ == "__main__":
    model_names = ["tiiuae/falcon-7b",
                   "mistralai/Mistral-7B-v0.1",
                   "meta-llama/Llama-2-7b-hf",
                   "google/gemma-7b",
                  ]
    midx = 2
    pidx = 1

    mname = model_names[midx]
    print("Model: %s" % mname)

    gen_txt(mname)

