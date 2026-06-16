<div align="center">
  <h1>Expressivity of Language Models</h1>
    <i>Why do transformers struggle to produce some distributions more than others? </i><br><br>
</div>


This repository includes a partial reproduction of *Distribution Prompting: Understanding the Expressivity of Language
Models Through the Next-Token Distributions They Can Produce* ([Wang et al., 2025](https://arxiv.org/abs/2505.12244)). Through a small further experiment, I believe the phenomenon on Llama 3.2 1B described in this paper is not a feature of attention but can rather be observed on the final components of the network, namely the unembedding and softmax. 

## Run Instructions

Clone repository and install dependencies
```bash
git clone https://github.com/AndyyyYuuu/lm-is-compressor.git
cd lm-is-compressor
pip install -r requirements.txt
```

Recommend creating and activating a [virtual environment](https://gist.github.com/ryumada/c22133988fd1c22a66e4ed1b23eca233).

Create `.env` following `.env.example`. Since [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) is a gated model, you will need to request access from your Hugging Face account and include your access key in the `.env`

To run an experiment, run [`main.ipynb`](main.ipynb). 

## Experiments

The experiments in this repo are based on soft prompt tuning on vanilla distributions, a method laid out in [Wang et al. (2025)](https://arxiv.org/abs/2505.12244). My own figures can be found in [`results/`](results/)

I first partially replicate the results of [Figure 4 (c)](https://arxiv.org/html/2505.12244v2#S5.F4). There is clearly a middle range of entropies that tuning LLM soft prompts struggles to express. 

<img width="332" height="312" alt="" src="https://github.com/user-attachments/assets/19f4ff53-7470-43a7-815d-3e67cd3c8ede" />

In my own analogous experiments, I tune injected values in the hidden layers of the transformer and investigate their expressivity. I find that the bottleneck on expressivity follows a nearly identical pattern even when I tune the output of the final layer of the transformer, and even when the intervention is placed after RMSNorm (shown below). This suggests that the cause of the phenomenon can be narrowed down to the final computations in a transformer (unembedding transform and softmax) and are largely not due to the multi-headed attention or other earlier components. 


<img width="332" height="312" alt="" src="https://github.com/user-attachments/assets/b25b46f9-2503-46b0-a6ab-159ece4a1a5f" />
<img width="332" height="312" alt="Screenshot 2026-06-15 at 5 15 59 PM" src="https://github.com/user-attachments/assets/f3f22981-560a-4a68-ae92-5618c527ac23" />

One curiousity worth pointing out is that although the shapes of the curves stayed identical when expressivity was measured from the embedding output v. the final layer's output, the faint downward trendline in the background of the former's scatter plot is not seen in the latter's. Spooky. 

## Future Work

If I feel like it, I'll do an intervention with a randomly initialized matrix and softmax to see how far this behaviour goes. 

## Useful References

- The paper by Wang et al. (2025): https://arxiv.org/abs/2505.12244
- Llama architecture source code: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py 
