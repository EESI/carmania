
# CARMANIA

**Context-Aware Regularization with Markovian Integration for Attention-Based Nucleotide Analysis**

<p align="center">
  <img src="carmania_logo.png" alt="CARMANIA Logo" width="300"/>
</p>


CARMANIA is a self-supervised genomic language model framework that augments next-token prediction with a transition-matrix regularization loss. This integration improves biological sequence modeling by aligning predicted transitions with empirical bigram(2-mer) statistics, allowing for better long-range dependency modeling and functional interpretation.

---

## 🧠 Pretrained Models

The following models are already available for use on [Hugging Face Hub](https://huggingface.co/MsAlEhR):

- 🦠🧬 [`MsAlEhR/carmania-big-10k-prok-genome`](https://huggingface.co/MsAlEhR/carmania-big-10k-prok-genome)  
- 🦠🧬 [`MsAlEhR/carmania-4k-scp-gene-taxa`](https://huggingface.co/MsAlEhR/carmania-4k-scp-gene-taxa)  
- 👤🧬 [`MsAlEhR/carmania-160k-seqlen-human`](https://huggingface.co/MsAlEhR/carmania-160k-seqlen-human)

---

## 🚀 Quick Start

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained(
    "MsAlEhR/carmania-160k-seqlen-human",
    trust_remote_code=True,
    torch_dtype=torch.float16,   # fixed dtype (or autocast)
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    "MsAlEhR/carmania-160k-seqlen-human",
    trust_remote_code=True,
    model_max_length=160000,
)

inputs = tokenizer("ACGTAGGCTA", return_tensors="pt").to("cuda")

outputs = model(**inputs)
```

## 🧪 Sequence-Guided Generation

An experimental notebook exploring **CARMANIA-driven sequence optimization** using Enformer scores is now available.  
This lightweight module perturbs input DNA sequences and uses Enformer’s predicted regulatory signals as a scoring function to iteratively generate variants with improved activity.

📄 **Notebook:**  
[carmania_enformer_guided_generation.ipynb](https://github.com/EESI/carmania/blob/main/notebooks/carmania_enformer_guided_generation.ipynb)


## Citation

```bibtex
@article{refahi2026context,
  title={Context-aware regularization with markovian integration for attention-based nucleotide analysis},
  author={Refahi, Mohammad Saleh and Abavisani, Mahdi and Sokhansanj, Bahrad and Brown, James R and Rosen, Gail},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={108518--108544},
  year={2026}
}


