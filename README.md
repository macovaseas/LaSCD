<div align=center>
  
# When Looking Is Not Enough: Visual Attention Structure Reveals Hallucination in MLLMs

[![arXiv](https://img.shields.io/badge/arXiv-2605.11559-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2605.11559)

</div>

## Overview

**LaSCD** is a training-free decoding strategy that selects informative layers via Laplacian energy and remaps next-token logits in closed form, designed for multi-modal hallucination mitigation.


## 📣 News
* `2026/05/16` 📌 Source code released!


## Experiments

### LLaVA-v1.5

#### 1. Setup

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Download the [**LLaVA-1.5** checkpoint](https://huggingface.co/liuhaotian/llava-v1.5-7b) and the official **evaluation assets** following the upstream [LLaVA](https://github.com/haotian-liu/LLaVA) instructions.


#### 2. Evaluation

During `model.generate`, LaSCD is enabled via `use_lascd=True` and the usual knobs passed through from each eval script, for example:

| Argument | Role |
|----------|---------------------|
| `alpha`, `beta` | Interpolation strengths in the logits remapping. |
| `start_layer`, `end_layer` | Decoder layer indices scanned for Laplacian / selection metrics (`early_exit_layers` in generation). |
| `threshold_top_k`, `threshold_top_p` | Top-\(p\) / top-\(k\) filtering applied in the decoding path. |
| `laplacian_mode` | One of `standard`, `sobel`, `log` (discrete Laplacian family). |
| `energy_mode` | `vision` or `text` slice for computing layer-wise energy. |
| `force_1d` | Ablates 2-D vs 1-D Laplacian when `laplacian_mode=standard`. |


You can use the following script to obtain the results:

```bash
cd LLaVA-1.5
bash script/mme.sh
```
#### TODO
We will provide code for more MLLMs (Qwen, GLM, ...) in a future release.

## ✏️ Citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```
@misc{cao2026looking,
      title={When Looking Is Not Enough: Visual Attention Structure Reveals Hallucination in MLLMs}, 
      author={Fanpu Cao and Xin Zou and Xuming Hu and Hui Xiong},
      year={2026},
      eprint={2605.11559},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.11559}, 
}
```


## 📝 Related Projects
- [MemVR](https://github.com/1zhou-Wang/MemVR): Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models
- [DeCo](https://github.com/zjunlp/Deco): MLLM Can See? Dynamic Correction Decoding For Hallucination Mitigation
- [OPERA](https://github.com/shikiw/OPERA): OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation
- [VCD](https://github.com/DAMO-NLP-SG/VCD): VCD: Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding
- [DoLa](https://github.com/voidism/DoLa): DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models
- [Contrastive Decoding](https://github.com/XiangLi1999/ContrastiveDecoding): Open-ended Text Generation as Optimization
- [GLM-4V](https://github.com/THUDM/GLM-4): ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL): A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
- [LLaVA 1.5](https://github.com/haotian-liu/LLaVA): Improved Baselines with Visual Instruction Tuning

## :e-mail: Contact

If you have any questions, please email [`fanpucao@gmail.com`](mailto:fanpucao@gmail.com)