python eval/vqa_llava.py \
  --dataset hallusionbench \
  --model-path /hpc2hdd/home/fcao628/models/llava-v1.5-7b \
  --dataset-file /hpc2hdd/home/fcao628/HallusionBench/HallusionBench.json \
  --dataset-root /hpc2hdd/home/fcao628/HallusionBench \
  --image-folder /hpc2hdd/home/fcao628/HallusionBench/hallusion_bench \
  --answers-file output/hallusionbench/answers.json \
  --cuda-device cuda:0 \
  --temperature 0 \
  --top_p 0.9 \
  --max_new_tokens 1024 \
  --alpha 0.1 \
  --start_layer 11 \
  --end_layer 29 \
  --threshold_top_p 0.9 \
  --threshold_top_k 20 \
  --laplacian_mode standard \
  --beta 0.1 \
  --energy_mode vision \
  --use-lascd


# cd /hpc2hdd/home/fcao628/HallusionBench
# python evaluation.py -i /hpc2hdd/home/fcao628/LaSCD/LLaVA-1.5/output/hallusionbench/answers.json 
# --gpt-model gpt-4o