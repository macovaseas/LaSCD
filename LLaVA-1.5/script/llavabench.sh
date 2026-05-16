python eval/vqa_llava.py \
  --dataset llavabench \
  --model-path /hpc2hdd/home/fcao628/models/llava-v1.5-7b \
  --question-file /hpc2hdd/home/fcao628/LLaVA/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
  --image-folder /hpc2hdd/home/fcao628/LLaVA/playground/data/eval/llava-bench-in-the-wild/images \
  --answers-file output/llavabench/answers.jsonl \
  --cuda-device cuda:0 \
  --temperature 0 \
  --top_p 0.9 \
  --max_new_tokens 1024 \
  --alpha 0.1 \
  --start_layer 11 \
  --end_layer 21 \
  --threshold_top_p 0.9 \
  --threshold_top_k 10 \
  --laplacian_mode standard \
  --beta 0.0 \
  --energy_mode vision \
  --use-lascd

bash script/llavabench_eval.sh