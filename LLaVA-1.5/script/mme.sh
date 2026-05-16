python eval/mme_llava.py \
  --model-path /hpc2hdd/home/fcao628/models/llava-v1.5-7b \
  --image-folder /hpc2hdd/home/fcao628/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
  --question-file /hpc2hdd/home/fcao628/LLaVA/playground/data/eval/MME/llava_mme.jsonl \
  --answers-file output/mme/mme.jsonl \
  --log_path log \
  --temperature 0 \
  --max_new_tokens 10 \
  --top_p 0.9 \
  --alpha 0.1 \
  --start_layer 2 \
  --end_layer 11 \
  --threshold_top_k 10 \
  --laplacian_mode standard \
  --beta 0.0 \
  --energy_mode vision
  # --force_1d  # uncomment to ablate 1-D vs 2-D Laplacian in standard mode

python /hpc2hdd/home/fcao628/LaSCD/eval_tool/convert_answer_to_mme.py \
  --output_path output/mme/mme.jsonl \
  --log_path output/mme

python /hpc2hdd/home/fcao628/LaSCD/eval_tool/calculation.py \
  --results_dir output/mme