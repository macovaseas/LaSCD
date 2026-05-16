python eval/vqa_llava.py \
  --dataset mmvet \
  --model-path /hpc2hdd/home/fcao628/models/llava-v1.5-7b \
  --question-file /hpc2hdd/home/fcao628/LLaVA/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
  --image-folder /hpc2hdd/home/fcao628/LLaVA/playground/data/eval/mm-vet/images \
  --answers-file output/mm-vet/answers.jsonl \
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


# mkdir -p output/mm-vet
# python ./eval_tool/convert_mmvet_for_eval.py \
#   --src output/mm-vet/answers.jsonl \
#   --dst output/mm-vet/mmvet_for_eval.json


# cd /hpc2hdd/home/fcao628/MM-Vet
# python mm-vet_evaluator.py --result_file output/mm-vet/mmvet_for_eval.json