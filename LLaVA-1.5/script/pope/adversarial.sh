python -m eval.pope_llava \
  --question-file /hpc2hdd/home/fcao628/VCD/experiments/data/POPE/coco/coco_pope_adversarial.json \
  --answers-file output/coco/adversarial.json \
  --conv-mode llava_v1 \
  --temperature 0 \
  --max_new_tokens 1 \
  --top_p 0.9 \
  --alpha 0.1 \
  --start_layer 11 \
  --end_layer 29 \
  --threshold_top_k 10 \
  --laplacian_mode standard \
  --beta 0.0 \
  --energy_mode vision
  # --force_1d  # uncomment to ablate 1-D vs 2-D Laplacian in standard mode


python /hpc2hdd/home/fcao628/LaSCD/eval_tool/eval_pope.py \
  --gt_files /hpc2hdd/home/fcao628/VCD/experiments/data/POPE/coco/coco_pope_adversarial.json \
  --gen_files output/coco/adversarial.json