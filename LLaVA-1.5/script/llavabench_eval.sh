#!/usr/bin/env bash

set -euo pipefail

QUESTION_FILE="${QUESTION_FILE:-/hpc2hdd/home/fcao628/LLaVA/playground/data/eval/llava-bench-in-the-wild/questions.jsonl}"
CONTEXT_FILE="${CONTEXT_FILE:-/hpc2hdd/home/fcao628/LLaVA/playground/data/eval/llava-bench-in-the-wild/context.jsonl}"
BASELINE_ANSWERS="${BASELINE_ANSWERS:-/hpc2hdd/home/fcao628/LLaVA/playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl}"
CANDIDATE_ANSWERS="${CANDIDATE_ANSWERS:- /hpc2hdd/home/fcao628/LaSCD/LLaVA-1.5/output/llavabench/answers.jsonl}"
JUDGE_MODEL="${JUDGE_MODEL:-${MODEL:-gpt-4o}}"
if [[ "$JUDGE_MODEL" == "gpt4o" ]]; then
  JUDGE_MODEL="gpt-4o"
fi
MAX_TOKENS="${MAX_TOKENS:-1024}"
REVIEW_FILE="${REVIEW_FILE:-/hpc2hdd/home/fcao628/LaSCD/LLaVA-1.5/output/llavabench/review_${JUDGE_MODEL//\//_}.jsonl}"

python /hpc2hdd/home/fcao628/LaSCD/eval_tool/eval_gpt_review_bench.py \
  --question "$QUESTION_FILE" \
  --context "$CONTEXT_FILE" \
  --answer-list "$BASELINE_ANSWERS" "$CANDIDATE_ANSWERS" \
  --output "$REVIEW_FILE" \
  --model "$JUDGE_MODEL" \
  --max-tokens "$MAX_TOKENS"

python /hpc2hdd/home/fcao628/LaSCD/eval_tool/summarize_gpt_review.py \
  --files "$REVIEW_FILE"
