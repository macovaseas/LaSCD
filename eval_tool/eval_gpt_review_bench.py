import argparse
import json
import os

import openai
from openai import OpenAI
import time

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LLAVABENCH_RULE = os.path.join(_TOOL_DIR, "llavabench_rule.json")

NUM_SECONDS_TO_SLEEP = 0.5


def _is_rate_limit_error(exc: Exception) -> bool:
    rate_limit_error = getattr(openai, "RateLimitError", None)
    if rate_limit_error is not None and isinstance(exc, rate_limit_error):
        return True

    error_mod = getattr(openai, "error", None)
    legacy_rate_limit_error = getattr(error_mod, "RateLimitError", None) if error_mod is not None else None
    return legacy_rate_limit_error is not None and isinstance(exc, legacy_rate_limit_error)


def _create_chat_completion(model_name: str, content: str, max_tokens: int):
    messages = [
        {
            'role': 'system',
            'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
        },
        {
            'role': 'user',
            'content': content,
        },
    ]

    client_cls = getattr(openai, "OpenAI", None)
    if client_cls is not None:
        # client = client_cls()
        OPENAI_API_KEY = '--'
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://openai.com/api/v1"  # <--- 修改这里
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return response['choices'][0]['message']['content']


def get_eval(model_name: str, content: str, max_tokens: int):
    while True:
        try:
            return _create_chat_completion(model_name, content, max_tokens)
        except KeyboardInterrupt:
            break
        except Exception as e:
            if _is_rate_limit_error(e):
                time.sleep(NUM_SECONDS_TO_SLEEP)
                continue
            print(e)
            raise
        time.sleep(NUM_SECONDS_TO_SLEEP)

    raise RuntimeError("Evaluation interrupted before a response was received.")


def parse_score(review: str):
    try:
        score_pair = review.split('\n')[0].strip()
        score_pair = score_pair.replace(',', ' ')
        sp = [x for x in score_pair.split(' ') if x]
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        print('error', review)
        return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


def validate_openai_env():
    if os.environ.get("OPENAI_API_KEY"):
        return

    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Please export a valid OpenAI API key before running llava-bench GPT review."
    )


def get_default_output_path(answer_path: str, model_name: str) -> str:
    answer_abs_path = os.path.abspath(os.path.expanduser(answer_path))
    answer_dir = os.path.dirname(answer_abs_path)
    model_tag = model_name.replace("/", "_")
    return os.path.join(answer_dir, f"review_{model_tag}.jsonl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument(
        '-r',
        '--rule',
        default=_DEFAULT_LLAVABENCH_RULE,
        help=f"默认: {_DEFAULT_LLAVABENCH_RULE}",
    )
    parser.add_argument('-o', '--output')
    parser.add_argument('--model', default='gpt-4o', help='judge model name')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    # validate_openai_env()

    if len(args.answer_list) != 2:
        raise ValueError("--answer-list must contain exactly two jsonl files: baseline first, candidate second.")

    if args.output is None:
        args.output = get_default_output_path(args.answer_list[1], args.model)

    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')

    context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
    image_to_context = {context['image']: context for context in context_list}

    idx = 0
    for ques_js, ans1_js, ans2_js in zip(f_q, f_ans1, f_ans2):
        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)

        inst = image_to_context[ques['image']]

        if isinstance(inst['caption'], list):
            cap_str = '\n'.join(inst['caption'])
        else:
            cap_str = inst['caption']

        category = 'llava_bench_' + ques['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f"Visual QA category not found in rule file: {category}."
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Context]\n{cap_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        cur_js = {
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', ans2.get('question_id')),
            'category': category,
            'judge_model': args.model,
        }
        if idx >= len(cur_reviews):
            review = get_eval(args.model, content, args.max_tokens)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        idx += 1
        print(idx)
    review_file.close()
