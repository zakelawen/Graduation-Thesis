import os
import time
SYSTEM_PROMPT = "You are a precise evaluator. Your task is to determine if a 'Generated Answer' is semantically identical to any of the 'Ground Truth' answers for a multiple-choice question. They are identical if they refer to the same option, even with minor formatting differences (e.g., 'A)' vs 'A.' or containing extra text like 'The answer is...'). Respond with only the single word 'yes' or 'no', and nothing else."

def make_client():
    import httpx
    from openai import OpenAI
    base_url = os.environ.get('OPENAI_BASE_URL') or os.environ.get('OPENAI_API_BASE')
    return OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=base_url, http_client=httpx.Client(base_url=base_url, follow_redirects=True) if base_url else None)

def judge(client, question, ground_truths, answer, model='gpt-4o-mini', retries=3) -> int:
    user_prompt = f'Question: {question}\nGround Truths: {ground_truths}\nGenerated Answer: {answer}'
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user_prompt}], temperature=0, max_tokens=5)
            return 1 if resp.choices[0].message.content.strip().lower() == 'yes' else 0
        except Exception:
            time.sleep(2 * (attempt + 1))
    return -1
