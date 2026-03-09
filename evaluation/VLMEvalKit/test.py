from vlmeval.api import OpenAIWrapper
model = OpenAIWrapper('gpt-4.1-2025-04-14', verbose=True)
msgs = [dict(type='text', value='Hello!')]
code, answer, resp = model.generate_inner(msgs)
print(code, answer, resp)
