import os

answer_path = "data/00/dev.answer"
question_path = "data/00/dev.question"
context_path = "data/00/dev.context"
span_path = "data/00/dev.span"

assert os.path.exists(answer_path)

keyword = "microsoft"

contexts = open(context_path, mode='r', encoding='utf8').read().splitlines()
answers = open(answer_path, mode='r', encoding='utf8').read().splitlines()
questions = open(question_path, mode='r', encoding='utf8').read().splitlines()
spans = open(span_path, mode='r', encoding='utf8').read().splitlines()

indices = [i for i, context in enumerate(contexts) if keyword in context]
print(len(indices), " contexts")
new_context = [contexts[i] for i in indices]
new_answer = [answers[i] for i in indices]
new_question = [questions[i] for i in indices]
new_span = [spans[i] for i in indices]

with open("data/dev.answer", mode='w', encoding='utf8') as file:
    file.write("\n".join(new_answer))

with open("data/dev.question", mode='w', encoding='utf8') as file:
    file.write("\n".join(new_question))

with open("data/dev.context", mode='w', encoding='utf8') as file:
    file.write("\n".join(new_context))

with open("data/dev.span", mode='w', encoding='utf8') as file:
    file.write("\n".join(new_span))
