""" Get all the text in the squad in single file"""

from squad import Squad

s = Squad(train=True, root="data", download=False)
text = [context for context, qas in s]
text = "\n".join(text)

with open("squad-base", mode='w', encoding='utf8') as file:
    file.write(text)
