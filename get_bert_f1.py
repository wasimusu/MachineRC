import json

from data_util.evaluate import f1_score

PREDICTIONS_FILE_NAME = './bert_test/predictions.json'
DEV_DATA_FILE_NAME = './data/dev-v1.1.json'

with open(DEV_DATA_FILE_NAME) as json_file:
    sets = json.load(json_file)['data']

id_answer_text_dict = {}
print('Loading dev data')
for topic_set in sets:
    for paragraph in topic_set['paragraphs']:
        for qas in paragraph['qas']:
            id_answer_text_dict[qas['id']] = qas['answers'][0]['text']

print('Loading predictions')
with open(PREDICTIONS_FILE_NAME) as json_file:
    predictions = json.load(json_file)

count = 0
f1 = 0
for qas_id, prediction in predictions.items():
    ground_truth = id_answer_text_dict[qas_id]
    f1 += f1_score(prediction, ground_truth)
    count += 1

f1 = f1 / (count + 1)

# Print the f1 score to console
print('Computed f1: ', f1)
