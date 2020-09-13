from modeling_bert import BertForQuestionAnswering
from tokenization_bert import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('model_path')
model = BertForQuestionAnswering.from_pretrained('bert_qa')

question, text = "what is your name?", "my name is nitish."
encoding = tokenizer.encode_plus(question, text)
input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
print(answer)
# assert answer == "a nice puppet"
