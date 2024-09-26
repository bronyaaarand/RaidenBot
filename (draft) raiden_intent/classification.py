import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from transformers import AutoTokenizer, RobertaModel
from googletrans import Translator

intents_labels = {
    'technical': 0,
    'stat': 1,
    'contact': 2,
}

ro_model = RobertaModel.from_pretrained('roberta-base')

class RoBertaIntentsClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(RoBertaIntentsClassifier, self).__init__()
        self.bert = ro_model
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        x = self.relu1(pooled_output)
        x = self.relu2(x)
        logits = self.fc(x)
        return logits
    
loaded_token = AutoTokenizer.from_pretrained("intent_model", local_files_only=True)
loaded_model = RoBertaIntentsClassifier(ro_model, 3).to("cpu")
state_dict = torch.load(f"intent_model\\raiden_itents_classifier_model.pth", map_location=torch.device('cpu'))
loaded_model.load_state_dict(state_dict)

translator = Translator()

def predict_intents(request, translator, model, tokenizer, intents_labels):
    sentence = translator.translate(request, src='vi', dest='en').text
    encoded_sentence = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(encoded_sentence['input_ids'], encoded_sentence['attention_mask'])
        predicted_label = torch.argmax(logits, dim=1).item()

    predicted_intent = [k for k, v in intents_labels.items() if v == predicted_label][0]

    return predicted_intent

# example
# cus_request = "Hãy cho tôi thông tin thống kê về cuộc gọi tháng 3"
# customer_intent = predict_intents(cus_request, translator, loaded_model, loaded_token, intents_labels)
# print(f"Customer intent is: {customer_intent}")