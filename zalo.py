import requests as req
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel
from googletrans import Translator

intents_labels = {
    'technical': 0,
    'statistic': 1,
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
    
loaded_token = AutoTokenizer.from_pretrained("raiden_intent\\intent_model", local_files_only=True)
loaded_model = RoBertaIntentsClassifier(ro_model, 3).to("cpu")
state_dict = torch.load(f"raiden_intent\\intent_model\\raiden_itents_classifier_model.pth", map_location=torch.device('cpu'))
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


uri_get_message = 'https://openapi.zalo.me/v2.0/oa/listrecentchat?data={"offset":0,"count":1}'
uri_post_message = 'https://openapi.zalo.me/v3.0/oa/message/cs'
uri_dify_api = 'https://api.dify.ai/v1/completion-messages'

auth = {
    "access_token": "k4fB48yXaLUtIqqUZosMKRP7JnsBEieXzpjrCSfQf3gsO20Jz1wC8xGeB2c86gy6yHyWCR8wiMFMEp1GxWIRU8PY4d7z1RnVkXHb6T82vWgU9tmRtX7Y0fmtS0_d1QagYsun5FTEaZcLB2yjzKYq09joEZNsLeWCr4u5HffkgGxJMp19ktsISED5E7MvNBrjXdeYO-04iaojBZ9xzodgDPiqLJBy2VaEf2bNNE4KsXg7949GopFDVf8qG7NnN-0vk7blQ-Xzld-hQsTFwL3xU81WSad_PEDUga4yKFzGuto7MbnInoJoLAq2Q47O0jDTxZ03SP8tXcMgCHn3v0UpPO1j8JVEUliWba9H9_ucpWY31q52vMB7MvfrI6hRRUL2bdbcQzDxlN--R6PE-s3AQPLGKcPfGWzJw7-3Dv9N"
}

def main_intent_classification(customer_request):
    customer_intent = predict_intents(customer_request, translator, loaded_model, loaded_token, intents_labels)
    return customer_intent

def check_message_event():
    previous_response = ""
    previous_bot = ""
    while True:
        try:
            res = req.get(uri_get_message, headers=auth)
            res.raise_for_status()

            message_data = res.json()['data'][0]
            current_message = message_data['message']

            if current_message != previous_response and current_message != previous_bot:
                print("Received a new message:")
                print(current_message)
                previous_response = current_message

                dify_body = {    
                    "inputs": {"question": current_message},
                    "response_mode": "blocking",
                    "user": "abc-123"
                }

                dify_auth = {
                    'Authorization': 'Bearer app-FGzgUscEa9ejglfb0HhnfHnw'
                }

                dify_response = req.post(uri_dify_api, json=dify_body, headers=dify_auth)

                current_bot = dify_response.json()["answer"]

                print(current_bot)

                previous_bot = current_bot

                body = {
                    "recipient": {
                        "user_id": "1696434873920451916"
                    },
                    "message": {
                        "text": current_bot
                    }
                }
                pos_res = req.post(uri_post_message, json=body, headers=auth)

        except req.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

        time.sleep(2)

check_message_event()
