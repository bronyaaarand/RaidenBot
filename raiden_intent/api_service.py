# from classification import translator, loaded_model, loaded_token, intents_labels, predict_intents
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route('/customer-intent', methods = ['POST'])
# def intent_classification():
#     data = request.get_json()
#     customer_request = data["customer_message"] if data["customer_message"] else "No idea"
#     customer_intent = predict_intents(customer_request, translator, loaded_model, loaded_token, intents_labels)
#     return jsonify(
#         {
#             'customer_intent': customer_intent
#         }
#     )

# if __name__ == "__main__":
#     app.run(debug= True)