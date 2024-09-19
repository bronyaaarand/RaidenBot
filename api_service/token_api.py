from flask import Flask, request, jsonify
from pymongo import MongoClient

app = Flask(__name__)

client = MongoClient("mongodb+srv://rinputin482:Rinputin482qh@cluster0.5n8iybx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["raiden"]
collection = db["const"]


@app.route('/access-token', methods = ['GET'])
def access_token():
    token_document = collection.find_one({"name": "auth_code"})

    if token_document and "value" in token_document:
        # Return the value of the auth_code
        return jsonify({"access_token": token_document["value"]})
    else:
        # If no token is found, return an error response
        return jsonify({"error": "auth_code not found"}), 404

if __name__ == "__main__":
    app.run(debug= True)