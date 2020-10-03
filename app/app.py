from flask import Flask, request
from classifier import ModelWrapper
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def index():
    return "Hello World"

@app.route("/predict_tweet", methods=['POST'])
def predict_tweet():
    req_data = request.get_json(force=True)
    print(req_data)

    if req_data is None or req_data['tweet'] is None or len(req_data['tweet']) == 0:
        return {'result':'bad tweet no good'},400

    result = model.transform_predict(req_data['tweet'])[0]
    return {"result": result}, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
