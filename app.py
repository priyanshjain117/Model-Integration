from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your model
model = pickle.load(open('fake_news_model.pkl', 'rb'))

@app.route('/')
def home():
    return "Fake News Detector API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    
    # Preprocess (make sure same preprocessing as training)
    # For example:
    # text = preprocess(text)
    
    prediction = model.predict([text])[0]
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
