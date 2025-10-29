from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load both model and vectorizer
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return "✅ Fake News Detector API is running!!!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    # Transform text before prediction
    X_input = vectorizer.transform([text])
    prediction = model.predict(X_input)[0]
    
    result = 'Fake' if prediction == 1 else 'Real'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
