from flask import Flask, request, render_template, jsonify
import pickle
import re

# -------------------------------------------------
# 🔹 Load model and vectorizer
# -------------------------------------------------
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# -------------------------------------------------
# 🔹 Initialize Flask app
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# 🔹 Text cleaning function (same as training preprocessing)
# -------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)     # remove URLs
    text = re.sub(r'<.*?>', '', text)              # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)           # remove punctuation and digits
    text = re.sub(r'\s+', ' ', text).strip()       # remove extra spaces
    return text

# -------------------------------------------------
# 🔹 Routes
# -------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from form or JSON
    text = request.form.get('news_text') or request.json.get('news_text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Clean and transform text
    cleaned_text = clean_text(text)
    vect_text = vectorizer.transform([cleaned_text])

    # Predict probabilities and label
    proba = model.predict_proba(vect_text)[0][1]  # Probability of being "Real"
    result = 'Real News ✅' if proba > 0.6 else 'Fake News ❌'

    # Return JSON if API call, else render HTML result
    if request.is_json:
        return jsonify({'prediction': result, 'probability': round(float(proba), 3)})
    else:
        return render_template('index.html', prediction=result, prob=round(float(proba), 3))

# -------------------------------------------------
# 🔹 Run app
# -------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
