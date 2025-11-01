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
# 🔹 Text cleaning function (same preprocessing as training)
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
    text = request.form.get('news_text') or (request.json.get('news_text') if request.is_json else None)
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Clean and transform text
    cleaned_text = clean_text(text)
    vect_text = vectorizer.transform([cleaned_text])

    # Predict label and probability
    proba = model.predict_proba(vect_text)[0][1]  # Probability of being "Real"
    prediction = model.predict(vect_text)[0]

    # Correct label meaning: 1 = Real, 0 = Fake
    if prediction == 1:
        result = "✅ This news article seems REAL."
    else:
        result = "🚨 This news article is likely FAKE!"

    # Web or JSON response
    if request.is_json:
        return jsonify({
            'prediction': result,
            'probability_real': round(float(proba), 3)
        })
    else:
        return render_template('index.html', prediction=result, prob=round(float(proba), 3), news_text=text)

# -------------------------------------------------
# 🔹 Run app
# -------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
