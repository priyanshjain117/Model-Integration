from flask import Flask, request, render_template, jsonify
import pickle
import re

# -------------------------------------------------
# 🔹 Load model and vectorizer
# -------------------------------------------------
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except:
    print("Error loading model files")
    model = None
    vectorizer = None

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
# 🔹 Text analysis function for UI indicators
# -------------------------------------------------
def analyze_text_indicators(text):
    """Analyze text for fake news indicators"""
    # Check for clickbait keywords
    clickbait_patterns = r'shocking|unbelievable|you won\'t believe|doctors hate|one trick|breaking|urgent|must see'
    has_clickbait = bool(re.search(clickbait_patterns, text, re.IGNORECASE))
    
    # Check for excessive punctuation
    exclamations = len(re.findall(r'!', text))
    has_exclamations = exclamations > 3
    
    # Check for all caps words
    has_capitals = bool(re.search(r'\b[A-Z]{4,}\b', text))
    
    # Check for source citations
    source_patterns = r'according to|study shows|research|source:|reported by|says|claims'
    has_sources = bool(re.search(source_patterns, text, re.IGNORECASE))
    
    return {
        'has_clickbait': has_clickbait,
        'has_exclamations': has_exclamations,
        'has_capitals': has_capitals,
        'has_sources': has_sources
    }

# -------------------------------------------------
# 🔹 Routes
# -------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input text from form
        text = request.form.get('news_text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Clean and transform text
        cleaned_text = clean_text(text)
        vect_text = vectorizer.transform([cleaned_text])
        
        # Predict label and probability
        proba = model.predict_proba(vect_text)[0]
        prediction = model.predict(vect_text)[0]
        
        # Get probability of being Real (class 1) and Fake (class 0)
        prob_fake = float(proba[0] * 100)
        prob_real = float(proba[1] * 100)
        
        # Analyze text for additional indicators
        indicators_data = analyze_text_indicators(text)
        
        # Calculate credibility score (0-100)
        credibility_score = int(prob_real)
        
        # Adjust based on text indicators
        if indicators_data['has_clickbait']:
            credibility_score = max(0, credibility_score - 15)
        if indicators_data['has_exclamations']:
            credibility_score = max(0, credibility_score - 10)
        if indicators_data['has_capitals']:
            credibility_score = max(0, credibility_score - 5)
        if indicators_data['has_sources']:
            credibility_score = min(100, credibility_score + 10)
        
        # Determine status
        if credibility_score >= 70:
            status = 'reliable'
        elif credibility_score >= 40:
            status = 'questionable'
        else:
            status = 'unreliable'
        
        # Build indicators for UI
        indicators = {
            'emotional': 'High' if (indicators_data['has_clickbait'] or indicators_data['has_exclamations']) else 'Low',
            'sources': 'Present' if indicators_data['has_sources'] else 'Missing',
            'bias': 'High' if credibility_score < 50 else 'Moderate' if credibility_score < 70 else 'Low',
            'factCheck': 'Verified' if credibility_score > 70 else 'Unverified' if credibility_score > 40 else 'Disputed'
        }
        
        # Build recommendations based on status
        if status == 'unreliable':
            recommendations = [
                'Verify this information with trusted news sources',
                'Look for original sources and citations',
                'Check if other reputable outlets are reporting this story',
                'Be extremely cautious before sharing this content'
            ]
        elif status == 'questionable':
            recommendations = [
                'Cross-reference with multiple reliable sources',
                'Look for expert opinions on this topic',
                'Be cautious about sharing until verified',
                'Check the publication date and author credentials'
            ]
        else:
            recommendations = [
                'This content appears credible based on initial analysis',
                'Still recommended to verify important claims independently',
                'Check publication date for currency of information',
                'Consider the source reputation and bias'
            ]
        
        # Prepare response
        result_data = {
            'score': credibility_score,
            'status': status,
            'indicators': indicators,
            'recommendations': recommendations,
            'model_prediction': 'REAL' if prediction == 1 else 'FAKE',
            'confidence_real': round(prob_real, 2),
            'confidence_fake': round(prob_fake, 2)
        }
        
        return jsonify(result_data)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# -------------------------------------------------
# 🔹 Run app
# -------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)