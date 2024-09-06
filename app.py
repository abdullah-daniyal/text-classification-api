from flask import Flask, request, jsonify, render_template, url_for

app = Flask(__name__, template_folder='templates')
classifier = pipeline('sentiment-analysis')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']  # Changed to handle form data
    if not text:
        return render_template('index.html', result="No text provided.")
    
    result = classifier(text)
    response = f"Sentiment: {result[0]['label']} with a score of {result[0]['score']:.2%}"
    return render_template('index.html', result=response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
