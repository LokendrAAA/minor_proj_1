from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
from metrices import calculate_cpp_metrics

app = Flask(__name__, static_folder='site/static', template_folder='site/templates')
CORS(app)

# Configure Gemini API
gemini_api_key = ""  # Replace with your actual Gemini API key
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-pro")

# Load fine-tuned models and tokenizers for text and code analysis
text_model_dir = "./path_to_your_model_directory"  # Update with your actual text model path
code_model_dir = "./fine_tuned_codebert"  # Update with your actual code model path

# Load models and tokenizers
text_tokenizer = RobertaTokenizer.from_pretrained(text_model_dir)
text_model = RobertaForSequenceClassification.from_pretrained(text_model_dir)
text_model.eval()

code_tokenizer = AutoTokenizer.from_pretrained(code_model_dir)
code_model = AutoModelForSequenceClassification.from_pretrained(code_model_dir)
code_model.eval()

@app.route('/')
def index():
    return render_template('index1.html')

# Text Check using Normal Model
@app.route('/check_text_model', methods=['POST'])
def check_text_model():
    data = request.get_json()
    input_text = data.get('text', '')

    def split_text_into_chunks(text, word_limit):
        words = text.split()
        return [' '.join(words[i:i + word_limit]) for i in range(0, len(words), word_limit)]

    # Split input text into chunks
    chunks = split_text_into_chunks(input_text, 10)  # 10 words per chunk
    line_results = []
    ai_count = 0
    human_count = 0

    for chunk in chunks:
        inputs = text_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = text_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

        threshold = 0.6
        normal_result = "AI-generated" if probabilities[0][1].item() > threshold else "Human-written"
        confidence = probabilities[0][1].item()

        # Count classifications for overall result
        if normal_result == "AI-generated":
            ai_count += 1
        else:
            human_count += 1

        line_results.append({
            'result': normal_result,
            'confidence': confidence
        })

    # Determine overall result
    if ai_count > human_count:
        overall_result = "AI-generated"
    elif human_count > ai_count:
        overall_result = "Human-written"
    else:
        overall_result = "Mixed Content"

    return jsonify({
        'line_results': line_results,
        'overall_result': overall_result
    })


# Text Check using Gemini API
@app.route('/check_text_api', methods=['POST'])
def check_text_api():
    data = request.get_json()
    input_text = data.get('text', '')

    try:
        gemini_response = gemini_model.generate_content(
            f"Analyze the following text and determine if it is human-written or AI-generated and just tell it in one word answer:\n\n{input_text}"
        )
        gemini_result = gemini_response.text.strip()
    except Exception as e:
        gemini_result = f"Error with Gemini API: {e}"

    return jsonify({
        'gemini_result': gemini_result
    })

# Code Check using Normal Model
@app.route('/check_code_model', methods=['POST'])
def check_code_model(): 
    data = request.get_json()
    code = data.get('code', '')
    metrics = calculate_cpp_metrics(code)

    inputs = code_tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = code_model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    normal_result = "Human-Written Code" if predicted_label == 0 else "AI-Generated Code"

    return jsonify({
        "normal_result": normal_result,
        "metrics": metrics
    })

# Code Check using Gemini API
@app.route('/check_code_api', methods=['POST'])
def check_code_api():
    data = request.get_json()
    code = data.get('code', '')

    try:
        gemini_response = gemini_model.generate_content(
            f"Analyze the following code and determine if it is human-written or AI-generated :\n\n{code}"
        )
        gemini_result = gemini_response.text.strip()
    except Exception as e:
        gemini_result = f"Error with Gemini API: {e}"

    return jsonify({
        "gemini_result": gemini_result
    })

if __name__ == '__main__':
    app.run(debug=True)
