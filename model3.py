from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__, static_folder='site/static', template_folder='site/templates')
CORS(app)

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
    return render_template('index.html')

@app.route('/check_text', methods=['POST'])
def check_text():
    data = request.get_json()
    input_text = data.get('text', '')

    # Tokenize and predict text
    inputs = text_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
    
    # Set threshold and determine result
    threshold = 0.6
    result = "AI-generated" if probabilities[0][1].item() > threshold else "Human-written"
    return jsonify({"result": result, "confidence": probabilities[0][1].item()})

@app.route('/check_code', methods=['POST'])
def check_code():
    data = request.get_json()
    code = data.get('code', '')

    # Tokenize and predict code
    inputs = code_tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = code_model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    # Interpret prediction result
    result = "Human-Written Code" if predicted_label == 0 else "AI-Generated Code"
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
