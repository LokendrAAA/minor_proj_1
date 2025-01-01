from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = Flask(__name__, static_folder='site/static', template_folder='site/templates')
CORS(app)

model_dir = "path_to_your_model_directory"  # Update this path
tokenizer = RobertaTokenizer.from_pretrained(model_dir)
model = RobertaForSequenceClassification.from_pretrained(model_dir)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_text', methods=['POST'])
def check_text():
    data = request.get_json()
    input_text = data.get('text', '')
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
    
    threshold = 0.6
    result = "AI-generated" if probabilities[0][1].item() > threshold else "Human-written"
    return jsonify({"result": result, "confidence": probabilities[0][1].item()})

if __name__ == '__main__':
    app.run(debug=True)

    
