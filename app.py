from flask import Flask, request, jsonify
from transformers import AutoConfig, AutoImageProcessor, ViTForImageClassification, BeitForImageClassification, BeitImageProcessor
from safetensors import safe_open
import torch
from PIL import Image
import io

app = Flask(__name__)

# Load skin preprocessor and model configurations
skin_config_path = "models/skin/config.json"
skin_preprocessor_config_path = "models/skin/preprocessor_config.json"
skin_model_weights_path = "models/skin/skin-model.safetensors"

skin_config = AutoConfig.from_pretrained(skin_config_path)
skin_processor = AutoImageProcessor.from_pretrained(skin_preprocessor_config_path)
skin_model = ViTForImageClassification(skin_config)
with safe_open(skin_model_weights_path, framework='pt') as f:
    skin_model.load_state_dict({
        k: f.get_tensor(k) for k in f.keys()
    })
skin_model.eval()

# Load breast model configurations
breast_model_config_path = 'models/breast/config.json'
breast_preprocessor_config_path = 'models/breast/preprocessor_config.json'
breast_model_weights_path = 'models/breast/breast-model.bin'

breast_preprocessor = BeitImageProcessor.from_pretrained(breast_preprocessor_config_path)
breast_model = BeitForImageClassification.from_pretrained('models/breast', config=breast_model_config_path)

breast_model.eval()

@app.route('/')
def home():
    if request.method == "POST":
        text = request.form.get('email-content')
    return 'Hello World!'

@app.route('/skin-predict', methods=['POST'])
def predict_skin():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    
    inputs = skin_processor(images=image, return_tensors='pt')
    
    with torch.no_grad():
        outputs = skin_model(**inputs)
        logits = outputs.logits
    
    predicted_class = logits.argmax(-1).item()
    label = skin_config.id2label[predicted_class]
    
    return jsonify({
        "label": label, 
        "class_id": predicted_class
    })

@app.route('/breast-predict', methods=['POST'])
def predict_breast():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    
    inputs = breast_preprocessor(images=image, return_tensors='pt')
    
    with torch.no_grad():
        outputs = breast_model(**inputs)
        logits = outputs.logits
    
    labels = {0: "benign", 1: "malignant"}
    predicted_class = logits.argmax(-1).item()
    label = labels.get(predicted_class, "unknown")
    return jsonify({
        "label": label, 
        "class_id": predicted_class
    })

if __name__ == '__main__':
    app.run(debug=True)