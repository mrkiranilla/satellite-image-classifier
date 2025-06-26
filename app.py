from flask import Flask, request, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# Set up the Flask application
app = Flask(__name__)
# Configure the upload folder. This is where uploaded images will be saved.
app.config['UPLOAD_FOLDER'] = 'uploads'

# ---------------------
# CNN model definition
# This is the same neural network structure you defined.
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------
# Load model + class names
device = torch.device('cpu')
model = CNN(num_classes=4)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Make sure this list is a valid Python list
classes = ['cloudy', 'desert', 'green_area', 'water']

# ---------------------
# Routes
# ---------------------

@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and prediction."""
    # This is the start of the function. All lines below must be indented.
    
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Ensure the 'uploads' directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image
    img = Image.open(filepath).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(img).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(1).item()
        conf = probs.squeeze().tolist()

    # --- THIS IS THE CORRECTED RESULT GENERATION BLOCK ---
    # All of this logic is now correctly indented inside the predict() function.

    predicted_class_name = classes[pred_class]
    confidence_percentage = conf[pred_class] * 100

    result = f"<h3>Analysis Complete: <span style='color: #28a745;'>{predicted_class_name.replace('_', ' ').title()}</span></h3>"
    result += f"<p>Confidence: <strong>{confidence_percentage:.2f}%</strong></p>"
    result += "<h4>Confidence Breakdown:</h4>"
    result += "<ul class='confidence-list'>"

    for cls, c in zip(classes, conf):
        percentage = c * 100
        # The multi-line f-string for each list item
        result += f"""
        <li>
            {cls.replace('_', ' ').title()}
            <div class="progress-bar">
                <div class="progress" data-width="{percentage:.2f}%" style="width: 0%;">
                    {percentage:.2f}%
                </div>
            </div>
        </li>
        """
    
    # This line must be AFTER the for-loop has finished.
    result += "</ul>"

    # This is the final return statement for the function.
    return result


# This block allows running the app directly with `python app.py` for testing
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
