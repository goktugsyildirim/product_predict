from flask import Flask, request, jsonify, render_template
from src.inference import predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    description = request.form.get('description')
    if not description:
        return render_template('index.html', prediction='No description provided', description='')
    category = predict(description)
    return render_template('index.html', prediction=category, description=description)

@app.route('/api/predict', methods=['POST'])
def api_predict_route():
    data = request.json
    description = data.get('description', None)
    if not description:
        return jsonify({'error': 'No description provided'}), 400
    category = predict(description)
    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
