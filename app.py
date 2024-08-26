import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session
from werkzeug.utils import secure_filename
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import logging
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model
model_path = os.getenv("MODEL_PATH", "my_model.keras")
model = load_model(model_path)
logging.info('Model loaded. Check http://127.0.0.1:5000/')

# Load the language model
groqllm = ChatGroq(model="llama3-8b-8192", temperature=0)
prompt = """(system: You are a crop assistant specializing in agriculture. If the user's question is related to agriculture, provide a detailed and helpful response. If the question is unrelated to agriculture, respond with "I'm sorry, I can only assist with agriculture-related queries.)
(user: Question: {question})"""
promptinstance = ChatPromptTemplate.from_template(prompt)

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

# In-memory storage for users (Not for production use)
users = {}

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Authenticate user
        user = users.get(email)
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = email  # Store user ID (email) in session
            flash('', 'success')
            return redirect(url_for('agrocare')) 
        else:
            flash('Invalid email or password. Please try again.', 'danger')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)  # Hash the password
        users[email] = {'password': hashed_password}
        
        flash('Sign up successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/agrocare')
def agrocare():
    if 'user_id' not in session:  # Check if user is logged in
        flash('You need to log in to access this page.', 'danger')
        return redirect(url_for('login'))
    return render_template('agrocare.html')

@app.route('/speech')
def speech():
    if 'user_id' not in session:  # Check if user is logged in
        flash('You need to log in to access this page.', 'danger')
        return redirect(url_for('login'))
    return render_template('speech.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    logging.info(f"Received question: {question}")
    try:
        response = promptinstance | groqllm | StrOutputParser()
        answer = response.invoke({'question': question})

        formatted_answer = format_answer(answer)
        logging.info(f"Response generated: {formatted_answer}")
        return jsonify({'answer': formatted_answer})
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return jsonify({'answer': f'Error processing your request: {str(e)}'}), 500

def format_answer(answer):
    answer = answer.replace("**", "<strong>").replace("**", "</strong>")
    formatted_answer = "<div style='text-align: left;'>"
    lines = answer.split('\n')
    for line in lines:
        if line.strip():
            formatted_answer += f"<p>{line.strip()}</p>"
    formatted_answer += "</div>"
    return formatted_answer

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'prediction': 'No image uploaded'}), 400

    f = request.files['file']
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    file_path = os.path.join(uploads_dir, secure_filename(f.filename))
    f.save(file_path)
    try:
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return jsonify({'prediction': predicted_label})
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({'prediction': f'Error processing image: {str(e)}'}), 500

@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user from session
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
