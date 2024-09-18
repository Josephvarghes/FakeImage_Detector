# venv
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import threading
import pyttsx3
from colorthief import ColorThief

UPLOAD_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model from the SavedModel format directory
# model_path = r"D:\Revature\P1\28-8-24\training_fake\easy_14_1111.jpg"
model = tf.keras.models.load_model("model_final.h5")

def speak(text):
    def speak_thread():
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        
        engine.say(text)
        engine.runAndWait()
    
    thread = threading.Thread(target=speak_thread)
    thread.start()

def extract_colors(image_path):
    color_thief = ColorThief(image_path)
    dominant_color = color_thief.get_color(quality=1)
    
    # Map RGB to color names
    colors = {
        'Red': (255, 0, 0),
        'Green': (0, 255, 0),
        'Blue': (0, 0, 255),
        'Black': (0, 0, 0),
        'White': (255, 255, 255),
        'Gray': (128, 128, 128),
        'Yellow': (255, 255, 0),
        'Cyan': (0, 255, 255),
        'Magenta': (255, 0, 255),
        'Orange': (255, 165, 0),
        'Purple': (128, 0, 128),
        'Pink': (255, 192, 203),
        'Brown': (165, 42, 42),
        'Fair': (245, 245, 220)  # Consider this as a fair skin tone
    }

    # Find the closest named color
    def closest_color(requested_color):
        min_colors = {}
        for key, name in colors.items():
            r_c, g_c, b_c = name
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colors[rd + gd + bd] = key
        return min_colors[min(min_colors.keys())]
    
    main_color_name = closest_color(dominant_color)
    return main_color_name

def classify_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    class_label = 1 if prediction[0] > 0.5 else 0
    class_name = 'Real' if class_label == 1 else 'Fake'
    
    # Extract dominant color
    main_color = extract_colors(image_path)
    
    description = f"This image is {class_name}. The dominant color is {main_color}."
    return description

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filename = filename.replace(" ", "_")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result = classify_image(filepath)
            speak(result)  # Use text-to-speech to say the description
            
            return render_template('result.html', result=result, filename=filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
