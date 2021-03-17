from flask import Flask,render_template,request,flash,url_for,redirect
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image


app=Flask(__name__)

UPLOAD_FOLDER='/home/dhananjaya/flask_app/venv/apple/uploads'
app.secret_key="secret_key"
app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER

model=load_model('apple.h5')

def model_predict(img_path,model):
    test_image=image.load_img(img_path,target_size=(224,224))
    test_image=image.img_to_array(test_image)
    test_image=test_image/255
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods=['POST','GET'])
def submit_file():
    if request.method=='POST':
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)

        file=request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename=secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

            result = model_predict(filename, model)

            categories = ['Healthy', 'Multiple Disease', 'Rust', 'Scab']

            pred_class = result.argmax()
            output = categories[pred_class]
            return output
        return None


if __name__=='__main__':
    app.run(debug=True)