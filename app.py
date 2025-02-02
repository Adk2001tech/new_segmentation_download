  
import os
import numpy as np

from flask import Flask, render_template, url_for, request, send_file  #############NEW send_file ################################
from werkzeug.utils import secure_filename


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from PIL import Image, ImageFile
import matplotlib.pyplot as plt

from io import BytesIO
import base64

import load_net

app= Flask(__name__)
img=None
g_list= []


@app.route('/')
def home():
    return render_template('home.html')
  
  
@app.route('/Prediction', methods = ['GET', 'POST'])
def pred():
    global img, g_list
    if request.method=='POST':
         file = request.files['file']
         org_img, feed_img= load_net.process_file(file)
         img = load_net.out_via_model_sm(feed_img)
         g_list.append(img)
         img_x=BytesIO()
         plt.imshow(img)
         plt.savefig(img_x,format='png')
         plt.close()
         img_x.seek(0)
         plot_url1=base64.b64encode(img_x.getvalue()).decode('utf8')

    return render_template('pred.html',  plot_url1=plot_url1)


@app.route('/download', methods = ['GET', 'POST'])
def download():
    arr= g_list[0].copy()
    arr=arr*255.0
    img = Image.fromarray(arr.astype('uint8'))
    file_object= BytesIO()
    img.save(file_object, format='png')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/PNG',attachment_filename='pic.png',  as_attachment=True)

        

if __name__=='__main__':
    app.run(debug=True)
