from flask import Flask

from flask import request, redirect, url_for,flash
import logging
from logging.handlers import RotatingFileHandler
import os
from werkzeug.utils import secure_filename
from io import BufferedReader

import cv2
import numpy as np
import pytesseract 
import re
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
from flask_cors import CORS, cross_origin
app = Flask(__name__)

CORS(app)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


UPLOAD_FOLDER ="./uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def hello():
    file = request.files['file']
    if file:
        print(file.filename)
        file.save(file.filename)
    else:
        print('IMAGE NOT SENT')


	# Image Process starts here!


	# Load the image
    image = cv2.imread('cheque1.png')
    image = cv2.resize(image, (1100, 500)) 
	# Grayscale the image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)



	# Thresholding 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(hsv,255,cv2.ADAPTIVE_THRESH_MEAN_C,            cv2.THRESH_BINARY,11,2)
    res,mask = cv2.threshold(hsv, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)




    cv2.imwrite('result.jpg',mask)


	# In[13]:


	# erosion
    mask = cv2.bitwise_not(mask)
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 10)
	#erosion = cv2.bitwise_not(erosion)


	# In[14]:


    cv2.imwrite('result.jpg',erosion)


	# In[15]:


    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


	# In[16]:


    i = 0
    f = open("result_ocr.txt", "w",encoding="utf-8")
    for c in contours:
        i = i + 1
        if i == 1:
            continue    
        x,y,w,h = cv2.boundingRect(c)
	    #print(w)

        ar = w / float(h)
        if True:
            cropped = mask[y:y+h, x:x+w]
            text = pytesseract.image_to_string(cropped)
            if not text == ' ':
                f.write(text)
                f.write('\n')
	        #cv2.imshow('img', cropped)
            cv2.waitKey(0)
    f.close()        


	# In[179]:


    f.close()


    file_id = 'result_ocr'


    f = open("result_ocr.txt", "r",encoding="utf-8")

    lines = f.read().split('\n')

    rib_value = False
    n_cheq_value= False
    rib_found = False
    for line in lines:
        if not rib_found:
            rib = re.sub(r" ", "", line, flags=re.I)
            rib_value = ''
            for c in rib:
                if c.isdigit():
                    rib_value += c
            if len(rib_value) == 20:
                rib_found = True
                continue
            rib_match = re.match(pattern='.*TND$', string= line)
            if rib_match:
                rib_found = True
                rib = re.sub('TND','',line)
                rib = re.sub(r" ", "", rib, flags=re.I)
                rib_value = ''
                for c in rib:
                    if c.isdigit():
                        rib_value += c
                if len(rib_value) == 20:
                    rib_found = True
                else:
                    if len(rib_value) > 20 :
                        diff = len(rib_value) - 20
                        rib_value = rib_value[diff:]
                    else:
                        rib_found = False

        if line.isnumeric():
            if len(line) == 7 :
                n_cheq_value = line
    print("RIB VALUE: "+str(rib_value))
    print("NUM CHEQUE: "+str(n_cheq_value))


    dict1 = {"rib": str(rib_value), "num": str(n_cheq_value)}
    return dict1

if __name__ == "__main__":
    app.run(debug=True) 