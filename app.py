import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil

# Database connection
connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

# Create user table if it doesn't exist
command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = ? AND password = ?"
        cursor.execute(query, (name, password))

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided, Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES (?, ?, ?, ?)", (name, password, mobile, email))
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/userlog.html')
def demo():
    return render_template('userlog.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        # Use the absolute path for the images directory
        dirPath = r"C:\Users\GC GAURAV\OneDrive\Desktop\LUNG_CANCER (2)\LUNG_CANCER\static\images"
        
        # Check if the directory exists; if not, create it
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        # Clear the directory
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(os.path.join(dirPath, fileName))
        
        fileName = request.form['filename']
        src_path = os.path.join("test", fileName)
        
        # Check if the source file exists before copying
        if not os.path.exists(src_path):
            return render_template('index.html', msg='File not found. Please check the filename and try again.')

        dst = dirPath
        
        # Copy the file
        shutil.copy(src_path, dst)
        image = cv2.imread(src_path)
        
        # Color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        
        # Apply the Canny edge detection
        edges = cv2.Canny(image, 100, 200)
        cv2.imwrite('static/edges.jpg', edges)
        
        # Apply thresholding to segment the image
        retval2, threshold2 = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        
        # Create the sharpening kernel
        kernel_sharpening = np.array([[-1, -1, -1],
                                       [-1, 9, -1],
                                       [-1, -1, -1]])

        # Apply the sharpening kernel to the image
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)
        cv2.imwrite('static/sharpened.jpg', sharpened)

        def segment_tumor(image, lower_gray_threshold, upper_white_threshold, pixel_to_mm_conversion):
            # Thresholding to segment the tumor
            mask = cv2.inRange(image, lower_gray_threshold, upper_white_threshold)
            
            # Find contours in the binary image
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create a mask for the tumor area
            tumor_mask = np.zeros_like(image)
            cv2.drawContours(tumor_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
            
            # Overlay the tumor mask on the original image
            tumor_area_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            tumor_area_image[tumor_mask == 255] = [0, 255, 0]  # Mark tumor area in green
            
            # Calculate area of the tumor
            tumor_area_pixel = sum(cv2.contourArea(contour) for contour in contours)
            tumor_area_mm = tumor_area_pixel * pixel_to_mm_conversion ** 2
                 
            # Calculate the diameter of the tumor
            _, radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
            tumor_diameter_pixel = 2 * radius
            tumor_diameter_mm = tumor_diameter_pixel * pixel_to_mm_conversion
            
            return tumor_area_mm, tumor_diameter_mm, tumor_area_image

        # Adjust these threshold values based on the characteristics of your image
        lower_gray_threshold = 150
        upper_white_threshold = 200

        # Pixel to millimeter conversion factor
        pixel_to_mm_conversion = 0.1  # Example conversion factor, adjust according to your image

        tumor_area_mm, tumor_diameter_mm, tumor_area_image = segment_tumor(gray_image, lower_gray_threshold, upper_white_threshold, pixel_to_mm_conversion)

        print("Area of Tumor:", tumor_area_mm, "mm")
        print("Diameter of Tumor:", tumor_diameter_mm, "mm")
        cv2.imwrite('static/tumor.png', tumor_area_image)

        verify_dir = dirPath
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'Lungcancer-{}-{}.model'.format(LR, '2conv-basic')

        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()

        tf.compat.v1.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 4, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')

        fig = plt.figure()
        
        str_label = " "
        accuracy = ""
        rem = ""
        rem1 = ""
        for num, data in enumerate(verify_data):
            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            model_out = model.predict([data])[0]

            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = "Adenocarcinoma"
                accuracy = "The predicted image of Adenocarcinoma is with an accuracy of {}%".format(model_out[0] * 100)
                rem = "The treatment for Adenocarcinoma includes:\n\n "
                rem1 = ["Surgery: Often the primary treatment for early stages of adenocarcinoma of the lung.",
                        "Chemotherapy and radiation therapy.",
                        "Targeted therapies or immunotherapy in advanced stages."]
            elif np.argmax(model_out) == 1:
                str_label = "Squamous cell carcinoma"
                accuracy = "The predicted image of Squamous cell carcinoma is with an accuracy of {}%".format(model_out[1] * 100)
                rem = "The treatment for Squamous cell carcinoma includes:\n\n"
                rem1 = ["Surgical removal of the tumor.",
                        "Chemotherapy or radiation therapy.",
                        "Immunotherapy in some cases."]
            elif np.argmax(model_out) == 2:
                str_label = "Large cell carcinoma"
                accuracy = "The predicted image of Large cell carcinoma is with an accuracy of {}%".format(model_out[2] * 100)
                rem = "The treatment for Large cell carcinoma includes:\n\n"
                rem1 = ["Surgery in early stages.",
                        "Chemotherapy or radiation therapy in advanced stages."]
            elif np.argmax(model_out) == 3:
                str_label = "Small cell carcinoma"
                accuracy = "The predicted image of Small cell carcinoma is with an accuracy of {}%".format(model_out[3] * 100)
                rem = "The treatment for Small cell carcinoma includes:\n\n"
                rem1 = ["Chemotherapy, radiation therapy, and immunotherapy."]
            print(str_label)

            plt.imshow(orig)
            plt.title('Prediction: {}'.format(str_label))
            plt.xlabel(accuracy)

        # Save the output image
        if not os.path.exists('static'):
            os.makedirs('static')
        plt.savefig('static/output.jpg')
        plt.close(fig)  # Close the figure to free up memory

        return render_template('results.html', rem1=rem1, rem=rem, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)