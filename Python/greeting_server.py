from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
from io import BytesIO
from PIL import ImageFile
import tensorflow as tf
import pygame
import cv2
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 500
global model
people = ["Juan","Graeme","Grayson","Liam","Spencer"]

def playMusic(prediction):
    print(prediction)
    for index, person in np.ndenumerate(prediction):
        if person > 0.95:
            global people
            print(index)
            pygame.mixer.music.load("/home/pi/Documents/Greeting/Greeting/Audio/" + people[index[0]] + ".mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() == True:
                continue
            break

class Serv(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        print("post")
        
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        data_array = np.frombuffer(post_data, dtype='uint8')
        image = cv2.imdecode(data_array, cv2.IMREAD_COLOR)
        
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(image, 1.01, 8, minSize=(800, 800))
        print(faces)
        face_crop = []
        for f in faces:
            x, y, w, h = [ v for v in f ]
            y-=500
            x-=500
            w+=500
            h+=500
            face_crop.append(image[y:y+h, x:x+w])
        
        if (np.size(face_crop) == 0):
            self._set_response()
            self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))
            return
        
        
        resizedImage = cv2.resize(face_crop[0], (IMG_SIZE,IMG_SIZE))
        cv2.imshow('resizedImage', resizedImage)
        cv2.waitKey(0)
        
        open_cv_image = np.array(resizedImage)
        imageArr = np.array(open_cv_image).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

        self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))
        
        global model
        prediction = model.predict(imageArr)
        print(prediction)
        playMusic(prediction[0])


def run(server_class=HTTPServer, handler_class=Serv, port=8080):
    server_address = ('', port)
    global model
    model = tf.keras.models.load_model('greeting_model')
    pygame.mixer.init()
    httpd = server_class(server_address, handler_class)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

if __name__ == '__main__':
    run()

