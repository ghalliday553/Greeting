from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
from io import BytesIO
from PIL import ImageFile
import pygame
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 500
model

def playMusic(prediction):
    pygame.mixer.music.load(prediction + ".wav")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue

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
        print(content_length)
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        image = Image.open(BytesIO(post_data))
        #scr.save("test.jpeg")
        resizedImg = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

        prediction = model.predict(resizedImg)
        playMusic(prediction)


def run(server_class=HTTPServer, handler_class=Serv, port=8080):
    server_address = ('', port)
    model = tf.keras.models.load_model('saved_model/my_model')
    pygame.mixer.init()
    httpd = server_class(server_address, handler_class)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

if __name__ == '__main__':
    run()

