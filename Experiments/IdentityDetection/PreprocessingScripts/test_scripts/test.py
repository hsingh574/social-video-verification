import insightface
import urllib
import urllib.request
import cv2
import numpy as np
from PIL import Image

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

#url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
#img = url_to_image(url)
img = np.asarray(Image.open('test_frame.jpg'))

model = insightface.app.FaceAnalysis()
ctx_id = 0
model.prepare(ctx_id = ctx_id, nms = 0.4)

faces = model.get(img)
for idx, face in enumerate(faces):
    print("Face [%d]:"%idx)
    print("\tage:%d"%(face.age))
    gender = 'Female' if face.gender == 0 else 'Male'
    print("\tgender:%s"%gender)
    print("\tembedding shape:%s"%face.embedding.shape)
    print("\tbbox:%s"%(face.bbox.astype(np.int).flatten()))
    print("")

