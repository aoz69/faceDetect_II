from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN


# confirm mtcnn was installed correctly
import mtcnn
print(mtcnn.__version__)
# load image from file
pixels = pyplot.imread("./images/857/0.png")

# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
results = detector.detect_faces(pixels)

# extract the bounding box from the first face
x1, y1, width, height = results[0]['box']
x2, y2 = x1 + width, y1 + height

# extract the face
face = pixels[y1:y2, x1:x2]

# resize pixels to the model size
image = Image.fromarray(face)
image = image.resize((224, 224))
face_array = asarray(image)

# extract a single face from a given photograph
def extract_face(filename):
 # load image from file
 pixels = pyplot.imread(filename)
 # create the detector, using default weights
 detector = MTCNN()
 # detect faces in the image
 results = detector.detect_faces(pixels)
 # extract the bounding box from the first face
 x1, y1, width, height = results[0]['box']
 x2, y2 = x1 + width, y1 + height
 # extract the face
 face = pixels[y1:y2, x1:x2]
 # resize pixels to the model size
 image = Image.fromarray(face)
 image = image.resize(224, 224)
 face_array = asarray(image)
 return face_array


extract_face('./test/2.png')