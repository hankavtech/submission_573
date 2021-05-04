import json
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Parse input path string')
parser.add_argument('path', help='Input Path', nargs='+')
args = parser.parse_args()
imgs_path = os.path.join(args.path[0],'images')
model_file = "./Model_Files/res10_300x300_ssd_iter_140000.caffemodel"
config_file = "./Model_Files/deploy.prototxt.txt"
model = cv2.dnn.readNetFromCaffe(config_file, model_file)
files = os.listdir(imgs_path)
face_json_arr = []
for cap in files:
    frame = cv2.imread(os.path.join(imgs_path,cap))
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (190,190)), 1.0, (190,190), (104.0, 117.0, 123.0))
    model.setInput(blob)
    faces = model.forward()
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            x1 = float(faces[0, 0, i, 3] * w)
            y1 = float(faces[0, 0, i, 4] * h)
            x2 = float(faces[0, 0, i, 5] * w)
            y2 = float(faces[0, 0, i, 6] * h)
            ele = {'iname': str(cap), 'bbox': [x1, y1, x2-x1, y2-y1]}
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            face_json_arr.append(ele)
            #cv2.imshow('im', frame)
            #cv2.waitKey(0)


#store character features in a file
with open("results.json", 'w') as f:
    json.dump(face_json_arr,f)

cv2.destroyAllWindows()
