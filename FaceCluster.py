import json

import face_recognition
import cv2
import os
import numpy as np
import argparse
import math




def assign_image_to_label(image, label_id, id):
    path = './' + '/cluster_' + str(label_id)
    if os.path.exists(path) == False:
        os.mkdir(path)
    filename = str(id) + '.jpg'
    cv2.imwrite(os.path.join(path, filename), image)
    return
    


"""  
get montage by passing the image array specifying the dimension of each image that we require and the number of images that we desire per row
"""
def get_collage(image_arr, dim, per_row):
    imgs = []
    
    for img in image_arr:
        img = cv2.resize(img, (dim[0], dim[1]), cv2.INTER_CUBIC)
        imgs.append(img)
    # calculate the number of rows required
    rows = len(imgs) / per_row
    # increase one more row if it is a decimal
    rows = math.ceil(rows)

    # for the first row, just horizontally concat first "per_row" images
    first_row = cv2.hconcat(imgs[0:per_row])

    # calculate the number of blank spaces to append in the last row
    blank_img_size = len(imgs) % per_row
    # array to store the blank images
    blank_arr = []

    # case when blank image need to be appended to the last row
    if blank_img_size > 0:
        for i in range(per_row - blank_img_size):
            blank_arr.append(np.zeros((dim[0], dim[1], 3), np.uint8))
        # join all the blank spaces  to form a blank image
        blank_image = cv2.hconcat(blank_arr)

    # code for joining all the rows starting from 2 to first row
    for row in range(1, rows):
        # find the index of the start element for the particular row
        start = per_row + ((row - 1) * per_row)
        # update the end element for row depending on whether it will have blank image or not
        if len(imgs) >= ((row + 1) * per_row):
            end = (row + 1) * per_row
        else:
            end = (row * per_row) + (len(imgs) % per_row)
        # horizontally concat all images from start to end
        h_imgs = cv2.hconcat(imgs[start: end])

        # check if it is the last row, if yes break the loop after concating the row to the first row
        if (row == 1 and (row == rows - 1)):
            # if blank imae size is greater than 0, concat it to the h_imgs image
            if blank_img_size > 0:
                h_imgs = cv2.hconcat([h_imgs, blank_image])
            f_imgs = cv2.vconcat([first_row, h_imgs])
            break
        # this condition says there is at least one more row left
        if (row == 1):
            f_imgs = cv2.vconcat([first_row, h_imgs])
        elif (row > 1):
            if (blank_img_size > 0 and (row == rows - 1)):
                h_imgs = cv2.hconcat([h_imgs, blank_image])
            #f_imgs = np.concatenate((f_imgs, h_imgs), axis=1)
            f_imgs = cv2.vconcat([f_imgs, h_imgs])
    if rows == 1:
        return first_row
    else:
        return f_imgs




def k_means(X, K=5, iteration_count=1000):
    # list to store all the clusters with each cluster being a list having the samples
    clusters = [[] for _ in range(K)]
    # array to store the mean center for each cluster
    centroids = []

    n_samples, n_features = X.shape

    # initialize centroids

    centroids.append(X[np.random.randint(X.shape[0]), :])
    # compute remaining k - 1 centroids
    for _ in range(K - 1):
        # initialize a list to store distances from nearest centroid
        dist = []
        for i in range(X.shape[0]):
            point = X[i, :]
            d = 922337203685477
            # calculate distance of sample from each of the centroid and store the minimum distance
            for j in range(len(centroids)):
                single_dist = np.sqrt(np.sum((point - centroids[j]) ** 2))
                d = min(d, single_dist)
            dist.append(d)
        # select points with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = X[np.argmax(dist), :]
        centroids.append(next_centroid)
    
    
 
    
    # optimize the clustering result by iterating
    for _ in range(iteration_count):

        # create clusters by assigning each sample to the nearest centroid
        clusters = [[] for _ in range(K)]
        for index, sample in enumerate(X):

            # list to store the distance of sample from all centroids
            distances = [np.sqrt(np.sum((sample - ctd) ** 2)) for ctd in centroids]
            # get the nearest centroid index
            nearest_ctd_index = np.argmin(distances)

            clusters[nearest_ctd_index].append(index)

        # assign current centroid to previous
        prev_centroids = centroids
        # Calculate new centroids from the clusters
        centroids = np.zeros((K, n_features))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(X[cluster], axis=0)
            centroids[cluster_index] = cluster_mean

        # Check if the algorithm has converged(cluster unchanged)

        distances = [np.sqrt(np.sum((prev_centroids[i] - centroids[i]) ** 2)) for i in range(K)]
        # returns True if distances have not changed else False
        if sum(distances) == 0:
            break

    # Classify each sample to the index of the cluster
    labels = np.empty(n_samples)
    # assign each sample with the label of the cluster
    for cluster_index, cluster in enumerate(clusters):
        for sample_index in cluster:
            labels[sample_index] = cluster_index


    return labels
    





parser = argparse.ArgumentParser(description='Parse input path string')
parser.add_argument('path', help='Input Path', nargs='+')
args = parser.parse_args()
cluster_size = args.path[0].split("/")[-1].split("_")[-1]
imgs_path = os.path.join(args.path[0])


model_file = "./Model_Files/res10_300x300_ssd_iter_140000.caffemodel"
config_file = "./Model_Files/deploy.prototxt.txt"
model = cv2.dnn.readNetFromCaffe(config_file, model_file)

img_names = os.listdir(imgs_path)
im_stats = []
X = []
face_count = 0
for img_name in img_names:
    frame = cv2.imread(os.path.join(imgs_path,img_name))
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (190, 190)), 1.0, (190, 190), (104.0, 117.0, 123.0))
    model.setInput(blob)
    faces = model.forward()
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.85:
            x1 = float(faces[0, 0, i, 3] * w)
            y1 = float(faces[0, 0, i, 4] * h)
            x2 = float(faces[0, 0, i, 5] * w)
            y2 = float(faces[0, 0, i, 6] * h)
            box = [(int(y1), int(x2), int(y2), int(x1))] # top, right, bottom, left
            face_encodings = face_recognition.face_encodings(frame, box)
            im_stat = [{"image_name": img_name,"encoding": face_encodings,"bbox":box}]
            im_stats.extend(im_stat)
            face_count = face_count + 1
            X.append(np.array(face_encodings))
        #break

k = int(cluster_size)
labels = k_means(np.array(X).reshape(face_count,128), K = k)
label_ids = np.unique(labels)


json_faces_cluster = []
for label_id in label_ids:
    label_id = int(label_id)
    indices = np.where(labels == label_id)[0]
    indices = np.random.choice(indices, size=min(50, len(indices)), replace=False)
    faces = []
    cluster_img_names = []

    for i in indices:
        cluster_img_names.append(im_stats[i]["image_name"])
        image = cv2.imread(os.path.join(imgs_path,im_stats[i]["image_name"]))
        [(top, right, bottom, left)] = im_stats[i]["bbox"]
        face = image[top:bottom, left:right]
        assign_image_to_label(image, label_id, i)
        faces.append(face)
    json_faces_cluster.append({"cluster_no": label_id, "elements": cluster_img_names})
    montage = get_collage(faces,(100,100),per_row=4)
    if(label_id == -1):
        cluster_name = "unrecognized_faces"
    else:
        cluster_name = "cluster_" + str(label_id)
    cv2.imwrite(os.path.join('./', cluster_name + '.jpg'), montage)

with open('clusters.json', 'w') as f:
    json.dump(json_faces_cluster, f)
