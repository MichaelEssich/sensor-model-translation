import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils import Sequence
from cv2 import imread, resize, polylines, circle
from __builtin__ import int
from os import listdir
from os.path import join
import csv
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et
import pickle
import os

joints_list = ['head', 'neck', 'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow',
                         'right_hand', 'left_hand', 'pelvis', 'right_hip', 'left_hip',
                         'right_knee', 'left_knee', 'right_foot', 'left_foot']
joints_to_connect_red = [["head", "neck"],
                         ["left_hand", "left_elbow", "left_shoulder", "neck", "pelvis", "left_hip", "left_knee", "left_foot"]]
joints_to_connect_blue = [["pelvis", "right_hip", "right_knee", "right_foot"],
                          ["neck", "right_shoulder", "right_elbow", "right_hand"]]
threshold_to_detect_joints = 30


def draw_heatmap(img, pt, sigma=1, heatmap_type='Gaussian'):
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    # Draw a 2D gaussian
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    
    if heatmap_type == 'Gaussian':
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif heatmap_type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)
        
    for i in range(len(g)):
        for j in range(len(g[i])):
            g[i][j] = int(g[i][j] * 255)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1], 0] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def crop_and_resize_image(img):
    middle = img.shape[1] / 2
    left_bound = middle - img.shape[0] / 2
    right_bound = middle + img.shape[0] / 2
    img_resized = img[0 : img.shape[0], left_bound : right_bound]
    img_resized = resize(img_resized, (256, 256))
    return img_resized


def get_joint_position(orig_image_shape, joint_pos):
    middle = orig_image_shape[1] / 2
    left_bound = middle - orig_image_shape[0] / 2
    right_bound = middle + orig_image_shape[0] / 2
    if joint_pos[0] < left_bound or joint_pos[0] > right_bound:
        return None
    else:
        joint_pos[0] = joint_pos[0] - left_bound
        joint_pos = [int(joint_pos[0] * 64 / float(right_bound - left_bound)),
                     int(joint_pos[1] * 64 / float(orig_image_shape[0]))]
        return joint_pos

'''
    Important when creating a new CustomSequence:
    Set correct number_of_hourglasses after creating CustomSequence!
'''


class CustomSequence(Sequence):
    
    def __init__(self, x_set, y_set, batch_size):
        self.x = sorted([join(x_set, filename) for filename in listdir(x_set)], key=lambda k: k.lower())
        self.y = sorted([join(y_set, filename) for filename in listdir(y_set)], key=lambda k: k.lower())
        self.batch_size = batch_size
        self.number_of_hourglasses = 1

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        input_data = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_data = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x = []
        orig_data_shapes = []
        for data in input_data:
            with open(data, 'rb') as f:
                if data.endswith(".csv"):
                    csv_reader = csv.reader(f, delimiter=";")
                    output = []
                    for row in csv_reader:
                        output.append(row)
                elif data.endswith(".pickle"):
                    output = pickle.load(f)
            tof_data = np.asarray(output, dtype=np.float32)
            orig_data_shapes.append(tof_data.shape)
            normalized = crop_and_resize_image(tof_data)
            normalized = (normalized - normalized.min()) / float(normalized.max() - normalized.min())
            batch_x.append(np.expand_dims(normalized, axis=2))
        
        output = []
        for i in range(len(label_data)):
            with open(label_data[i], 'rb') as f:
                tree = et.parse(f)  
                root = tree.getroot()
                joints = np.zeros((64, 64, 15), dtype=int)
                for j in range(len(joints_list)):       
                    obj = root.find(".//*[name='%s']/name/.." % joints_list[j])
                    heatmap = np.zeros((64, 64, 1), dtype=int)
                    if obj is not None:
                        x = int(float(obj.find("polygon/pt/x").text))
                        y = int(float(obj.find("polygon/pt/y").text))
                        joint_pos = get_joint_position(orig_data_shapes[i], [x, y])
                        if (joint_pos != None):
                            heatmap = draw_heatmap(heatmap, joint_pos)
                    joints[0:64, 0:64, j] = heatmap[0:64, 0:64, 0]
                
            output.append(joints)
        
        if self.number_of_hourglasses > 1:
            batch_y = []
            for i in range(self.number_of_hourglasses):
                batch_y.append(np.array(output))
            return np.array(batch_x), batch_y
        else:
            return np.array(batch_x), np.array(output)


def get_predict_batch(input_data):
    batch_x = []
    if not input_data.endswith(os.sep):
        input_data = input_data + os.sep
    for data in sorted([join(input_data, filename) for filename in listdir(input_data)], key=lambda k: k.lower()):
        with open(data, 'rb') as f:
            if data.endswith(".csv"):
                csv_reader = csv.reader(f, delimiter=";")
                output = []
                for row in csv_reader:
                    output.append(row)
            elif data.endswith(".pickle"):
                output = pickle.load(f)
        tof_data = np.asarray(output, dtype=np.float32)
        normalized = crop_and_resize_image(tof_data)
        normalized = (normalized - normalized.min()) / float(normalized.max() - normalized.min())
        batch_x.append(np.expand_dims(normalized, axis=2))
    return np.array(batch_x)


def get_imgs_for_visualization(imgs_vis_path, batch_size, i):
    if not imgs_vis_path.endswith(os.sep):
        imgs_vis_path = imgs_vis_path + os.sep
    input_data = sorted([os.path.join(imgs_vis_path, filename) for filename in os.listdir(imgs_vis_path)], key=lambda k: k.lower())
    size = len(input_data)
    if (i * batch_size) < size:
        input_data = input_data[i * batch_size : (i + 1) * batch_size]
    else:
        input_data = input_data[i * batch_size : size - 1]
    batch_x = []
    for img_file in input_data:
            img = imread(img_file)
            batch_x.append(crop_and_resize_image(img))
    return np.array(batch_x)


def get_callbacks():
    return [ModelCheckpoint("weights/weights_epoch_{epoch:02d}-val_loss_{val_loss:.2f}.hdf5",
                            monitor='val_loss', verbose=1, save_weights_only=True,
                            save_best_only=True, mode='min'),
            CSVLogger("training_log.csv", ";", True)]


def show_heatmaps_w_maximum_per_joint(results, img_vis_batch):
    for i in range(len(results)):
        img = img_vis_batch[i]
        heatmaps_list = []
        imgs_resized_list = []
        for j in range(results[i].shape[2]):
            tmp = np.zeros((64, 64, 3), dtype=int)
            tmp = results[i][0:64, 0:64, j]
            heatmaps_list.append(tmp)
            imgs_resized_list.append(resize(img, (64, 64)))
        
        start_h = 0
        heatmaps = []
        images_resized = []
        imgs_per_line = 5
        for i in range((imgs_per_line - len(heatmaps_list) % imgs_per_line) % imgs_per_line):
            heatmaps_list.append(np.zeros((64, 64, 3), dtype=int))
        for i in range(len(heatmaps_list) / imgs_per_line):
            end_h = start_h + imgs_per_line
            heatmaps.append(np.hstack((heatmaps_list[start_h:end_h])))
            images_resized.append(np.hstack((imgs_resized_list[start_h:end_h])))
            start_h = end_h
        heatmaps = np.vstack((heatmaps[0:len(heatmaps_list)]))
        images_resized = np.vstack((images_resized[0:len(imgs_resized_list)]))
        plt.imshow(heatmaps)
        plt.imshow(images_resized, alpha=0.3)
        plt.show()


def show_pose(results, imgs_vis_batch):
    for i in range(len(results)):
        img = imgs_vis_batch[i]
        joints_pos = {}
        for j in range(results[i].shape[2]):
            max_pos = np.argwhere(results[i][0:64, 0:64, j] == np.max(results[i][0:64, 0:64, j]))
            if results[i][max_pos[0][0], max_pos[0][1], j] > threshold_to_detect_joints:
                joints_pos[joints_list[j]] = [max_pos[0][1] * 4 + 1, max_pos[0][0] * 4 + 1]
            else:
                joints_pos[joints_list[j]] = None
        
        for j in range(len(joints_to_connect_red)):
            line = []
            for k in range(len(joints_to_connect_red[j])):
                if joints_pos[joints_to_connect_red[j][k]]:
                    line.append(joints_pos[joints_to_connect_red[j][k]])
            if line:
                pts = np.array(line, np.int32)
                polylines(img, [pts], False, (255, 0, 0), thickness=2)
        
        for j in range(len(joints_to_connect_blue)):
            line = []
            for k in range(len(joints_to_connect_blue[j])):
                if joints_pos[joints_to_connect_blue[j][k]]:
                    line.append(joints_pos[joints_to_connect_blue[j][k]])
            if line:
                pts = np.array(line, np.int32)
                polylines(img, [pts], False, (0, 0, 255), thickness=2)
        
        plt.imshow(img)
        plt.show()


def save_pose(results, imgs_vis_batch, filename):
    for i in range(len(results)):
        img = imgs_vis_batch[i]
        joints_pos = {}
        for j in range(results[i].shape[2]):
            max_pos = np.argwhere(results[i][0:64, 0:64, j] == np.max(results[i][0:64, 0:64, j]))
            if results[i][max_pos[0][0], max_pos[0][1], j] > threshold_to_detect_joints:
                joints_pos[joints_list[j]] = [max_pos[0][1] * 4 + 1, max_pos[0][0] * 4 + 1]
            else:
                joints_pos[joints_list[j]] = None
        
        for j in range(len(joints_to_connect_red)):
            line = []
            for k in range(len(joints_to_connect_red[j])):
                if joints_pos[joints_to_connect_red[j][k]]:
                    line.append(joints_pos[joints_to_connect_red[j][k]])
            if line:
                pts = np.array(line, np.int32)
                polylines(img, [pts], False, (255, 0, 0), thickness=2)
        
        for j in range(len(joints_to_connect_blue)):
            line = []
            for k in range(len(joints_to_connect_blue[j])):
                if joints_pos[joints_to_connect_blue[j][k]]:
                    line.append(joints_pos[joints_to_connect_blue[j][k]])
            if line:
                pts = np.array(line, np.int32)
                polylines(img, [pts], False, (0, 0, 255), thickness=2)
        
        plt.imsave(filename + "_%05d" % i, img)

        
def show_joints(results, imgs_vis_batch):
    for i in range(len(results)):
        img = imgs_vis_batch[i]
        for j in range(results[i].shape[2]):
            max_pos = np.argwhere(results[i][0:64, 0:64, j] == np.max(results[i][0:64, 0:64, j]))
            if results[i][max_pos[0][0], max_pos[0][1], j] > threshold_to_detect_joints:
                if "right" not in joints_list[j]:
                    circle(img, (max_pos[0][1] * 4 + 1, max_pos[0][0] * 4 + 1), 1, (255, 0, 0), thickness=1, lineType=8, shift=0)
                else:
                    circle(img, (max_pos[0][1] * 4 + 1, max_pos[0][0] * 4 + 1), 1, (0, 0, 255), thickness=1, lineType=8, shift=0)
                
        plt.imshow(img)
        plt.show()


def save_joints(results, imgs_vis_batch, filename):
    for i in range(len(results)):
        img = imgs_vis_batch[i]
        for j in range(results[i].shape[2]):
            max_pos = np.argwhere(results[i][0:64, 0:64, j] == np.max(results[i][0:64, 0:64, j]))
            if results[i][max_pos[0][0], max_pos[0][1], j] > threshold_to_detect_joints:
                if "right" not in joints_list[j]:
                    circle(img, (max_pos[0][1] * 4 + 1, max_pos[0][0] * 4 + 1), 1, (255, 0, 0), thickness=1, lineType=8, shift=0)
                else:
                    circle(img, (max_pos[0][1] * 4 + 1, max_pos[0][0] * 4 + 1), 1, (0, 0, 255), thickness=1, lineType=8, shift=0)
                
        plt.imsave(filename + "_%05d" % i, img)

