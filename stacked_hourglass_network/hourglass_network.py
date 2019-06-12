'''
Created on 22.08.2018

@author: michael
'''

from keras.layers import Conv2D, BatchNormalization, Input, MaxPool2D, Add
from keras.utils import plot_model
from keras.models import Model
from keras.layers.convolutional import UpSampling2D
from stacked_hourglass_network.utils import joints_list, CustomSequence, threshold_to_detect_joints
import numpy as np
import math
import datetime
import os
from keras.optimizers import Adam
import pickle
import csv
from stacked_hourglass_network import utils


class HourglassNetwork():
    
    def __init__(self, weights=None, number_of_joints=15, number_of_hourglasses=8, create_model_png=False):
        print "Output order of predictions:"
        print joints_list
        self.d_number = 0
        self.number_of_hourglasses = number_of_hourglasses
        self.model_train, self.model_predict = self.__create_model(number_of_joints, number_of_hourglasses, create_model_png)
        if weights:
            self.model_train.load_weights(weights)
        
    def __create_model(self, number_of_joints, number_of_hourglasses, create_model_png=False):
        input_shape = Input(shape=(256, 256, 1), name="input_shape")
        residual_module_counter = 1
        hg_counter = 1
        cl_counter = 1
        mp_counter = 1
        output = []
        cl_counter, mp_counter, before_hourglass, residual_module_counter, layers = self.__create_pre_hourglass(cl_counter,
                                                                                                         mp_counter,
                                                                                                         residual_module_counter,
                                                                                                         input_shape)
        for i in range(number_of_hourglasses):
            hg_counter, mp_counter, residual_module_counter, layers = self.__create_hourglass(hg_counter, mp_counter,
                                                                                       residual_module_counter, layers)
            cl_counter, layers, layer_for_intermediate_loss = self.__create_intermediate_results(cl_counter, number_of_joints, before_hourglass, layers)
            if i != number_of_hourglasses - 1:
                output.append(layer_for_intermediate_loss)
            
        layers = self.__create_output(cl_counter, number_of_joints, layers)
        if number_of_hourglasses == 1:
            output = layers
        else:
            output.append(layers)
        
        optimizer = Adam()
        
        model_predict = Model(inputs=input_shape, outputs=layers, name="model_predict")
        
        model_train = Model(inputs=input_shape, outputs=output, name="model_train")
        model_train.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])
        
        print ("model_train: %s" % model_train.metrics_names)
        
        if create_model_png:
            plot_model(model_predict, to_file='model_predict.png', show_shapes=True)
            plot_model(model_train, to_file='model_train.png', show_shapes=True)
        return model_train, model_predict
    
    def __create_pre_hourglass(self, cl_counter, mp_counter, residual_module_counter, input_shape):
        layers = input_shape
        layers = Conv2D(filters=256, kernel_size=(7, 7), strides=(2, 2), activation="relu", padding="same",
                        name="cl_%s" % cl_counter)(layers)
        cl_counter = cl_counter + 1
        layers = BatchNormalization()(layers)
        residual_module_counter, layers = self.__create_residual_module(residual_module_counter, layers, "front")
        layers = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="mp_%s" % mp_counter)(layers)
        mp_counter = mp_counter + 1
        residual_module_counter, layers = self.__create_residual_module(residual_module_counter, layers, "front")
        residual_module_counter, layers = self.__create_residual_module(residual_module_counter, layers, "front")
        before_hourglass = layers
        return cl_counter, mp_counter, before_hourglass, residual_module_counter, layers
    
    def __create_hourglass(self, hg_counter, mp_counter, residual_module_counter, layers):
        residual_module_counter, layers = self.__create_residual_module(residual_module_counter, layers, "hg_%s" % hg_counter)
        residual_module_counter, branch_1 = self.__create_residual_module(residual_module_counter, layers, "hg_%s" % hg_counter)
        layers = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="hg_%s_mp_%s" % (hg_counter, mp_counter))(layers)
        mp_counter = mp_counter + 1
        residual_module_counter, layers = self.__create_residual_module(residual_module_counter, layers, "hg_%s" % hg_counter)
        residual_module_counter, branch_2 = self.__create_residual_module(residual_module_counter, layers, "hg_%s" % hg_counter)
        layers = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="hg_%s_mp_%s" % (hg_counter, mp_counter))(layers)
        mp_counter = mp_counter + 1
        residual_module_counter, layers = self.__create_residual_module(residual_module_counter, layers, "hg_%s" % hg_counter)
        residual_module_counter, branch_3 = self.__create_residual_module(residual_module_counter, layers, "hg_%s" % hg_counter)
        layers = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="hg_%s_mp_%s" % (hg_counter, mp_counter))(layers)
        mp_counter = mp_counter + 1
        residual_module_counter, layers = self.__create_residual_module(residual_module_counter, layers, "hg_%s" % hg_counter)
        residual_module_counter, branch_4 = self.__create_residual_module(residual_module_counter, layers, "hg_%s" % hg_counter)
        layers = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="hg_%s_mp_%s" % (hg_counter, mp_counter))(layers)
        mp_counter = mp_counter + 1
        residual_module_counter, layers = self.__create_residual_module(residual_module_counter, layers, "hg_%s" % hg_counter)
        residual_module_counter, layers = self.__create_residual_module(residual_module_counter, layers, "hg_%s" % hg_counter)
        residual_module_counter, layers = self.__create_residual_module(residual_module_counter, layers, "hg_%s" % hg_counter)
        layers = UpSampling2D()(layers)
        layers = Add(name="hg_%s_connect_branch_4" % hg_counter)([branch_4, layers])
        layers = UpSampling2D()(layers)
        layers = Add(name="hg_%s_connect_branch_3" % hg_counter)([branch_3, layers])
        layers = UpSampling2D()(layers)
        layers = Add(name="hg_%s_connect_branch_2" % hg_counter)([branch_2, layers])
        layers = UpSampling2D()(layers)
        layers = Add(name="hg_%s_connect_branch_1" % hg_counter)([branch_1, layers])
        hg_counter = hg_counter + 1
        return hg_counter, mp_counter, residual_module_counter, layers
    
    def __create_intermediate_results(self, cl_counter, number_of_joints, before_hourglass, layers):
        # produce intermediate heatmaps as input for next hourglass
        layers = Conv2D(filters=512, kernel_size=(1, 1),
                        activation="relu", padding="same", name="cl_%s" % cl_counter)(layers)
        cl_counter = cl_counter + 1
        layers = BatchNormalization()(layers)
        # Add intermediate loss here:
        skip = Conv2D(filters=number_of_joints, kernel_size=(1, 1),
                        activation="relu", padding="same", name="cl_%s" % cl_counter)(layers)
        layer_for_intermediate_loss = skip
        cl_counter = cl_counter + 1
        skip = BatchNormalization()(skip)
        skip = Conv2D(filters=256, kernel_size=(1, 1),
                        activation="relu", padding="same", name="cl_%s" % cl_counter)(skip)
        cl_counter = cl_counter + 1
        skip = BatchNormalization()(skip)
        layers = Conv2D(filters=256, kernel_size=(1, 1),
                        activation="relu", padding="same", name="cl_%s" % cl_counter)(layers)
        cl_counter = cl_counter + 1
        layers = BatchNormalization()(layers)
        layers = Add()([skip, layers, before_hourglass])
        return cl_counter, layers, layer_for_intermediate_loss
    
    def __create_output(self, cl_counter, number_of_joints, layers):
        layers = Conv2D(filters=256, kernel_size=(1, 1),
                        activation="relu", padding="same", name="cl_%s" % cl_counter)(layers)
        cl_counter = cl_counter + 1
        layers = BatchNormalization()(layers)
        layers = Conv2D(filters=number_of_joints, kernel_size=(1, 1),
                        activation="relu", padding="same", name="cl_%s" % cl_counter)(layers)
        cl_counter = cl_counter + 1
        return layers
        
    def __create_residual_module(self, residual_module_counter, input_layer, hg_number):
        skip = input_layer
    
        rm = Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="same",
                     name="%s_rm_%s_cl_1" % (hg_number, residual_module_counter))(input_layer)
        rm = BatchNormalization()(rm)
        rm = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same",
                     name="%s_rm_%s_cl_2" % (hg_number, residual_module_counter))(rm)
        rm = BatchNormalization()(rm)
        rm = Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same",
                     name="%s_rm_%s_cl_3" % (hg_number, residual_module_counter))(rm)
        rm = BatchNormalization()(rm)
        rm = Add(name="%s_skip_rm_%s" % (hg_number, residual_module_counter))([skip, rm])
        residual_module_counter += 1
        return residual_module_counter, rm
    
    def calc_distance_metric(self, data_path, annotations_path, batch_size, g_ts=None):
        sequence = CustomSequence(data_path, annotations_path, batch_size)
        sequence.number_of_hourglasses = self.number_of_hourglasses
        sum_distance = 0
        total_number_of_joints = 0
        for batch_index in range(len(sequence)):
            data, labels = sequence.__getitem__(batch_index)
            labels = labels[-1]
            if g_ts:
                data = g_ts.predict_on_batch(data)
            results = self.model_predict.predict_on_batch(data)
            for result_index in range(len(results)):
                prediction = results[result_index]
                label = labels[result_index]
                for heatmap_index in range(prediction.shape[2]):
                    max_pos_prediction = np.argwhere(prediction[0:64, 0:64, heatmap_index] == np.max(prediction[0:64, 0:64, heatmap_index]))[0]
                    max_pos_label = np.argwhere(label[0:64, 0:64, heatmap_index] == np.max(label[0:64, 0:64, heatmap_index]))[0]
                    if prediction[max_pos_prediction[0], max_pos_prediction[1], heatmap_index] <= threshold_to_detect_joints:
                        max_pos_prediction[0] = 0
                        max_pos_prediction[1] = 0
                    distance = max_pos_prediction - max_pos_label
                    distance = math.sqrt(math.pow(distance[0], 2) + math.pow(distance[1], 2))
                    sum_distance = sum_distance + distance
                    total_number_of_joints = total_number_of_joints + 1
        return sum_distance / float(total_number_of_joints)

    def calc_threshold_metric(self, data_path, annotations_path, batch_size, threshold, g_ts=None):
        sequence = CustomSequence(data_path, annotations_path, batch_size)
        sequence.number_of_hourglasses = self.number_of_hourglasses
        total_number_of_joints = 0
        correct_joints = 0
        for batch_index in range(len(sequence)):
            data, labels = sequence.__getitem__(batch_index)
            labels = labels[-1]
            if g_ts:
                data = g_ts.predict_on_batch(data)
            results = self.model_predict.predict_on_batch(data)
            for result_index in range(len(results)):
                prediction = results[result_index]
                label = labels[result_index]
                for heatmap_index in range(prediction.shape[2]):
                    max_pos_prediction = np.argwhere(prediction[0:64, 0:64, heatmap_index] == np.max(prediction[0:64, 0:64, heatmap_index]))[0]
                    max_pos_label = np.argwhere(label[0:64, 0:64, heatmap_index] == np.max(label[0:64, 0:64, heatmap_index]))[0]
                    distance = max_pos_prediction - max_pos_label
                    distance = math.sqrt(math.pow(distance[0], 2) + math.pow(distance[1], 2))
                    if distance < threshold:
                        correct_joints = correct_joints + 1
                    total_number_of_joints = total_number_of_joints + 1
        acc = correct_joints * 100.0 / float(total_number_of_joints)
        return acc

    def train(self, epochs, epoch_to_resume, batch_size,
                 train_data_path, train_annotations_path,
                 val_data_path, val_annotations_path, val_pck_threshold):
        start_time = datetime.datetime.now()
        
        if epoch_to_resume == 1 and os.path.exists("training_log.csv"):
            os.remove("training_log.csv")
        
        sequence = CustomSequence(train_data_path, train_annotations_path, batch_size)
        sequence.number_of_hourglasses = self.number_of_hourglasses
        if self.number_of_hourglasses == 1:
            field_names = ["Epoch", "HG loss", "HG acc", "Train PCK acc (TH %s)" % val_pck_threshold]
        else:
            field_names = ["Epoch", "HG total loss", "HG loss", "HG acc", "Train PCK acc (TH %s)" % val_pck_threshold]
        if val_data_path and val_annotations_path:
            field_names.append("Val PCK acc (TH %s)" % val_pck_threshold)
        pck_acc = None
        best_pck_acc = None
        for epoch in range(epoch_to_resume, epochs + 1):
            hg_total_loss_list = []
            hg_loss_list = []
            hg_acc_list = []
            for i in range(len(sequence)):
                data, labels = sequence.__getitem__(i)
                hg_loss = self.model_train.train_on_batch(data, labels)

                elapsed_time = datetime.datetime.now() - start_time
                
                if self.number_of_hourglasses == 1:
                    output = ("[Epoch: %d/%d] [Batch: %d/%d] [HG loss: %f, HG acc: %f] time: %s " \
                                                                        % (epoch , epochs,
                                                                            i + 1, len(sequence),
                                                                            hg_loss[0], hg_loss[1],
                                                                            elapsed_time))
                    hg_loss_list.append(hg_loss[0])
                    hg_acc_list.append(hg_loss[1])
                else:
                    output = ("[Epoch: %d/%d] [Batch: %d/%d] [HG total loss: %f, HG loss: %f, HG acc: %f] time: %s " \
                                                                        % (epoch, epochs,
                                                                            i + 1, len(sequence),
                                                                            hg_loss[0], hg_loss[self.number_of_hourglasses],
                                                                            hg_loss[self.number_of_hourglasses * 2],
                                                                            elapsed_time))
                    hg_total_loss_list.append(hg_loss[0])
                    hg_loss_list.append(hg_loss[self.number_of_hourglasses])
                    hg_acc_list.append(hg_loss[self.number_of_hourglasses * 2])
                
                print(output)
                if i == len(sequence) - 1:
                    with open("training_log.csv", "a") as f:
                        history = []
                        history.append(epoch)
                        if self.number_of_hourglasses > 1:
                            history.append(np.mean(hg_total_loss_list))
                        history.append(np.mean(hg_loss_list))
                        history.append(np.mean(hg_acc_list))
                        pck_acc = self.calc_threshold_metric(train_data_path, train_annotations_path, batch_size, threshold=val_pck_threshold, g_ts=None)
                        history.append(pck_acc)
                        if val_data_path and val_annotations_path:
                            pck_acc = self.calc_threshold_metric(val_data_path, val_annotations_path, batch_size, threshold=val_pck_threshold, g_ts=None)
                            history.append(pck_acc)
                        writer = csv.DictWriter(f, fieldnames=field_names)
                        if epoch == 1:
                            writer.writeheader()
                        writer.writerow(dict(zip(field_names, history)))
                    if not os.path.exists("saved_weights"):
                        os.mkdir("saved_weights")
                    self.model_train.save_weights("saved_weights/weights.hdf5", True)
                    if pck_acc and (not best_pck_acc or pck_acc > best_pck_acc):
                        best_pck_acc = pck_acc
                        self.model_train.save_weights("saved_weights/best_weights.hdf5", True)
                    
    def get_data_t(self, data_t_path, batch_size, i):
        input_data = sorted([os.path.join(data_t_path, filename) for filename in os.listdir(data_t_path)], key=lambda k: k.lower())
        size = len(input_data)
        if (i * batch_size) < size:
            input_data = input_data[i * batch_size : (i + 1) * batch_size]
        else:
            input_data = input_data[i * batch_size : size - 1]
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
            normalized = utils.crop_and_resize_image(tof_data)
            normalized = (normalized - normalized.min()) / float(normalized.max() - normalized.min())
            batch_x.append(np.expand_dims(normalized, axis=2))
        return np.asarray(batch_x)
