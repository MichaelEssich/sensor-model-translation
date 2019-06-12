import datetime
from stacked_hourglass_network.utils import CustomSequence, crop_and_resize_image
import os
import numpy as np
import csv
import pickle
from stacked_hourglass_network import utils
from cv2 import imread
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Activation, Conv2DTranspose, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model
from stacked_hourglass_network.hourglass_network import HourglassNetwork
from keras.layers.merge import concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

# The architecture of the generator and discriminator is based on CycleGAN implementation of
# R. Atienza. Advanced Deep Learning with Keras. Packt Publishing Ltd, 2018. ISBN: 978-1-78862-941-6.
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter7-cross-domain-gan/cyclegan-7.1.1.py

smt_progress_images_out_dir = "images_training_progress"


class SMT():
    __weights_combined = "saved_weights/weights_combined.hdf5"
    __weights_d_s = "saved_weights/weights_d_s.hdf5"
    __weights_d_t = "saved_weights/weights_d_t.hdf5"
    __weights_g_st = "saved_weights/weights_g_st.hdf5"
    __weights_g_ts = "saved_weights/weights_g_ts.hdf5"
    __weights_combined_best = "saved_weights/best_weights_combined.hdf5"
    __weights_d_s_best = "saved_weights/best_weights_d_s.hdf5"
    __weights_d_t_best = "saved_weights/best_weights_d_t.hdf5"
    __weights_g_st_best = "saved_weights/best_weights_g_st.hdf5"
    __weights_g_ts_best = "saved_weights/best_weights_g_ts.hdf5"

    def __init__(self, hgn_weights, source_data_path, source_annotations_path, target_data_path, target_imgs_for_visualization,
                 create_model_png=False, epoch_to_resume=1, number_of_hourglasses=4):
        
        self.epoch_to_resume = epoch_to_resume
        self.source_data_path = source_data_path
        self.target_data_path = target_data_path
        self.source_annotations_path = source_annotations_path
        self.target_imgs_for_visualization = target_imgs_for_visualization
        self.number_of_hourglasses = number_of_hourglasses
        shapes = ((256, 256, 1), (256, 256, 1))
        # Source model
        self.hgn = HourglassNetwork(weights=hgn_weights, number_of_hourglasses=number_of_hourglasses)
        self.task_model = self.hgn.model_predict
        self.task_model.trainable = False
        self.task_model.name = "task_model"
        self.use_patchgan = True
        self.g_ts, self.g_st, self.d_s, self.d_t, self.combined = self.build_SMT(shapes, create_model_png)
        if self.epoch_to_resume != 1:
            self.g_ts.load_weights(self.__weights_g_ts)
            self.g_st.load_weights(self.__weights_g_st)
            self.d_s.load_weights(self.__weights_d_s)
            self.d_t.load_weights(self.__weights_d_t)
            self.combined.load_weights(self.__weights_combined)

        patch = int(256 / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

    def train(self, epochs, batch_size=1, sample_interval=50,
              target_annotations_path_for_metric=None, val_data_path=None, val_annotations_path=None, val_pck_threshold=2):

        start_time = datetime.datetime.now()
        
        if self.epoch_to_resume == 1 and os.path.exists("training_log.csv"):
            os.remove("training_log.csv")

        sequence = CustomSequence(self.source_data_path, self.source_annotations_path, batch_size)
        sequence.number_of_hourglasses = self.number_of_hourglasses
        
        field_names = ["Epoch", "d_s_loss", "d_s_acc", "d_t_loss", "d_t_acc", "g_loss", "g_d_s_loss", "g_d_t_loss", "g_recon_s_loss", "g_recon_t_loss",
                       "task_model_recon_loss", "g_d_s_acc", "g_d_t_acc", "g_recon_s_acc", "g_recon_t_acc", "task_model_recon_acc"]
        if target_annotations_path_for_metric:
            field_names.append("Train PCK acc (TH %s)" % val_pck_threshold)
        if val_data_path and val_annotations_path:
            field_names.append("Val PCK acc (TH %s)" % val_pck_threshold)
        pck_acc = None
        best_pck_acc = None
        for epoch in range(self.epoch_to_resume, epochs + 1):
            d_s_loss_list = []
            d_s_acc_list = []
            d_t_loss_list = []
            d_t_acc_list = []
            g_loss_list = []
            g_d_s_loss_list = []
            g_d_t_loss_list = []
            g_recon_s_loss_list = []
            g_recon_t_loss_list = []
            g_d_s_acc_list = []
            g_d_t_acc_list = []
            g_recon_s_acc_list = []
            g_recon_t_acc_list = []
            task_model_recon_loss_list = []
            task_model_recon_acc_list = []
            for i in range(len(sequence)):
                data_s, labels_s = sequence.__getitem__(i)
                data_t = self.get_data_t(self.target_data_path, batch_size, i)
                
                valid = np.ones((len(data_s),) + self.disc_patch)
                fake = np.zeros((len(data_s),) + self.disc_patch)
                
                fake_t = self.g_st.predict_on_batch(data_s)
                fake_s = self.g_ts.predict_on_batch(data_t)
                
                pred_real = self.task_model.predict_on_batch(data_s)
                pred_fake = self.task_model.predict_on_batch(fake_s)
                
                d_s_loss_real = self.d_s.train_on_batch([data_s, labels_s[-1]], valid)
                d_s_loss_fake = self.d_s.train_on_batch([fake_s, pred_fake], fake)
                d_s_loss = 0.5 * np.add(d_s_loss_real, d_s_loss_fake)

                d_t_loss_real = self.d_t.train_on_batch(data_t, valid)
                d_t_loss_fake = self.d_t.train_on_batch(fake_t, fake)
                d_t_loss = 0.5 * np.add(d_t_loss_real, d_t_loss_fake)
                
                d_loss = 0.5 * np.add(d_s_loss, d_t_loss)
                
                g_loss = self.combined.train_on_batch([data_s, data_t], [valid, valid, data_s, data_t, pred_real])
                
                elapsed_time = datetime.datetime.now() - start_time

                output = ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %f%%] [G loss: %f, adv: %f, recon: %f, task_model loss recon: %f task_model acc recon: %f] time: %s " \
                                                                        % (epoch, epochs,
                                                                            i + 1, len(sequence),
                                                                            d_loss[0], d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            g_loss[5],
                                                                            g_loss[10],
                                                                            elapsed_time))
                print(output)
                d_s_loss_list.append(d_s_loss[0])
                d_s_acc_list.append(d_s_loss[1])
                d_t_loss_list.append(d_t_loss[0])
                d_t_acc_list.append(d_t_loss[1])
                g_loss_list.append(g_loss[0])
                g_d_s_loss_list.append(g_loss[1])
                g_d_t_loss_list.append(g_loss[2])
                g_recon_s_loss_list.append(g_loss[3])
                g_recon_t_loss_list.append(g_loss[4])
                task_model_recon_loss_list.append(g_loss[5])
                g_d_s_acc_list.append(g_loss[6])
                g_d_t_acc_list.append(g_loss[7])
                g_recon_s_acc_list.append(g_loss[8])
                g_recon_t_acc_list.append(g_loss[9])
                task_model_recon_acc_list.append(g_loss[10])

                if (i + 1) % sample_interval == 0:
                    self.sample_images(epoch, i + 1, data_s, data_t)
                    self.sample_prediction(epoch, batch_size, i + 1, data_t)
                if i == len(sequence) - 1:
                    with open("training_log.csv", "a") as f:
                        history = []
                        history.append(epoch)
                        history.append(np.mean(d_s_loss_list))
                        history.append(np.mean(d_s_acc_list))
                        history.append(np.mean(d_t_loss_list))
                        history.append(np.mean(d_t_acc_list))
                        history.append(np.mean(g_loss_list))
                        history.append(np.mean(g_d_s_loss_list))
                        history.append(np.mean(g_d_t_loss_list))
                        history.append(np.mean(g_recon_s_loss_list))
                        history.append(np.mean(g_recon_t_loss_list))
                        history.append(np.mean(task_model_recon_loss_list))
                        history.append(np.mean(g_d_s_acc_list))
                        history.append(np.mean(g_d_t_acc_list))
                        history.append(np.mean(g_recon_s_acc_list))
                        history.append(np.mean(g_recon_t_acc_list))
                        history.append(np.mean(task_model_recon_acc_list))
                        if target_annotations_path_for_metric:
                            pck_acc = self.hgn.calc_threshold_metric(self.target_data_path, target_annotations_path_for_metric, batch_size * 2, threshold=val_pck_threshold, g_ts=self.g_ts)
                            history.append(pck_acc)
                        if val_data_path and val_annotations_path:
                            pck_acc = self.hgn.calc_threshold_metric(val_data_path, val_annotations_path, batch_size * 2, threshold=val_pck_threshold, g_ts=self.g_ts)
                            history.append(pck_acc)
                        writer = csv.DictWriter(f, fieldnames=field_names)
                        if epoch == 1:
                            writer.writeheader()
                        writer.writerow(dict(zip(field_names, history)))
                    if not os.path.exists("saved_weights"):
                        os.mkdir("saved_weights")
                    self.d_s.save_weights(self.__weights_d_s, True)
                    self.d_t.save_weights(self.__weights_d_t, True)
                    self.combined.save_weights(self.__weights_combined, True)
                    self.g_st.save_weights(self.__weights_g_st, True)
                    self.g_ts.save_weights(self.__weights_g_ts, True)
                    if pck_acc and (not best_pck_acc or pck_acc > best_pck_acc):
                        best_pck_acc = pck_acc
                        self.d_s.save_weights(self.__weights_d_s_best, True)
                        self.d_t.save_weights(self.__weights_d_t_best, True)
                        self.combined.save_weights(self.__weights_combined_best, True)
                        self.g_st.save_weights(self.__weights_g_st_best, True)
                        self.g_ts.save_weights(self.__weights_g_ts_best, True)

    def get_data_t(self, data_t_path, batch_size, i):
        input_data = sorted([os.path.join(data_t_path, filename) for filename in os.listdir(data_t_path)], key=lambda k: k.lower())
        size = len(input_data)
        if (i + 1) * batch_size % size == 0:
            input_data = input_data[(i * batch_size % size):size]
        elif i * batch_size % size < (i + 1) * batch_size % size:
            input_data = input_data[(i * batch_size % size):((i + 1) * batch_size % size)]
        else:
            part_1 = input_data[(i * batch_size % size):size]
            part_2 = input_data[0:((i + 1) * batch_size % size)]
            input_data = part_1 + part_2
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
    
    def get_imgs_t_for_visualization(self, imgs_vis_t_path, batch_size, i):
        input_data = sorted([os.path.join(imgs_vis_t_path, filename) for filename in os.listdir(imgs_vis_t_path)], key=lambda k: k.lower())
        size = len(input_data)
        if (i + 1) * batch_size % size == 0:
            input_data = input_data[(i * batch_size % size):size]
        elif i * batch_size % size < (i + 1) * batch_size % size:
            input_data = input_data[(i * batch_size % size):((i + 1) * batch_size % size)]
        else:
            part_1 = input_data[(i * batch_size % size):size]
            part_2 = input_data[0:((i + 1) * batch_size % size)]
            input_data = part_1 + part_2
        batch_x = []
        for data in input_data:
            img = imread(data)
            batch_x.append(crop_and_resize_image(img))
        return np.asarray(batch_x)

    def sample_images(self, epoch, batch_i, data_s, data_t):
        if not os.path.exists("%s/smt/" % smt_progress_images_out_dir):
            os.makedirs("%s/smt/" % smt_progress_images_out_dir)
        r, c = 2, 3

        data_s = np.expand_dims(data_s[0], 0)
        data_t = np.expand_dims(data_t[0], 0)
        fake_t = self.g_st.predict_on_batch(data_s)
        fake_s = self.g_ts.predict_on_batch(data_t)

        reconstr_s = self.g_ts.predict_on_batch(fake_t)
        reconstr_t = self.g_st.predict_on_batch(fake_s)

        gen_imgs = np.concatenate([data_s, fake_t, reconstr_s, data_t, fake_s, reconstr_t])
        tmp = gen_imgs[:, :, :, 0]

#         gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(tmp[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("%s/smt/smt_epoch_%03d_batch_%05d.png" % (smt_progress_images_out_dir, epoch, batch_i))
        plt.close()

    def sample_prediction(self, epoch, batch_size, i, data_t):
        if not os.path.exists("%s/pose/" % smt_progress_images_out_dir):
            os.makedirs('%s/pose/' % smt_progress_images_out_dir)
        if not os.path.exists("%s/joints/" % smt_progress_images_out_dir):
            os.makedirs('%s/joints/' % smt_progress_images_out_dir)
        tmp = self.g_ts.predict_on_batch(np.expand_dims(data_t[0], 0))
        results = self.task_model.predict_on_batch(tmp)
        utils.save_pose(results, self.get_imgs_t_for_visualization(self.target_imgs_for_visualization, batch_size, i - 1),
                        "%s/pose/pose_epoch_%03d_batch_%05d" % (smt_progress_images_out_dir, epoch, i))
        utils.save_joints(results, self.get_imgs_t_for_visualization(self.target_imgs_for_visualization, batch_size, i - 1),
                          "%s/joints/joints_epoch_%03d_batch_%05d" % (smt_progress_images_out_dir, epoch, i))

    def build_SMT(self, shapes, create_model_png=False
                       ):
        source_shape, target_shape = shapes
        g_st = build_generator(input_shape=(256, 256, 1), output_shape=(256, 256, 1), name="g_st")
        g_ts = build_generator(input_shape=(256, 256, 1), output_shape=(256, 256, 1), name="g_ts")
        if create_model_png:
            plot_model(g_st, "g_st.png", True)
        if create_model_png:
            plot_model(g_ts, "g_ts.png", True)
    
        d_t = build_discriminator_target(input_shape=(256, 256, 1), name="d_t")
        d_s = build_discriminator_source(input_shape=(256, 256, 1), name="d_s")
        if create_model_png:
            plot_model(d_t, "d_t.png", True)
        if create_model_png:
            plot_model(d_s, "d_s.png", True)
    
        lr = 0.0002
        
        optimizer_d = Adam(lr=lr)
        d_t.compile(loss="mse", optimizer=optimizer_d, metrics=['accuracy'])
        d_s.compile(loss="mse", optimizer=optimizer_d, metrics=['accuracy'])
    
        d_t.trainable = False
        d_s.trainable = False
    
        source_input = Input(shape=source_shape, name="data_s")
        fake_target = g_st(source_input)
        preal_target = d_t(fake_target)
        reco_source = g_ts(fake_target)
    
        target_input = Input(shape=target_shape, name="data_t")
        fake_source = g_ts(target_input)
        fake_source_prediction = self.task_model(fake_source)
        preal_source = d_s([fake_source, fake_source_prediction])
        reco_target = g_st(fake_source)
    
        prediction_recons = self.task_model(reco_source)
    
        loss = ["mse", "mse", "mae", "mae", "mae"]
        loss_weights = [1., 1., 10., 10., 1.]
        inputs = [source_input, target_input]
        outputs = [preal_source, preal_target, reco_source, reco_target, prediction_recons]
    
        combined = Model(inputs, outputs, name='adversarial')
        optimizer_g = Adam(lr=lr)
        combined.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer_g, metrics=['accuracy'])
        if create_model_png:
            plot_model(combined, "combined.png", True)
    
        return g_ts, g_st, d_s, d_t, combined

    
def encoder_layer(inputs, filters=16, kernel_size=3, strides=2, activation='relu', instance_norm=True):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    return x


def decoder_layer(inputs, paired_inputs, filters=16, kernel_size=3, strides=2, activation='relu', instance_norm=True):
    conv = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    x = concatenate([x, paired_inputs])
    return x


def build_generator(input_shape, output_shape=None, kernel_size=3, name=None):
    inputs = Input(shape=input_shape)
    channels = int(output_shape[-1])
    e1 = encoder_layer(inputs, 32, kernel_size=kernel_size, activation='leaky_relu', strides=1)
    e2 = encoder_layer(e1, 64, activation='leaky_relu', kernel_size=kernel_size)
    e3 = encoder_layer(e2, 128, activation='leaky_relu', kernel_size=kernel_size)
    e4 = encoder_layer(e3, 256, activation='leaky_relu', kernel_size=kernel_size)
    d1 = decoder_layer(e4, e3, 128, kernel_size=kernel_size)
    d2 = decoder_layer(d1, e2, 64, kernel_size=kernel_size)
    d3 = decoder_layer(d2, e1, 32, kernel_size=kernel_size)
    outputs = Conv2DTranspose(channels, kernel_size=kernel_size, strides=1, activation='sigmoid', padding='same')(d3)
    generator = Model(inputs, outputs, name=name)
    return generator


def build_discriminator_target(input_shape, kernel_size=3, patchgan=True, name=None):
    inputs = Input(shape=input_shape)
    x = encoder_layer(inputs, 32, kernel_size=kernel_size, activation='leaky_relu', instance_norm=False)
    x = encoder_layer(x, 64, kernel_size=kernel_size, activation='leaky_relu', instance_norm=False)
    x = encoder_layer(x, 128, kernel_size=kernel_size, activation='leaky_relu', instance_norm=False)
    x = encoder_layer(x, 256, kernel_size=kernel_size, strides=1, activation='leaky_relu', instance_norm=False)
    if patchgan:
        x = LeakyReLU(alpha=0.2)(x)
        outputs = Conv2D(1, kernel_size=kernel_size, strides=2, padding='same')(x)
    else:
        x = Flatten()(x)
        x = Dense(1)(x)
        outputs = Activation('linear')(x)
    discriminator = Model(inputs, outputs, name=name)
    return discriminator


def build_discriminator_source(input_shape, kernel_size=3, patchgan=True, name=None):
    inputs = Input(shape=input_shape)
    input_label = Input((64, 64, 15))
    x = encoder_layer(inputs, 32, kernel_size=kernel_size, activation='leaky_relu', instance_norm=False)
    y = encoder_layer(input_label, 32, kernel_size=kernel_size, strides=1, activation='leaky_relu', instance_norm=False)
    x = encoder_layer(x, 64, kernel_size=kernel_size, activation='leaky_relu', instance_norm=False)
    y = encoder_layer(y, 64, kernel_size=kernel_size, strides=1, activation='leaky_relu', instance_norm=False)
    x = concatenate([x, y])
    x = encoder_layer(x, 128, kernel_size=kernel_size, activation='leaky_relu', instance_norm=False)
    x = encoder_layer(x, 256, kernel_size=kernel_size, strides=1, activation='leaky_relu', instance_norm=False)
    if patchgan:
        x = LeakyReLU(alpha=0.2)(x)
        outputs = Conv2D(1, kernel_size=kernel_size, strides=2, padding='same')(x)
    else:
        x = Flatten()(x)
        x = Dense(1)(x)
        outputs = Activation('linear')(x)
    discriminator = Model([inputs, input_label], outputs, name=name)
    return discriminator

