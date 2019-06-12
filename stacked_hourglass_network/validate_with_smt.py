from stacked_hourglass_network.hourglass_network import HourglassNetwork
from sensor_model_translation.smt import build_generator

val_data_path = "../smt_dataset/real/val/depth/"
val_annotations_path = "../smt_dataset/real/val/labels/"
batch_size = 4
hgn_weights = "weights/best_weights_epoch_43_sim_data_only.hdf5"
g_ts_weights = "../sensor_model_translation/weights/best_weights_g_ts_epoch_428.hdf5"
number_of_hourglasses = 4
val_pck_threshold = 2

if __name__ == '__main__':
    g_ts = build_generator(input_shape=(256, 256, 1), output_shape=(256, 256, 1))
    g_ts.load_weights(g_ts_weights)
    hgn = HourglassNetwork(number_of_hourglasses=number_of_hourglasses, weights=hgn_weights)
    print ("Avg. pixel distance metric: %f" % hgn.calc_distance_metric(val_data_path, val_annotations_path,
                                                                 batch_size, g_ts=g_ts))
    print ("PCK metric (distance < %i pixels): %f" % (val_pck_threshold, hgn.calc_threshold_metric(val_data_path,
                                                            val_annotations_path, batch_size, threshold=val_pck_threshold, g_ts=g_ts)))
            
