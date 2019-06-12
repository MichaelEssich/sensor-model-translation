from stacked_hourglass_network.hourglass_network import HourglassNetwork

val_data_path = "../smt_dataset/real/val/depth/"
val_annotations_path = "../smt_dataset/real/val/labels/"
batch_size = 12
hgn_weights = "weights/best_weights_epoch_43_sim_data_only.hdf5"  # or for model trained on real data: "weights/best_weights_epoch_45_real_data_only.hdf5"
val_pck_threshold = 2

if __name__ == '__main__':
    hgn = HourglassNetwork(number_of_hourglasses=4, weights=hgn_weights)
    print ("Avg. pixel distance metric: %f" % hgn.calc_distance_metric(val_data_path, val_annotations_path, batch_size))
    print ("PCK metric (distance < %i pixels): %f" % (val_pck_threshold, hgn.calc_threshold_metric(val_data_path,
                                                                    val_annotations_path, batch_size, threshold=val_pck_threshold)))
            
