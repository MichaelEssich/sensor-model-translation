from sensor_model_translation.smt import SMT

# When training SMT multiple weights are saved as SMT is a combination of multiple models.
# After training is completed, g_ts is the model for sensor model translation from target to source data,
# i.e. from Kinect 2 TM data to simulated data in our case,
# on which the task model prediction can be performed.

hgn_weights = "../stacked_hourglass_network/weights/best_weights_epoch_43_sim_data_only.hdf5" # use model trained on simulated data as basis to train translation to real data
number_of_hourglasses = 4
source_data_path = "../smt_dataset/sim/train/depth/"
source_annotations_path = "../smt_dataset/sim/train/labels/"
target_data_path = "../smt_dataset/real/train/depth/"
target_annotations_path_for_metric = "../smt_dataset/real/train/labels/"
target_imgs_for_visualization = "../smt_dataset/real/train/imgs/"
val_data_path = "../smt_dataset/real/val/depth/"
val_annotations_path = "../smt_dataset/real/val/labels/"
epoch_to_resume = 1
epochs_total = 500
batch_size = 4
sample_output_interval = 150
val_pck_threshold = 2

if __name__ == '__main__':
    smt = SMT(hgn_weights=hgn_weights,
                   source_data_path=source_data_path,
                   source_annotations_path=source_annotations_path,
                   target_data_path=target_data_path,
                   target_imgs_for_visualization=target_imgs_for_visualization, create_model_png=False,
                   epoch_to_resume=epoch_to_resume, number_of_hourglasses=number_of_hourglasses)
    smt.train(epochs=epochs_total, batch_size=batch_size, sample_interval=sample_output_interval,
              target_annotations_path_for_metric=target_annotations_path_for_metric,
              val_data_path=val_data_path, val_annotations_path=val_annotations_path,
              val_pck_threshold=val_pck_threshold)
    
