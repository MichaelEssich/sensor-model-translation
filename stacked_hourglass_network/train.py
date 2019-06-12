from stacked_hourglass_network.hourglass_network import HourglassNetwork

train_data_path = "../smt_dataset/sim/train/depth/"
train_annotations_path = "../smt_dataset/sim/train/labels/"
val_data_path = "../smt_dataset/sim/val/depth/"
val_annotations_path = "../smt_dataset/sim/val/labels/"
batch_size = 12
number_of_hourglasses = 4
epochs_total = 50
epoch_to_resume = 1
weights = None  # or load weights to resume training: "weights/best_model_no_DA_epoch_43_sim_data.hdf5"
val_pck_threshold = 2

if __name__ == '__main__':
    task_model = HourglassNetwork(weights=weights, number_of_hourglasses=number_of_hourglasses, create_model_png=False)
    task_model.train(epochs=epochs_total, epoch_to_resume=epoch_to_resume, batch_size=batch_size,
                 train_data_path=train_data_path, train_annotations_path=train_annotations_path,
                 val_data_path=val_data_path, val_annotations_path=val_annotations_path, val_pck_threshold=val_pck_threshold)
    
    
