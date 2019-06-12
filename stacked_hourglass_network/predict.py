from stacked_hourglass_network.hourglass_network import HourglassNetwork
from stacked_hourglass_network.utils import get_imgs_for_visualization, show_pose
import math, os

pred_data_path = "../data_samples_for_prediction/real/depth/" # or for sim data: "../data_samples_for_prediction/real/depth/"
pred_img_for_visualization_path = "../data_samples_for_prediction/real/imgs/" # or for sim data: "../data_samples_for_prediction/real/depth/"
batch_size = 12
weights = "weights/best_weights_epoch_43_sim_data_only.hdf5" # or for model trained on real data: "weights/best_weights_epoch_45_real_data_only.hdf5"
number_of_hourglasses = 4

if __name__ == '__main__':
    batches = int(math.ceil(len([name for name in os.listdir(pred_data_path)]) / float(batch_size)))
    task_model = HourglassNetwork(number_of_hourglasses=number_of_hourglasses, weights=weights)
    for i in range(batches):
        results = task_model.model_predict.predict_on_batch(task_model.get_data_t(pred_data_path, batch_size, i))
        show_pose(results, get_imgs_for_visualization(pred_img_for_visualization_path, batch_size, i))
