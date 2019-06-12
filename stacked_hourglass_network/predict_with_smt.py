from stacked_hourglass_network.hourglass_network import HourglassNetwork
from stacked_hourglass_network.utils import get_imgs_for_visualization, show_pose
from sensor_model_translation.smt import build_generator
import math, os

pred_data_path = "../data_samples_for_prediction/real/depth/"
pred_img_for_visualization_path = "../data_samples_for_prediction/real/imgs/"
batch_size = 8
hgn_weights = "weights/best_weights_epoch_43_sim_data_only.hdf5"
number_of_hourglasses = 4
g_ts_weights = "../sensor_model_translation/weights/best_weights_g_ts_epoch_428.hdf5"

if __name__ == '__main__':
    batches = int(math.ceil(len([name for name in os.listdir(pred_data_path)]) / float(batch_size)))
    g_ts = build_generator(input_shape=(256, 256, 1), output_shape=(256, 256, 1))
    g_ts.load_weights(g_ts_weights)
    task_model = HourglassNetwork(number_of_hourglasses=number_of_hourglasses, weights=hgn_weights)
    for i in range(batches):
        results = task_model.model_predict.predict_on_batch(g_ts.predict_on_batch(task_model.get_data_t(pred_data_path, batch_size, i)))
        show_pose(results, get_imgs_for_visualization(pred_img_for_visualization_path, batch_size, i))
