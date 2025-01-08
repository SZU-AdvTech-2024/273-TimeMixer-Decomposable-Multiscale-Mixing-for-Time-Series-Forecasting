import numpy as np
import scipy
import h5py
import matplotlib.pyplot as plt

data = np.load("/home4/hwj/project/TimeMixer-main/results/long_term_forecast_test_none_TimeMixer_SST_sl10_pl5_dm16_nh4_el2_dl1_df32_fc1_ebtimeF_dtTrue_test_0/pred.npy")
true = np.load("/home4/hwj/project/TimeMixer-main/results/long_term_forecast_test_none_TimeMixer_SST_sl10_pl5_dm16_nh4_el2_dl1_df32_fc1_ebtimeF_dtTrue_test_0/true.npy")
diff = np.abs(data - true)
# print(data.shape)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# true = true.reshape(data.shape[0], data.shape[1], 216, 216)
data = data.reshape(data.shape[0], data.shape[1], 1440, 2880)
# rmse = rmse(true[:, :, :15, -26:], data[:, :, :15, -26:])
# print(rmse)
# zero = np.zeros_like(diff)
# rmse = rmse(zero, data)
# print(rmse)


# print(data.shape)
# plot_data = data[0, 0, :15, -26:]

plot_data = data[0, 0, :, :]

plt.imshow(plot_data)
plt.show()
# print(data[0, 0, :, :])
