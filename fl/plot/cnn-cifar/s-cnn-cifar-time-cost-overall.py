import sys

from plot.utils.time_acc_base import plot_time_cost

# iot-2-network
fed_async = [15.99, 16.98, 16.92, 16.46, 16.94, 18.07, 16.93, 17.55, 17.34, 18.05, 18.11, 18.78, 18.3, 18.45, 18.78, 19.01, 19.32, 19.03, 18.63, 19.06, 20.09, 18.54, 19.44, 19.28, 19.17, 19.16, 19.25, 19.5, 18.64, 18.34, 18.46, 18.44, 17.52, 17.29, 17.91, 18.42, 18.58, 18.29, 18.48, 17.96, 18.5, 18.01, 18.33, 18.58, 18.65, 18.08, 17.91, 18.38, 18.31, 18.17]
fed_avg = [59.85, 52.45, 52.42, 53.39, 54.36, 56.64, 55.43, 56.7, 57.04, 58.71, 58.33, 58.97, 59.74, 59.09, 59.37, 59.42, 60.39, 59.83, 60.97, 60.93, 59.96, 59.95, 59.81, 59.38, 61.61, 61.72, 60.37, 60.87, 60.51, 60.39, 60.89, 60.87, 60.65, 61.05, 59.94, 59.37, 60.65, 61.24, 60.9, 60.86, 61.37, 60.81, 60.7, 61.06, 60.27, 61.2, 60.48, 60.68, 61.06, 60.69]
fed_localA = [26.34, 109.73, 27.3, 28.03, 28.4, 28.19, 28.47, 28.48, 28.64, 29.17, 29.51, 864.5, 28.81, 28.92, 28.99, 28.68, 28.98, 28.83, 28.66, 29.09, 29.73, 912.2, 28.45, 28.7, 28.84, 28.9, 29.08, 28.96, 28.93, 28.73, 29.32, 913.53, 29.09, 28.81, 29.03, 28.86, 29.18, 28.68, 28.88, 29.38, 29.49, 913.54, 28.49, 28.79, 28.94, 28.43, 28.81, 28.5, 28.85, 28.93]
fed_sync = [51.66, 49.17, 49.58, 49.59, 50.65, 51.63, 53.02, 53.42, 55.12, 55.76, 55.2, 55.46, 56.24, 56.44, 57.26, 57.61, 58.97, 56.79, 58.32, 56.64, 57.87, 58.18, 57.97, 58.41, 59.22, 57.7, 59.02, 57.51, 58.81, 59.19, 57.9, 57.89, 58.06, 58.41, 58.37, 57.41, 57.55, 58.41, 58.25, 57.58, 58.14, 57.82, 57.61, 57.79, 57.95, 59.11, 57.9, 58.41, 57.9, 57.55]
local_train = [12.59, 12.54, 12.64, 12.75, 13.2, 13.65, 13.63, 13.63, 14.06, 13.87, 14.03, 14.31, 14.11, 13.83, 14.47, 14.34, 14.07, 14.42, 14.49, 14.7, 14.52, 14.33, 14.48, 14.47, 14.37, 14.46, 14.41, 14.54, 14.17, 14.35, 14.63, 14.19, 14.57, 14.39, 14.52, 14.44, 14.42, 14.67, 14.56, 14.42, 14.73, 14.41, 14.71, 14.52, 14.7, 14.64, 14.31, 14.33, 14.2, 14.4]
fed_asofed = [12.67, 12.93, 14.4, 15.62, 16.54, 17.26, 17.71, 17.78, 18.36, 18.78, 18.83, 18.84, 18.97, 19.09, 18.89, 19.14, 19.23, 19.7, 19.44, 19.39, 19.57, 19.08, 19.62, 19.8, 19.82, 19.41, 19.93, 20.07, 19.62, 19.76, 19.56, 20.02, 19.99, 19.53, 19.63, 19.46, 19.36, 19.65, 19.54, 19.37, 20.15, 19.63, 19.35, 19.6, 19.54, 19.37, 19.59, 19.63, 19.45, 20.03]
fed_bdfl = [35.95, 36.09, 38.81, 40.99, 34.93, 31.28, 30.37, 30.24, 31.5, 32.08, 31.4, 31.66, 32.59, 31.89, 31.97, 32.6, 32.32, 32.84, 31.79, 32.77, 33.34, 32.48, 32.28, 32.91, 33.43, 32.14, 33.43, 32.87, 33.61, 34.13, 33.96, 33.08, 33.3, 33.06, 34.99, 32.93, 32.77, 32.86, 32.57, 33.13, 32.09, 32.74, 32.15, 32.51, 33.27, 33.42, 32.25, 33.84, 32.31, 32.06]

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_time_cost("", 150, fed_async, fed_avg, fed_sync, fed_localA, local_train, fed_asofed, fed_bdfl, save_path, plot_size="S")