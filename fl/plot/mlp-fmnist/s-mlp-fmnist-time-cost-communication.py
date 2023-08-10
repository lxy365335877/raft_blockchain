import sys

from plot.utils.time_acc_base import plot_time_cost

# iot-1-network
fed_async = [3.64, 3.65, 3.31, 3.6, 3.27, 3.6, 3.35, 4.47, 4.54, 4.57, 4.75, 4.28, 3.67, 2.72, 4.05, 4.51, 4.33, 4.2, 3.66, 2.62, 2.09, 2.6, 2.46, 2.8, 2.21, 2.25, 1.57, 2.09, 1.93, 2.87, 1.65, 2.33, 1.89, 2.05, 1.48, 1.91, 1.97, 2.08, 1.78, 1.94, 1.87, 2.21, 2.07, 2.55, 2.75, 2.7, 2.66, 2.82, 2.94, 2.53]
fed_avg = [0.05, 0.06, 0.05, 0.07, 0.06, 0.06, 0.06, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.06, 0.04, 0.05, 0.04, 0.04, 0.04, 0.05, 0.06, 0.05, 0.06, 0.06, 0.05, 0.06, 0.06, 0.04, 0.05, 0.05, 0.06, 0.06, 0.05, 0.06, 0.04, 0.05, 0.05, 0.05, 0.03, 0.05, 0.06, 0.04, 0.05, 0.05, 0.06, 0.05, 0.04]
fed_localA = [0.03, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
fed_sync = [2.24, 2.91, 2.86, 2.91, 2.94, 2.86, 2.97, 2.86, 2.91, 3.1, 2.86, 2.95, 2.93, 3.0, 2.92, 2.92, 2.76, 3.05, 2.88, 3.0, 2.92, 2.86, 2.84, 2.91, 2.98, 2.94, 2.87, 2.96, 2.9, 2.93, 2.78, 2.81, 2.92, 2.8, 2.92, 2.94, 2.89, 2.89, 2.96, 3.19, 2.82, 2.93, 2.82, 2.88, 3.06, 2.92, 3.02, 2.78, 2.84, 2.92]
local_train = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
fed_asofed = [0.09, 0.51, 0.07, 0.05, 0.06, 0.08, 0.05, 0.08, 0.05, 0.04, 0.04, 0.04, 0.05, 0.06, 0.04, 0.05, 0.05, 0.06, 0.06, 0.04, 0.04, 0.05, 0.04, 0.04, 0.04, 0.05, 0.05, 0.09, 0.04, 0.04, 0.05, 0.05, 0.04, 0.05, 0.06, 0.04, 0.05, 0.05, 0.04, 0.05, 0.05, 0.04, 0.04, 0.06, 0.06, 0.04, 0.03, 0.05, 0.03, 0.05]
fed_bdfl = [3.72, 7.02, 4.96, 7.53, 13.89, 7.04, 5.33, 6.66, 8.19, 3.4, 4.54, 6.2, 4.97, 4.7, 7.05, 4.92, 3.48, 5.93, 4.98, 5.05, 9.71, 5.56, 5.2, 10.99, 15.22, 5.66, 3.16, 4.9, 10.79, 3.22, 6.14, 5.69, 6.11, 6.09, 4.69, 8.37, 6.45, 7.36, 3.06, 5.66, 5.3, 10.9, 9.98, 4.77, 4.22, 3.41, 3.03, 5.79, 4.27, 1.94]

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_time_cost("", 8, fed_async, fed_avg, fed_sync, fed_localA, local_train, fed_asofed, fed_bdfl, save_path, plot_size="S")