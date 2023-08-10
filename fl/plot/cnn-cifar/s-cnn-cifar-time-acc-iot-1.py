import sys

from plot.utils.time_acc_base import plot_time_acc

fed_async = [10.05, 12.32, 21.14, 22.92, 28.38, 29.03, 32.27, 32.27, 33.62, 34.11, 34.49, 35.14, 35.41, 37.35, 36.65, 38.32, 38.27, 44.76, 44.43, 45.46, 45.35, 46.16, 46.27, 44.0, 44.0, 44.97, 44.97, 44.81, 45.41, 45.35, 44.7, 45.14, 45.78, 45.35, 46.11, 45.41, 45.73, 45.73, 44.43, 45.19, 44.81, 45.08, 45.51, 45.51, 45.24, 44.81, 44.97, 45.24, 44.97, 44.97, 45.14, 45.41, 45.19, 44.32, 44.59, 44.43, 44.81, 44.05, 44.27, 44.7, 44.43, 44.54, 44.86, 44.7, 44.81, 45.51, 44.86, 45.89, 44.97, 44.81, 45.35, 44.49, 44.32, 44.59, 44.32, 44.0, 43.51, 44.32, 44.38, 44.65, 45.19, 44.59, 44.38, 44.16, 44.76, 44.7, 44.76, 44.86, 44.97, 45.62, 45.03, 45.03, 44.59, 44.65, 44.22, 44.59, 43.95, 44.38]
fed_avg = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 13.14, 13.14, 13.14, 26.32, 26.32, 26.32, 26.32, 26.32, 26.32, 26.32, 26.32, 26.32, 26.32, 26.32, 26.32, 26.32, 26.32, 26.32, 28.38, 28.38, 30.65, 32.49, 32.49, 32.49, 32.49, 32.49, 32.49, 32.49, 32.49, 32.49, 32.49, 32.49, 32.49, 32.49, 32.49, 34.32, 34.32, 34.32, 41.84, 41.84, 41.84, 41.84, 41.84, 41.84, 41.84, 41.84, 41.84, 41.84, 41.84, 41.84, 41.84, 41.84, 41.84, 42.76, 42.76, 43.35, 44.92, 44.92, 44.92, 44.92, 44.92, 44.92, 44.92, 44.92, 44.92, 44.92, 44.92, 44.92, 44.92, 44.92, 44.92, 44.86, 44.86, 44.86, 45.89, 45.89, 45.89, 45.89, 45.89, 45.89, 45.89]
fed_sync = [9.08, 9.08, 9.08, 9.08, 9.08, 9.08, 9.08, 9.08, 9.08, 9.08, 9.08, 9.08, 9.08, 9.08, 9.08, 9.08, 9.08, 26.92, 26.92, 26.92, 26.92, 26.92, 26.92, 26.92, 26.92, 26.92, 26.92, 26.92, 26.92, 26.92, 26.92, 26.92, 26.92, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 43.3, 43.3, 43.3, 43.3, 43.3, 43.3, 43.3, 43.3, 43.3, 43.3, 43.3, 43.3, 43.3, 43.3, 43.3, 43.3, 43.3, 45.84, 45.84, 45.84, 45.84, 45.84, 45.84, 45.84, 45.84, 45.84, 45.84, 45.84, 45.84, 45.84, 45.84, 45.84, 45.84, 45.84, 47.19, 47.19, 47.19, 47.19, 47.19, 47.19, 47.19, 47.19, 47.19, 47.19, 47.19, 47.19, 47.19, 47.19]
fed_localA = [10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 16.65, 23.19, 25.41, 31.78, 31.78, 32.59, 33.68, 33.78, 33.78, 35.41, 34.43, 34.22, 33.73, 33.73, 33.46, 33.46, 33.3, 33.41, 33.78, 35.24, 35.46, 35.46, 35.62, 35.62, 35.68, 35.68, 35.68, 35.68, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57, 39.57]
local_train = [10.05, 10.05, 21.84, 26.81, 30.76, 32.16, 33.35, 31.95, 32.27, 31.41, 32.59, 32.54, 32.81, 34.16, 34.32, 34.59, 34.81, 34.7, 34.81, 34.76, 34.81, 34.76, 34.65, 34.76, 34.76, 34.76, 34.81, 34.92, 34.92, 34.92, 34.92, 34.97, 38.43, 38.49, 38.54, 38.49, 38.54, 38.54, 38.54, 38.49, 38.49, 38.54, 38.59, 38.59, 38.65, 38.65, 38.7, 38.7, 39.73, 39.68, 39.68, 39.68, 38.99, 38.99, 38.92, 38.92, 38.99, 38.99, 38.2, 37.84, 37.84, 37.84, 37.84, 32.16, 32.16, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 37.84, 35.95, 35.95, 35.95, 35.95, 35.95, 35.95, 35.95, 35.95, 35.95, 35.95, 35.95, 35.95, 35.95, 35.95, 35.95, 35.95]
fed_asofed = [10.0, 14.92, 22.49, 27.19, 30.32, 32.3, 32.76, 32.92, 33.27, 33.59, 33.27, 33.11, 34.32, 35.76, 38.7, 38.11, 38.59, 44.43, 45.89, 46.65, 45.89, 45.24, 44.65, 46.49, 46.16, 46.65, 46.32, 45.24, 45.62, 45.95, 45.78, 45.68, 44.76, 44.43, 45.3, 44.32, 44.43, 45.24, 44.76, 43.78, 44.32, 45.73, 44.59, 45.35, 45.35, 45.08, 45.68, 44.59, 44.81, 45.14, 45.57, 44.97, 46.62, 46.28, 45.47, 46.35, 46.35, 47.3, 46.35, 46.76, 46.35, 46.35, 46.62, 46.62, 46.62, 46.76, 46.49, 46.49, 46.49, 47.16, 45.41, 45.41, 45.41, 45.27, 45.27, 45.27, 45.27, 45.27, 45.27, 45.41, 45.41, 45.41, 45.41, 45.41, 45.41, 45.41, 45.41, 45.41, 45.41, 47.03, 47.03, 47.03, 47.03, 47.03, 47.03, 47.03, 47.03, 47.03]
fed_bdfl = [10.0, 16.86, 23.08, 25.46, 27.95, 30.54, 31.73, 34.16, 34.86, 35.3, 35.95, 36.54, 37.41, 38.54, 38.11, 39.03, 39.46, 38.59, 37.46, 36.92, 37.24, 37.73, 37.84, 36.0, 36.22, 37.3, 37.3, 36.81, 44.49, 44.0, 43.89, 44.97, 44.59, 44.38, 44.16, 43.84, 44.49, 43.73, 44.92, 43.95, 43.46, 44.16, 43.73, 42.97, 42.86, 44.05, 44.86, 45.24, 43.68, 44.11, 44.49, 43.84, 43.84, 44.59, 44.97, 44.54, 44.11, 43.89, 43.78, 43.89, 44.7, 45.24, 45.24, 44.97, 44.59, 45.51, 45.19, 45.03, 44.38, 44.81, 45.3, 45.14, 45.3, 44.86, 45.62, 44.97, 44.59, 44.81, 44.38, 44.32, 44.65, 45.24, 45.84, 45.19, 44.81, 44.22, 43.78, 44.05, 43.89, 43.84, 43.95, 44.7, 45.78, 45.57, 44.92, 45.03, 44.92, 45.03]

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_time_acc("", 3, 300, fed_async, fed_avg, fed_sync, fed_localA, local_train, fed_asofed, fed_bdfl, save_path, plot_size="S")