import sys

from plot.utils.time_acc_base import plot_static_time_acc

fed_async = [10.0, 11.57, 15.24, 18.0, 22.38, 22.38, 25.51, 25.14, 25.89, 26.65, 28.0, 28.92, 29.24, 29.95, 30.22, 30.81, 29.14, 45.51, 45.51, 45.19, 44.43, 44.7, 46.0, 45.62, 46.49, 46.49, 45.08, 45.57, 45.35, 44.65, 44.86, 45.3, 45.08, 45.14, 45.89, 46.22, 46.32, 46.05, 45.89, 46.22, 45.95, 45.73, 45.41, 45.35, 46.11, 45.73, 45.24, 45.84, 45.84, 45.89, 45.68, 46.59, 47.08, 46.86, 46.92, 47.24, 47.24, 46.59, 46.65, 46.86, 47.19, 47.46, 47.95, 47.51, 46.49, 46.59, 46.81, 46.54, 47.19, 47.08, 47.76, 47.76, 46.27, 46.27, 46.32, 46.46, 46.3, 46.62, 46.73, 46.95, 46.51, 46.62, 46.11, 46.0, 46.0, 46.16, 46.05, 46.84, 46.65, 46.14, 46.3, 46.14, 46.78, 46.78, 46.46, 46.57, 46.73, 46.05]
fed_async_f05 = [10.0, 12.27, 16.05, 18.0, 22.65, 22.65, 26.65, 26.65, 27.24, 28.81, 29.08, 30.7, 30.92, 30.81, 30.97, 32.16, 32.22, 44.38, 44.22, 45.84, 45.84, 46.49, 46.49, 46.05, 46.65, 46.32, 46.05, 46.27, 46.97, 46.97, 45.51, 45.89, 45.73, 48.0, 47.3, 48.0, 47.73, 47.73, 47.14, 47.19, 47.84, 47.95, 47.35, 47.35, 47.78, 48.0, 47.41, 47.3, 47.3, 46.92, 46.92, 47.62, 47.62, 47.89, 47.35, 46.92, 47.08, 46.76, 46.97, 46.97, 46.97, 46.54, 46.22, 46.22, 45.84, 46.16, 46.16, 46.38, 46.38, 46.0, 46.0, 47.08, 47.03, 46.27, 46.54, 46.54, 46.76, 46.76, 46.7, 46.43, 46.54, 46.16, 46.92, 46.97, 46.7, 46.49, 46.32, 46.32, 46.54, 46.49, 47.14, 47.3, 47.3, 46.7, 46.7, 46.49, 46.54, 47.14]
fed_async_f10 = [10.16, 12.22, 16.65, 17.89, 21.35, 21.35, 22.32, 24.11, 27.24, 27.24, 27.03, 28.97, 29.3, 30.0, 30.38, 30.97, 30.54, 36.59, 38.65, 38.76, 41.57, 42.11, 42.65, 42.38, 42.86, 42.86, 41.41, 41.41, 41.73, 43.35, 43.35, 42.81, 42.38, 42.16, 42.59, 46.43, 46.76, 47.08, 46.92, 46.22, 46.43, 45.68, 45.51, 45.3, 45.3, 45.68, 45.62, 46.0, 47.03, 46.65, 46.38, 46.7, 45.68, 47.08, 48.49, 48.43, 48.86, 48.32, 48.11, 47.78, 47.62, 47.62, 47.3, 47.14, 46.97, 47.19, 47.19, 47.3, 46.86, 46.86, 47.24, 45.51, 46.38, 46.05, 46.32, 45.57, 46.11, 46.05, 45.84, 46.32, 45.89, 46.32, 45.89, 45.89, 45.89, 46.7, 46.0, 45.95, 46.32, 46.16, 46.76, 46.27, 46.32, 45.84, 46.0, 46.32, 46.16, 45.89]
fed_async_f15 = [10.0, 12.59, 19.08, 19.35, 21.57, 23.62, 26.11, 26.38, 28.11, 28.76, 29.14, 29.14, 30.49, 30.49, 30.81, 30.97, 30.0, 32.97, 36.11, 39.84, 39.68, 40.05, 40.76, 41.24, 40.97, 40.22, 40.54, 41.3, 40.86, 41.08, 40.59, 41.03, 41.19, 46.43, 46.22, 46.38, 46.76, 46.81, 46.86, 46.92, 47.51, 47.51, 47.14, 47.14, 47.35, 46.16, 46.16, 46.7, 46.7, 47.46, 47.19, 47.08, 47.24, 46.59, 46.38, 46.49, 46.92, 46.92, 46.97, 47.24, 47.3, 47.08, 46.65, 47.3, 47.08, 47.08, 48.43, 48.43, 48.16, 47.95, 48.22, 47.3, 47.41, 47.14, 46.97, 46.97, 46.97, 47.24, 46.76, 47.19, 46.97, 47.3, 48.0, 48.0, 47.24, 47.24, 46.97, 46.92, 46.76, 47.46, 47.46, 46.05, 46.05, 46.97, 47.46, 47.97, 45.74, 45.74]

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_static_time_acc("", 3, 300, fed_async, fed_async_f05, fed_async_f10, fed_async_f15, save_path, plot_size="M")
