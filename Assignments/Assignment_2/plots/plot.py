from cProfile import label
import matplotlib.pyplot as plt

x = [256, 512, 1024, 2048]

y1 = [39.10, 41.79, 48.64, 47.69]
y2 = [42.90, 46.63, 50.85, 52.89]
y3 = [82.08, 99.81, 127, 138.2]
y4 = [87.7, 112.67, 134.81, 146.57]

plt.xlabel("Image size")
plt.ylabel("Scaling factor")
plt.plot(x, y1, label="Kernel-1")
plt.plot(x, y2, label="Kernel-3")
plt.plot(x, y3, label="Kernel-4")
plt.plot(x, y4, label="Kernel-5")
plt.legend()
plt.rcParams.update({'font.size': 22})
plt.show()