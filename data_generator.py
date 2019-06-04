import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

shape = 100000
a = np.linspace(-100, 100, shape)
b = np.linspace(-10, 10, shape)
c = np.linspace(20, 111, shape)
d = np.linspace(-15, 15, shape)
e = np.linspace(-12, 8, shape)
noise_a = np.random.normal(-20, 20, a.shape)
noise_b = np.random.normal(-20, 20, a.shape)
noise_c = np.random.normal(-20, 20, a.shape)
noise_d = np.random.normal(-20, 20, a.shape)
noise_e = np.random.normal(-20, 20, a.shape)

a += noise_a
b += noise_b
c += noise_c
d += noise_d
e += noise_e

y3 = (10.1 * a) + (2.3 * b) ** 2 - (0.5 * c) * 2
y4 = y3 + 14.8 * d
y5 = y4 + 4.3 * e

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X, attr_b, attr_a, s=1)
# plt.show()

df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'actual': y3}).transpose()
df.to_csv('3d.txt', header=False, index=False, sep=',', mode='w')
df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'actual': y4}).transpose()
df.to_csv('4d.txt', header=False, index=False, sep=',', mode='w')
df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'actual': y5}).transpose()
df.to_csv('5d.txt', header=False, index=False, sep=',', mode='w')

print('done')
