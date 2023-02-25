import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

c = np.eye((6))
print(c)

F = np.arange(4)
G = np.array([[1, 4, 5, 6, 1], [2, 5, 6, 6, 1], [2, 3, 1, 1, 1], [8, 12, 14, 20, 1]])
##print(G[:,2])

H = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#print(H)
#print(np.shape(H))
H2 = np.reshape(H,(5,2))
#print(H2)
M = np.linspace(1, 17, 100)
#print(np.shape(M))

M2 = M[np.newaxis, :]
#print(M2)


#z = np.array([[10, 0, 0], [1, 11, 1], [2, 2, 12]])
#print(np.sum(z, axis=0))



Z = np.array([[10, 0, 0], [1, 11, 1], [2, 2, 12]])
b = np.array([[2], [1], [10]])

print(np.linalg.solve(Z, b))
"""
"""
url = 'https://github.com/uu-sml/course-sml-public/raw/master/data/auto.csv'
Auto = pd.read_csv(url)
print(f'Auto.shape: {Auto.shape}')
print(Auto)
print(Auto.describe())
print(Auto.info())
print(Auto.head())
print(Auto.shape)
"""

""""
year_data = Auto.groupby('year').mean().reset_index()
print(year_data)
year = year_data[['year']].to_numpy()
print(year)
acceleration = year_data[['acceleration']].to_numpy()
print(acceleration)


plt.figure(1)
plt.plot(year, acceleration, 'g-*', label='Mean acceleration by Year')
plt.legend()
plt.title('Mean acceleration over Time')
plt.xlabel('Year')
plt.ylabel('Acceleration')
#plt.savefig('dinosaur_fossil.png')   # you can use this command to save a figure to the main project folder
plt.show()

"""



from mpl_toolkits.mplot3d import Axes3D # library to create a 3D plot

# Data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = -(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x**2 + y**2)/np.pi)))

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
surf = ax.plot_surface(x, y, z, cmap = 'coolwarm')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
cbar = fig.colorbar(surf, shrink = 0.5, aspect = 5)
plt.show()