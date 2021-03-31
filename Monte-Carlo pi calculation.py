"""
The steps to estimate π are very simple:
    1. First, we generate some random points inside the square.
    2. Then we can calculate the number of points that fall inside the circle by using the equation   
                x^2 + y^2 <= size.
    3. Then we calculate the value of π by multiplying four to the division of the number of points inside the circle to the number of points inside the square.
    4. If we increase the number of samples (number of random points), the better we can approximate

"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
# %matplotlib inline

square_size = 1
points_inside_circle = 0
points_inside_square = 0
sample_size = 1000
arc = np.linspace(0, np.pi/2, 100)

"""
a function called generate_points(), which generates random points
inside the square.
"""
def generate_points(size):
    x = random.random()*size
    y = random.random()*size
    return (x, y)

"""
a function called is_in_circle(), which will check if the point we generated falls within the circle'
"""
def is_in_circle(point, size):
    return math.sqrt(point[0]**2 + point[1]**2) <= size

"""
a function for calculating the π value:

"""
def compute_pi(points_inside_circle, points_inside_square):
    return 4 * (points_inside_circle / points_inside_square)

"""
Then for the number of samples, 
we generate some random points inside the square and increment our points_inside_square variable.
then we will check if the points we generated lie inside the circle. 

If yes, then we increment the points_inside_circle variable

"""
plt.axes().set_aspect('equal')
plt.plot(1*np.cos(arc), 1*np.sin(arc))
for i in range(sample_size):
    point = generate_points(square_size)
    plt.plot(point[0], point[1], 'c.')
    points_inside_square += 1
    if is_in_circle(point, square_size):
        points_inside_circle += 1


plt.show()
print("Approximate value of pi is {}".format(compute_pi(points_inside_circle, points_inside_square)))

print(" The greater the sampling size, the better our approximation will be. ")