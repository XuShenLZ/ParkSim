import math

theta = math.radians(-118.6)
offset_x = 240.181
offset_y = 67.795
a = math.cos(theta)
b = math.sin(theta)
def translate(x, y):
    new_x = x * a - y * b + offset_x
    new_y = x * b + y * a + offset_y
    return new_x, new_y

print(translate(5, 5))
print(translate(28.53, 73.73))