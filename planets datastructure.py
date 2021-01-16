#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sun_radius = 696340
df_planets = pd.read_csv("planets.csv")
df_moons = pd.read_csv("moons.csv")

# [
#   (name, radius, dist_from_sun,
#       [
#           (moon name, radius, dist_from_planet), ...
#       ]
#   ), ...
# ]
# planets sorted by distance from sun, moons sorted by distance from planet
data = []
planet_to_index = dict()

planet_names = df_planets["PLANET"].tolist()
planet_radius = df_planets["RADIUS (km)"].tolist()
planet_dist = df_planets["DIST FROM SUN (km)"].tolist()
for i in range(len(planet_names)):
    data.append((planet_names[i], planet_radius[i], planet_dist[i], []))
    planet_to_index[planet_names[i]] = i

moon_names = df_moons["MOON"].tolist()
moon_planet = df_moons["PLANET"].tolist()
moon_radius = df_moons["RADIUS (km)"].tolist()
moon_dist = df_moons["DIST FROM PLANET (km)"].tolist()

for i in range(len(moon_names)):
    planet_index = planet_to_index[moon_planet[i]]
    data[planet_index][3].append((moon_names[i], moon_radius[i], moon_dist[i]))

print(data)

""" linear """
# A = np.array([[planet_dist[0], 1], [planet_dist[-1], 1]])
# b = np.array([[3440/9], [3440 - 3440/9]])
# res = np.linalg.inv(A) @ b
# s = res[0, 0]
# h = res[1, 0]
# print(s, h)
# centers = s*np.array([0] + planet_dist) + h
#
# # 250 and 50
# A = np.array([[planet_radius[0], 1], [planet_radius[-1], 1]])
# b = np.array([[2], [5]])
# res = np.linalg.inv(A) @ b
# s = res[0, 0]
# h = res[1, 0]
# print(s, h)
# radii = s*np.array([sun_radius] + planet_radius) + h

""" logarithm """
# A = np.array([[np.log(planet_radius[0]), 1], [np.log(planet_radius[-1]), 1]])
# b = np.array([[25], [150]])
# res = np.linalg.inv(A) @ b
# s = res[0, 0]
# h = res[1, 0]
# print(s, h)
# radii = s*np.log(planet_radius) + h
#
# A = np.array([[np.log(planet_dist[0]), 1], [np.log(planet_dist[-1]), 1]])
# b = np.array([[650], [3440 - 650]])
# res = np.linalg.inv(A) @ b
# s = res[0, 0]
# h = res[1, 0]
# print(s, h)
# centers = s*np.log(planet_dist) + h

""" power """
dist_power = 0.225
A = np.array([[np.power(planet_dist[0], dist_power), 1], [np.power(planet_dist[-1], dist_power), 1]])
b = np.array([[575], [3440 - 575/2]])
res = np.linalg.inv(A) @ b
dist_s = res[0, 0]
dist_h = res[1, 0]
print(dist_s, dist_h)
centers = dist_s*np.power(planet_dist, dist_power) + dist_h
centers = np.insert(centers, 0, dist_h)

interval = 1e8
dists = np.arange(0, int(planet_dist[-1]/interval)*2)*1e8
dists_scale = dist_s*np.power(dists, dist_power) + dist_h

radii_power = 1
A = np.array([[np.power(planet_radius[0], radii_power), 1], [np.power(planet_radius[4], radii_power), 1]])
b = np.array([[10], [125]])
res = np.linalg.inv(A) @ b
radius_s = res[0, 0]
radius_h = res[1, 0]
print(radius_s, radius_h)
radii = radius_s*np.power(planet_radius, radii_power) + radius_h
radii = np.insert(radii, 0, radius_s*np.power(sun_radius, radii_power) + radius_h)

# print(centers[0] + radii[0])
# print(planet_dist[0])
# print(planet_dist[-1])

""" moons """
moon_radii = [[]]*8
moon_dists = [[]]*8
for i in range(8):
    planet_moon_radii = []
    planet_moon_dists = []
    for tuple in data[i][3]:
        # (moon_names[i], moon_radius[i], moon_dist[i])
        planet_moon_radii.append(tuple[1])
        planet_moon_dists.append(tuple[2])
    moon_radii[i] = planet_moon_radii
    moon_dists[i] = planet_moon_dists

min_planet_dist = moon_dists[6][0]
max_planet_dist = moon_dists[5][6]
A = np.array([[np.power(min_planet_dist, dist_power), 1], [np.power(max_planet_dist, dist_power), 1]])
b = np.array([[100], [500]])
res = np.linalg.inv(A) @ b
moon_dist_s = res[0, 0]
moon_dist_h = res[1, 0]
print(moon_dist_s, moon_dist_h)

# account for sun
moon_radii = [[]] + moon_radii
moon_dists = [[]] + moon_dists

from PIL import Image, ImageDraw

background = (1, 28, 44)
foreground = (249, 123, 35)
image = Image.new('RGBA', (3440, 1440), background)

""" draw stars """
for i in range(1000):
    x = np.random.randint(0, 3440)
    y = np.random.randint(0, 1440)
    image.putpixel((x, y), (255, 255, 255))

""" draw planets + sun """
for i in range(centers.shape[0]):
    draw = ImageDraw.Draw(image)
    bound = (centers[i]-radii[i], 1440/2-radii[i], centers[i]+radii[i], 1440/2+radii[i])
    draw.ellipse(bound, fill = foreground, outline = foreground)

    if i > 4 or i == 0:
        width = 8 # width of circles
        draw = ImageDraw.Draw(image)
        bound = (centers[i]-radii[i]+width, 1440/2-radii[i]+width, centers[i]+radii[i]-width, 1440/2+radii[i]-width)
        draw.ellipse(bound, fill = background, outline = background)

    for j in range(len(moon_radii[i])):
        moon_center_x = centers[i]
        moon_center_y = 720 + moon_dist_s*np.power(moon_dists[i][j], dist_power) + moon_dist_h
        moon_radius = radius_s*np.power(moon_radii[i][j], radii_power) + radius_h
        print(moon_center_x, moon_center_y)

        draw = ImageDraw.Draw(image)
        bound = (
            moon_center_x-moon_radius,
            moon_center_y-moon_radius,
            moon_center_x+moon_radius,
            moon_center_y+moon_radius
        )
        draw.ellipse(bound, fill = foreground, outline = foreground)


""" draw scale """
for i in range(len(dists_scale)):
    location = round(dists_scale[i])
    draw = ImageDraw.Draw(image)
    if i%10 == 0:
        # draw.line((location, 1440-50, location, 1440), fill=foreground)
        draw.line((location, 50, location, 0), fill=foreground)
    else:
        # draw.line((location, 1440-25, location, 1440), fill=foreground)
        draw.line((location, 25, location, 0), fill=foreground)

image.show()
# image.save('wallpaper.png')
