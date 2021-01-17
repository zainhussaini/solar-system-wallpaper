#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


class WallpaperImage:
    background = (1, 28, 44)
    foreground = (249, 123, 35)

    def __init__(self, image_width=3440, image_height=1440):
        self.image_width = image_width
        self.image_height = image_height
        self.image = Image.new('RGBA', (image_width, image_height), WallpaperImage.background)

    def add_stars(self, count):
        # TODO: match actual stars and constellations
        for i in range(count):
            x = np.random.randint(0, self.image_width)
            y = np.random.randint(0, self.image_height)
            self.image.putpixel((x, y), (255, 255, 255))

    def draw_circle(self, pixel_x, pixel_y, pixel_r, foreground=True):
        if foreground == True:
            color = WallpaperImage.foreground
        else:
            color = WallpaperImage.background

        draw = ImageDraw.Draw(self.image)
        bound = (
            pixel_x - pixel_r,
            pixel_y - pixel_r,
            pixel_x + pixel_r,
            pixel_y + pixel_r)
        draw.ellipse(bound, fill=color, outline=color)

    def draw_empty_circle(self, pixel_x, pixel_y, pixel_r, thickness):
        self.draw_circle(pixel_x, pixel_y, pixel_r, True)
        self.draw_circle(pixel_x, pixel_y, pixel_r - thickness, False)

    def draw_scale_top(self, pixels_x, height_ones, height_tens):
        draw = ImageDraw.Draw(self.image)
        for i, pixel_x in enumerate(pixels_x):
            if i%10 == 0:
                draw.line((pixel_x, 0, pixel_x, height_tens), fill=WallpaperImage.foreground)
            else:
                draw.line((pixel_x, 0, pixel_x, height_ones), fill=WallpaperImage.foreground)

    def draw_scale_right(self, pixels_y, height_ones, height_tens):
        draw = ImageDraw.Draw(self.image)
        for i, pixel_y in enumerate(pixels_y):
            if i%10 == 0:
                height = height_tens
            else:
                height = height_ones

            draw.line((
                self.image_width - height,
                self.image_height/2 + pixel_y,
                self.image_width,
                self.image_height/2 + pixel_y), fill=WallpaperImage.foreground)
            draw.line((
                self.image_width - height,
                self.image_height/2 - pixel_y,
                self.image_width,
                self.image_height/2 - pixel_y), fill=WallpaperImage.foreground)


    def show(self):
        self.image.show()

    def save(self):
        self.image.save('wallpaper.png')


sun_radius = 696340
df_planets = pd.read_csv('planets.csv')
df_moons = pd.read_csv('moons.csv')

df_planets.set_index("PLANET", inplace=True)
df_moons.set_index("MOON", inplace=True)

# list of tuples: (location x, location y, radius, solid)
circles = [(0, 0, sun_radius, False)]

for index, row in df_planets.iterrows():
    location_x = row["DIST FROM SUN (km)"]
    location_y = 0
    radius = row["RADIUS (km)"]
    solid = index in ("Mercury", "Venus", "Earth", "Mars")
    circles.append((location_x, location_y, radius, solid))

for index, row in df_moons.iterrows():
    location_x = df_planets["DIST FROM SUN (km)"][row["PLANET"]]
    location_y = row["DIST FROM PLANET (km)"]
    radius = row["RADIUS (km)"]
    circles.append((location_x, location_y, radius, True))

""" calculate radius parameters """
jupiter_radius_pixels = 150

jupiter_radius = df_planets["RADIUS (km)"]["Jupiter"]
sr = jupiter_radius_pixels / jupiter_radius

""" calculate distance y parameters """
max_moon_dist_pixels = 600
min_moon_dist_pixels = 150

# py = sy*np.power(y, ey)
# s = ?
# e = ?
# py1 = s*np.power(y1, e)
# py2 = s*np.power(y2, e)

py1 = max_moon_dist_pixels
py2 = min_moon_dist_pixels
y1 = max(df_moons["DIST FROM PLANET (km)"])
y2 = min(df_moons["DIST FROM PLANET (km)"])

A = np.array([[1/np.log(y1), 1], [1/np.log(y2), 1]])
b = np.array([[np.log(py1)/np.log(y1)], [np.log(py2)/np.log(y2)]])
res = np.linalg.inv(A) @ b
sy = np.exp(res[0][0])
ey = res[1][0]

""" calculate distance x parameters """
sun_visibility_pixels = 200
neptune_loc_pixels = 3000
mercury_dist_pixels = 200 # distance from sun

# px = dx + sx*np.power(x, ex)

""" NOTE THIS IS A UNCONSISTENT CHANGE!!! """
# dx = sun_visibility_pixels - sun_radius*sr
dx = sun_visibility_pixels

py1 = mercury_dist_pixels + sun_visibility_pixels - dx
py2 = neptune_loc_pixels - dx
y1 = df_planets["DIST FROM SUN (km)"]["Mercury"]
y2 = df_planets["DIST FROM SUN (km)"]["Neptune"]

A = np.array([[1/np.log(y1), 1], [1/np.log(y2), 1]])
b = np.array([[np.log(py1)/np.log(y1)], [np.log(py2)/np.log(y2)]])
res = np.linalg.inv(A) @ b
sx = np.exp(res[0][0])
ex = res[1][0]

""" make image """
image = WallpaperImage()
image.add_stars(2000)
for location_x, location_y, radius, solid in circles:
    location_x_pixel = dx + sx*np.power(location_x, ex)
    location_y_pixel = image.image_height/2 + sy*np.power(location_y, ey)
    radius_pixel = sr*radius

    """ NOTE THIS IS A UNCONSISTENT CHANGE!!! """
    if location_x == 0:
        location_x_pixel = sun_visibility_pixels - sun_radius*sr

    if solid:
        image.draw_circle(location_x_pixel, location_y_pixel, radius_pixel)
    else:
        image.draw_empty_circle(location_x_pixel, location_y_pixel, radius_pixel, 8)


ones_x = 1e8
location_max = np.power((3440 - dx)/sx, 1/ex)
ints = np.arange(0, int(np.ceil(location_max/ones_x)))
x_coords = dx + sx*np.power(ints*ones_x, ex)
image.draw_scale_top(x_coords, 20, 50)

# TODO: scale shouldn't be 2 times something right?
ones_y = 2e5
location_max = np.power(720/sy, 1/ey)
ints = np.arange(0, int(np.ceil(location_max/ones_y)))
y_coords = sy*np.power(ints*ones_y, ey)
image.draw_scale_right(y_coords, 20, 50)


image.show()
image.save()
