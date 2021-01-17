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

    def add_stars(self, pxys):
        # TODO: match actual stars and constellations
        for px, py in pxys:
            self.image.putpixel((int(px), int(py)), (255, 255, 255))

            # make 10% stars extra big
            if np.random.uniform(0, 1) < 0.1:
                for i in range(2):
                    for j in range(2):
                        if int(px+i) < self.image_width and int(py+j) < self.image_height:
                            self.image.putpixel((int(px+i), int(py+j)), (255, 255, 255))

    def draw_circle(self, pixel_x, pixel_y, pixel_r):
        antialias_scale = 8

        pixel_x = int(pixel_x + 0.5)
        pixel_y = int(pixel_y + 0.5)
        pixel_r = int(pixel_r + 0.5)

        upscale_size = 2*pixel_r*antialias_scale
        mask = Image.new('RGBA', (upscale_size, upscale_size), WallpaperImage.background)
        draw = ImageDraw.Draw(mask)
        bound = (0, 0, upscale_size, upscale_size)
        draw.ellipse(bound, fill=WallpaperImage.foreground, outline=WallpaperImage.background)
        mask = mask.resize((2*pixel_r, 2*pixel_r), Image.ANTIALIAS)

        bound = (
            pixel_x - pixel_r,
            pixel_y - pixel_r,
            pixel_x + pixel_r,
            pixel_y + pixel_r)
        self.image.paste(mask, box=bound)

    def draw_empty_circle(self, pixel_x, pixel_y, pixel_r, thickness):
        antialias_scale = 8

        pixel_x = int(pixel_x + 0.5)
        pixel_y = int(pixel_y + 0.5)
        pixel_r = int(pixel_r + 0.5)

        upscale_size = 2*pixel_r*antialias_scale
        mask = Image.new('RGBA', (upscale_size, upscale_size), WallpaperImage.background)
        draw = ImageDraw.Draw(mask)
        bound = (0, 0, upscale_size, upscale_size)
        draw.ellipse(bound, fill=WallpaperImage.foreground, outline=WallpaperImage.background)
        bound = (
            antialias_scale*thickness,
            antialias_scale*thickness,
            upscale_size - antialias_scale*thickness,
            upscale_size - antialias_scale*thickness)
        draw.ellipse(bound, fill=WallpaperImage.background, outline=WallpaperImage.background)
        mask = mask.resize((2*pixel_r, 2*pixel_r), Image.ANTIALIAS)

        bound = (
            pixel_x - pixel_r,
            pixel_y - pixel_r,
            pixel_x + pixel_r,
            pixel_y + pixel_r)
        self.image.paste(mask, box=bound)

    def draw_scale_horizontal(self, pixels_x, height_ones, height_tens):
        draw = ImageDraw.Draw(self.image)
        for i, pixel_x in enumerate(pixels_x):
            if i%10 == 0:
                draw.line((pixel_x, 720-height_tens/2, pixel_x, 720+height_tens/2), fill="white")
            else:
                draw.line((pixel_x, 720-height_ones/2, pixel_x, 720+height_ones/2), fill="white")
        draw.line((200, 720, 3440, 720), fill="white")

    def draw_scale_vertical(self, pixels_y, height_ones, height_tens):
        draw = ImageDraw.Draw(self.image)
        for i, pixel_y in enumerate(pixels_y):
            if i%10 == 0:
                height = height_tens
            else:
                height = height_ones

            draw.line((
                200 + height,
                self.image_height/2 + pixel_y,
                200,
                self.image_height/2 + pixel_y), fill="white")
            draw.line((
                200 + height,
                self.image_height/2 - pixel_y,
                200,
                self.image_height/2 - pixel_y), fill="white")
        draw.line((200, 0, 200, 1440), fill="white")

    def show(self):
        self.image.show()

    def save(self):
        self.image.save('wallpaper.png')


class CoordinateMapper:
    def __init__(self, px0, py0):
        """px0 is pixel corresponding to 0 x location, similar for py0"""
        self.dx = px0
        self.sx = None
        self.ex = None
        self.dy = py0
        self.sy = None
        self.ey = None

    def calc_x(self, px1, ax1, px2, ax2):
        A = np.array([
            [1/np.log(ax1), 1],
            [1/np.log(ax2), 1]])
        b = np.array([
            [np.log(px1-self.dx)/np.log(ax1)],
            [np.log(px2-self.dx)/np.log(ax2)]])
        res = np.linalg.inv(A) @ b
        self.sx = np.exp(res[0][0])
        self.ex = res[1][0]

    def calc_y(self, py1, ay1, py2, ay2):
        A = np.array([
            [1/np.log(ay1), 1],
            [1/np.log(ay2), 1]])
        b = np.array([
            [np.log(py1-self.dy)/np.log(ay1)],
            [np.log(py2-self.dy)/np.log(ay2)]])
        res = np.linalg.inv(A) @ b
        self.sy = np.exp(res[0][0])
        self.ey = res[1][0]

    def pixel_to_real(self, px, py, pr):
        if px >= self.dx:
            ax = np.power((px - self.dx)/self.sx, 1/self.ex)
        else:
            ax = -np.power((self.dx - px)/self.sx, 1/self.ex)
        if py >= self.dy:
            ay = np.power((py - self.dy)/self.sy, 1/self.ey)
        else:
            ay = -np.power((self.dy - py)/self.sy, 1/self.ey)
        ar = np.power(py/self.sy, 1/self.ey)
        return (ax, ay, ar)

    def real_to_pixel(self, ax, ay, ar):
        if ax >= 0:
            px = self.dx + self.sx*np.power(ax, self.ex)
        else:
            px = self.dx - self.sx*np.power(-ax, self.ex)
        if ay >= 0:
            py = self.dy + self.sy*np.power(ay, self.ey)
        else:
            py = self.dy - self.sy*np.power(-ay, self.ey)
        pr = self.sy*np.power(ar, self.ey)
        return (px, py, pr)


""" parse csv files """
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


""" initialize mapper """
mapper = CoordinateMapper(200, 720)
# mapper.calc_r(150, df_planets["RADIUS (km)"]["Jupiter"])
mapper.calc_x(
    600, df_planets["DIST FROM SUN (km)"]["Mercury"],
    3200, df_planets["DIST FROM SUN (km)"]["Neptune"])
mapper.calc_y(
    720+100, df_planets["RADIUS (km)"]["Jupiter"],
    720+600, df_moons["DIST FROM PLANET (km)"]["Iapetus"])


""" make image """
image = WallpaperImage()

# draw stars
num = 3000
axmin, aymin, _ = mapper.pixel_to_real(0, 0, 0)
axmax, aymax, _ = mapper.pixel_to_real(3440, 1440, 0)
axs = np.random.uniform(axmin, axmax, num)
ays = np.random.uniform(aymin, aymax, num)
pxys = []
for i in range(num):
    px, py, _ = mapper.real_to_pixel(axs[i], ays[i], 0)
    pxys.append((px, py))
image.add_stars(pxys)

# draw planets + moons
for ax, ay, ar, solid in circles:
    px, py, pr = mapper.real_to_pixel(ax, ay, ar)
    if solid:
        image.draw_circle(px, py, pr)
    else:
        image.draw_empty_circle(px, py, pr, 8)

# draw horizontal scale
ones_x = 5e7
max_ax = mapper.pixel_to_real(3440, mapper.dy, 0)[0]
axs = np.arange(0, max_ax, ones_x)
pxs = [mapper.real_to_pixel(ax, 0, 0)[0] for ax in axs]
image.draw_scale_horizontal(pxs, 20, 50)

# draw vertical scale
ones_y = 1e5
max_ay = mapper.pixel_to_real(mapper.dx, 1440, 0)[1]
ays = np.arange(0, max_ay, ones_y)
pys = [mapper.real_to_pixel(0, ay, 0)[1] - 720 for ay in ays]
image.draw_scale_vertical(pys, 20, 50)

image.save()
image.show()
