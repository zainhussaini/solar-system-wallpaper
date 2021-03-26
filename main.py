#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


class WallpaperImage:
    """ WallpaperImage focuses on creating the image itself, with ease-of-use functions """

    def __init__(self, image_width, image_height, background_color, draw_color):
        """Initializer creates empty image of dimensions filled with background_color.

        Inputs:
            image_width - width of image in pixels
            image_height - height of image in pixels
            background_color - tuple of RGB values for background color (range 0 to 256)
            draw_color - tuple of RGB values for any shapes drawn (range 0 to 256)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.background_color = background_color
        self.draw_color = draw_color
        self.image = Image.new('RGBA', (image_width, image_height), background_color)

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

        # round each to nearest int
        pixel_x = int(pixel_x + 0.5)
        pixel_y = int(pixel_y + 0.5)
        pixel_r = int(pixel_r + 0.5)

        upscale_size = 2*pixel_r*antialias_scale
        mask = Image.new('RGBA', (upscale_size, upscale_size), self.background_color)
        draw = ImageDraw.Draw(mask)
        bound = (0, 0, upscale_size, upscale_size)
        draw.ellipse(bound, fill=self.draw_color, outline=self.background_color)
        mask = mask.resize((2*pixel_r, 2*pixel_r), Image.ANTIALIAS)

        bound = (
            pixel_x - pixel_r,
            pixel_y - pixel_r,
            pixel_x + pixel_r,
            pixel_y + pixel_r)
        self.image.paste(mask, box=bound)

    def draw_empty_circle(self, pixel_x, pixel_y, pixel_r, thickness):
        antialias_scale = 8

        # round each to nearest int
        pixel_x = int(pixel_x + 0.5)
        pixel_y = int(pixel_y + 0.5)
        pixel_r = int(pixel_r + 0.5)

        upscale_size = 2*pixel_r*antialias_scale
        mask = Image.new('RGBA', (upscale_size, upscale_size), self.background_color)
        draw = ImageDraw.Draw(mask)
        bound = (0, 0, upscale_size, upscale_size)
        draw.ellipse(bound, fill=self.draw_color, outline=self.background_color)
        bound = (
            antialias_scale*thickness,
            antialias_scale*thickness,
            upscale_size - antialias_scale*thickness,
            upscale_size - antialias_scale*thickness)
        draw.ellipse(bound, fill=self.background_color, outline=self.background_color)
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
                draw.line((
                    pixel_x, self.image_height/2-height_tens/2,
                    pixel_x, self.image_height/2+height_tens/2),
                    fill="white")
            else:
                draw.line((
                    pixel_x, self.image_height/2-height_ones/2,
                    pixel_x, self.image_height/2+height_ones/2),
                    fill="white")
        draw.line((0, self.image_height/2, self.image_width, self.image_height/2), fill="white")

    def draw_scale_vertical(self, pixel_x, pixels_y, height_ones, height_tens):
        draw = ImageDraw.Draw(self.image)
        for i, pixel_y in enumerate(pixels_y):
            height = height_tens if i%10 == 0 else height_ones

            draw.line((
                pixel_x + height,
                self.image_height/2 + pixel_y,
                pixel_x,
                self.image_height/2 + pixel_y), fill="white")
            draw.line((
                pixel_x + height,
                self.image_height/2 - pixel_y,
                pixel_x,
                self.image_height/2 - pixel_y), fill="white")
        draw.line((pixel_x, 0, pixel_x, self.image_height), fill="white")

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

    def is_ready(self):
        for param in [self.dx, self.sx, self.ex, self.dy, self.sy, self.ey]:
            if param is None:
                return False
        return True

    def calc_x(self, px1, ax1, px2, ax2):
        """Calculates x scaling with two pairs of pixel values and actual values"""
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
        """Calculates y scaling with two pairs of pixel values and actual values"""
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
        if not self.is_ready():
            raise Exception("pixel_to_real called without setting every parameter")

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
        if not self.is_ready():
            raise Exception("pixel_to_real called without setting every parameter")

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


def load_data():
    # parse csv files
    sun_radius = 696340 # not added to any csv file
    df_planets = pd.read_csv('data/planets.csv')
    df_moons = pd.read_csv('data/moons.csv')

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

    return df_planets, df_moons, circles

def generate_wallpaper(width, height):
    # all of these are in pixels
    SUN_X = 200 # position of sun along x axis
    MERCURY_X = 400 # distance of mercury from sun along x axis
    JUPITER_R = 100 # radius of jupiter (largest planet)
    IAPETUS_Y = 600 # distance of iapetus (furthest moon) along y axis
    NEPTUNE_X = 3000 # distanc of neptune (furthest planet) along x axis

    NUM_STARS = 3000

    # all of these are in pixels
    X_TICKS_HEIGHT_ONES = 20
    X_TICKS_HEIGHT_TENS = 50
    Y_TICKS_HEIGHT_ONES = 20
    Y_TICKS_HEIGHT_TENS = 50

    background_color = (1, 28, 44)
    draw_color = (249, 123, 35)

    """ load data """
    df_planets, df_moons, circles = load_data()


    """ initialize mapper """
    mapper = CoordinateMapper(SUN_X, height/2)

    # mapper.calc_x(
    #     SUN_X + MERCURY_X, df_planets["DIST FROM SUN (km)"]["Mercury"],
    #     SUN_X + NEPTUNE_X, df_planets["DIST FROM SUN (km)"]["Neptune"])
    # mapper.calc_y(
    #     height/2 + JUPITER_R, df_planets["RADIUS (km)"]["Jupiter"],
    #     height/2 + IAPETUS_Y, df_moons["DIST FROM PLANET (km)"]["Iapetus"])
    mapper.calc_y(
        height/2 + JUPITER_R, df_planets["RADIUS (km)"]["Jupiter"],
        height/2 + IAPETUS_Y, df_moons["DIST FROM PLANET (km)"]["Iapetus"])
    mapper.calc_x(
        SUN_X + MERCURY_X, df_planets["DIST FROM SUN (km)"]["Mercury"],
        SUN_X + NEPTUNE_X, df_planets["DIST FROM SUN (km)"]["Neptune"])

    """ make image """
    image = WallpaperImage(width, height, background_color, draw_color)

    # draw stars
    np.random.seed(0)
    axmin, aymin, _ = mapper.pixel_to_real(0, 0, 0)
    axmax, aymax, _ = mapper.pixel_to_real(width, height, 0)
    axs = np.random.uniform(axmin, axmax, NUM_STARS)
    ays = np.random.uniform(aymin, aymax, NUM_STARS)
    pxys = []
    for i in range(NUM_STARS):
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
    max_ax = mapper.pixel_to_real(width, mapper.dy, 0)[0]
    axs = np.arange(0, max_ax, ones_x)
    pxs = [mapper.real_to_pixel(ax, 0, 0)[0] for ax in axs]
    image.draw_scale_horizontal(pxs, X_TICKS_HEIGHT_ONES, X_TICKS_HEIGHT_TENS)

    # draw vertical scale
    ones_y = 2e5
    max_ay = mapper.pixel_to_real(mapper.dx, width, 0)[1]
    ays = np.arange(0, max_ay, ones_y)
    pys = [mapper.real_to_pixel(0, ay, 0)[1] - height/2 for ay in ays]
    image.draw_scale_vertical(SUN_X, pys, Y_TICKS_HEIGHT_ONES, Y_TICKS_HEIGHT_TENS)

    return image

if __name__ == "__main__":
    image = generate_wallpaper(3440, 1440)
    image.save()
