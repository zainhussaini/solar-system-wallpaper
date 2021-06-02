# SolarSystemWallpaper
Code to generate a solar system wallpaper with accurate scaling.

## Images
### Ultrawide 3440x1440

![ultrawide wallpaper](https://raw.githubusercontent.com/zainhussaini/SolarSystemWallpaper/blob/main/images/3440x1440.png?raw=true)

### Dual monitor 1920x1080

Image 1| Image 2
:-------------------------:|:-------------------------:
![dual monitor 1](https://raw.githubusercontent.com/zainhussaini/SolarSystemWallpaper/blob/main/images/1920x1080_part1.png?raw=true)| ![dual monitor 2](https://raw.githubusercontent.com/zainhussaini/SolarSystemWallpaper/blob/main/images/1920x1080_part2.png?raw=true)

### Dual monitor 3840x2160

Image 1| Image 2
:-------------------------:|:-------------------------:
![dual monitor 1](https://raw.githubusercontent.com/zainhussaini/SolarSystemWallpaper/blob/main/images/3840x2160_part1.png?raw=true)| ![dual monitor 2](https://raw.githubusercontent.com/zainhussaini/SolarSystemWallpaper/blob/main/images/3840x2160_part2.png?raw=true)

## Features
Proportional scaling would not generate the most interesting wallpaper (take a look at [if the moon were only 1 pixel](https://joshworth.com/dev/pixelspace/pixelspace_solarsystem.html) to see what it would have to look like). Logarithmic scaling doesn't quite work since distance of 1 unit maps to 0, and distances less than 1 map to negative units which doesn't make sense for distances either. Instead a "root" scale was used instead which raises distance from the origin to an exponent less than 1, with the exponent chosen to fit everything nicely.

Horizontal scaling (along x-axis) shows the distance of planets from the sun. Vertical scaling (along y-axis) is used to show the distance of moons from the planet. This scale is also used for the planet and sun radii (if they were centered on the axis). The horizontal and vertical scalings keep the sun radius consistent as well.

The stars are distributed to reflect the scaling, with their density along the x and y axes showing the density changes in the scale.
