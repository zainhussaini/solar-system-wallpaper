# SolarSystemWallpaper
Code to generate a solar system wallpaper with accurate scaling.

## Images
### Ultrawide 3440x1440

![ultrawide wallpaper](https://github.com/zainhussaini/SolarSystemWallpaper/blob/main/wallpaper.png?raw=true)

### Dual monitor 1920x1080

Image 1| Image 2
:-------------------------:|:-------------------------:
![dual monitor 1](https://github.com/zainhussaini/SolarSystemWallpaper/blob/main/wallpaper0.png?raw=true)| ![dual monitor 2](https://github.com/zainhussaini/SolarSystemWallpaper/blob/main/wallpaper1.png?raw=true)

## Features
Proportional scaling would not generate the most interesting wallpaper (take a look at [if the moon were only 1 pixel](https://joshworth.com/dev/pixelspace/pixelspace_solarsystem.html) to see what it would have to look like). Logarithmic scaling doesn't quite work since distance of 1 unit maps to 0, and negative units don't make sense for distances either. Instead a "root" scale was used instead which raises distance from the origin to an exponent less than 1, with the exponent chosen to fit everything nicely.

Horizontal scaling (along x-axis) shows the distance of planets from the sun. Vertical scaling (along y-axis) is used to show the distance of moons from the planet. This scale is also used for the planet and sun radii. Note that the horizontal scaling is different than the vertical one so the planets and moons could actually be visible. In the dual monitor images the horizontal scaling includes the sun radius, but in the ultrawide this did not look good so Mercury is set 0.5x sun radius away from the sun.

The stars are distributed to reflect the scaling, with their density along the x and y axes showing the density changes in the scale.
