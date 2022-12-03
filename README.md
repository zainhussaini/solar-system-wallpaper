# SolarSystemWallpaper
Code to generate a solar system wallpaper with accurate scaling.

## Images
### Ultrawide 3440x1440

![ultrawide wallpaper](https://drive.google.com/file/d/1cxd_JfTHP5y0otNQ2z1oeDXdzO51BKeJ/preview)

### Dual monitor 1920x1080

Image 1| Image 2
:-------------------------:|:-------------------------:
![dual monitor 1](https://drive.google.com/file/d/13c6XMzbPpuF3-7OryUxL-Gqw8CiOu-e4/preview)| ![dual monitor 2](https://drive.google.com/file/d/1Uhicz3rfwuzSLcJ3vc10g5H8B7tBr4l7/preview)

### Dual monitor 3840x2160

Image 1| Image 2
:-------------------------:|:-------------------------:
![dual monitor 1](https://drive.google.com/file/d/1P1yO5OX8R7pgkuXMqpHmganphu-k2K1D/preview)| ![dual monitor 2](https://drive.google.com/file/d/1x0ffuhgtBPGyvU8qizFH6vt1pO0aWtit/preview)

## Features
Proportional scaling would not generate the most interesting wallpaper (take a look at [if the moon were only 1 pixel](https://joshworth.com/dev/pixelspace/pixelspace_solarsystem.html) to see what it would have to look like). Logarithmic scaling doesn't quite work since distance of 1 unit maps to 0, and distances less than 1 map to negative units which doesn't make sense for distances either. Instead a "root" scale was used instead which raises distance from the origin to an exponent less than 1, with the exponent chosen to fit everything nicely.

Horizontal scaling (along x-axis) shows the distance of planets from the sun. Vertical scaling (along y-axis) is used to show the distance of moons from the planet. This scale is also used for the planet and sun radii (if they were centered on the axis). The horizontal and vertical scalings keep the sun radius consistent as well.

The stars are distributed to reflect the scaling, with their density along the x and y axes showing the density changes in the scale.
