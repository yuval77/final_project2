import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

good_palettes = []


for i in range(1, 101):
    img = Image.open(f'p2/good-palettes-images/{i}.png')
    rgb_im = img.convert('RGB')

    r0, g0, b0 = rgb_im.getpixel((0, 0))
    r1, g1, b1 = rgb_im.getpixel((1, 0))
    r2, g2, b2 = rgb_im.getpixel((2, 0))
    r3, g3, b3 = rgb_im.getpixel((3, 0))

    palette = [r0, g0, b0, r1, g1, b1, r2, g2, b2, r3, g3, b3]
    good_palettes.append(palette)

print("")
print("good_palettes:")
print(good_palettes)





























