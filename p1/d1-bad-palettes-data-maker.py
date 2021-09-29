from PIL import Image
import random

palette_img1 = Image.new('RGB', (2, 2), color=(0, 0, 0))
palette_img1.save('palette.png')
palette_img = palette_img1.load()

bad_palettes = []
good_palettes = []
In_work = True
while In_work:

    R0_random = random.randint(1, 255)
    G0_random = random.randint(1, 255)
    B0_random = random.randint(1, 255)
    palette_img[0, 0] = (R0_random, G0_random, B0_random)
    R1_random = random.randint(1, 255)
    G1_random = random.randint(1, 255)
    B1_random = random.randint(1, 255)
    palette_img[1, 0] = (R1_random, G1_random, B1_random)
    R2_random = random.randint(1, 255)
    G2_random = random.randint(1, 255)
    B2_random = random.randint(1, 255)
    palette_img[0, 1] = (R2_random, G2_random, B2_random)
    R3_random = random.randint(1, 255)
    G3_random = random.randint(1, 255)
    B3_random = random.randint(1, 255)
    palette_img[1, 1] = (R3_random, G3_random, B3_random)

    resized_image = palette_img1.resize((200, 200))
    resized_image.show()

    gudge = input("1 = good palette, 2 = bad palette , stop = stop:")
    if gudge == "1":
        good_palette = [R0_random, G0_random, B0_random, R1_random, G1_random, B1_random, R2_random, G2_random, B2_random, R3_random, G3_random, B3_random]
        print(good_palette)
        good_palettes.append(good_palette)
    if gudge == "2":
        bad_palette = [R0_random, G0_random, B0_random, R1_random, G1_random, B1_random, R2_random, G2_random, B2_random, R3_random, G3_random, B3_random]
        print(bad_palette)
        bad_palettes.append(bad_palette)
    if gudge == "stop":
        print("")
        print("bad_palettes:")
        print(bad_palettes)
        print("good_palettes:")
        print(good_palettes)
        In_work = False
