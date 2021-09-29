from tkinter import *
from tkinter import colorchooser
import np as np
import matplotlib.pyplot as plt

from palettes_data import *
from DL3 import *

train_X, train_Y, test_X, test_Y = palettes_data()

# adjusting the data:
np_train_X = np.array(train_X)
train_x = np_train_X.transpose() / 64 - 2
np_train_Y = np.array(train_Y)
train_y = np.zeros((1, 7532))
for i in range(len(np_train_Y[0])):
    if np_train_Y[0][i] == 1:
        train_y[0][i] = 1

np_test_X = np.array(test_X)
test_x = np_test_X.transpose() / 64 - 2
np_test_Y = np.array(test_Y)
test_y = np.zeros((1, 100))
for i in range(len(np_test_Y[0])):
    if np_test_Y[0][i] == 1:
        test_y[0][i] = 1

# making the model:
layer1 = DLLayer("layer1", 20, (12,), "leaky_relu", "random")
layer2 = DLLayer("layer2", 5, (20,), "tanh", "random")
layer3 = DLLayer("layer3", 1, (5,), "sigmoid", "random")
model = DLModel()
model.add(layer1)
model.add(layer2)
model.add(layer3)
model.compile("cross_entropy")

# running the model:
costs = model.train(train_x, train_y, 2000)
plt.plot(costs)
plt.show()
predictions = model.predict(train_x)
print('Train accuracy: %d' % float(
    (np.dot(train_y, predictions.T) + np.dot(1 - train_y, 1 - predictions.T)) / float(train_y.size) * 100) + '%')
predictions = model.predict(test_x)
print('Test accuracy: %d' % float(
    (np.dot(test_y, predictions.T) + np.dot(1 - test_y, 1 - predictions.T)) / float(test_y.size) * 100) + '%')


# system:
print("")
print("--------------------pick 3 colors that go together and press X, i'll get you the fourth!--------------------")
print("")


def fourth_color():
    third_color = []
    real_test_X1 = []

    def choose_color():
        color_code = colorchooser.askcolor(title="Choose color")
        third_color.append(int(color_code[0][0]))
        third_color.append(int(color_code[0][1]))
        third_color.append(int(color_code[0][2]))

    def trio_picker():
        root = Tk()
        button = Button(root, text="Select color", command=choose_color)
        button.pack()
        root.geometry("400x400")
        root.mainloop()

    trio_picker()

    for a in range(13):
        b = 20 * a
        for c in range(13):
            d = 20 * c
            for e in range(13):
                f = 20 * e
                completed_palette = [b, d, f] + third_color
                real_test_X1.append(completed_palette)
    return real_test_X1


def show_palette(palette):
    from PIL import Image

    img = Image.new('RGB', (2, 2), color=(0, 0, 0))
    img.save('palette.png')
    palette_img = img.load()

    palette_img[0, 0] = (palette[0], palette[1], palette[2])
    palette_img[1, 0] = (palette[3], palette[4], palette[5])
    palette_img[0, 1] = (palette[6], palette[7], palette[8])
    palette_img[1, 1] = (palette[9], palette[10], palette[11])

    resized_image = img.resize((200, 200))
    resized_image.show()
    print(palette)


real_test_X2 = fourth_color()
real_test_X = np.array(real_test_X2)
real_test_x = real_test_X.transpose() / 64 - 2
predictions = model.predict(real_test_x)

print(predictions[0])
print(real_test_X2)

for i in range(len(predictions[0])):
    if predictions[0][i]:
        print("cool:", real_test_X2[i])
        show_palette(real_test_X2[i])

        gudge = input("want more?")
        if gudge == "no":
            break
