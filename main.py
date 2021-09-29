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
costs = model.train(train_x, train_y, 20000)
plt.plot(costs)
plt.show()
predictions = model.predict(train_x)
print('Train accuracy: %d' % float((np.dot(train_y, predictions.T) + np.dot(1 - train_y, 1 - predictions.T)) / float(train_y.size) * 100) + '%')
predictions = model.predict(test_x)
print('Test accuracy: %d' % float((np.dot(test_y, predictions.T) + np.dot(1 - test_y, 1 - predictions.T)) / float(test_y.size) * 100) + '%')
