import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

optimizers_dict = {
    "SGD": optimizers.SGD(learning_rate=0.01),
    "SGD+Momentum": optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "SGD+Nesterov": optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
    "Adam": optimizers.Adam(learning_rate=0.001),
    "RMSprop": optimizers.RMSprop(learning_rate=0.001),
    "Adagrad": optimizers.Adagrad(learning_rate=0.01)
}

histories = {}

for name, optimizer in optimizers_dict.items():
    print(f"\nTraining with optimizer: {name}")
    model = create_model()
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(train_images, train_labels,
                        epochs=5, batch_size=64,
                        validation_split=0.1, verbose=1)
    histories[name] = history

plt.figure(figsize=(12, 6))
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=f'{name} Val Acc')
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for name, history in histories.items():
    plt.plot(history.history['val_loss'], label=f'{name} Val Loss')
plt.title("Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

for name, optimizer in optimizers_dict.items():
    model = create_model()
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_images, train_labels,
              epochs=5, batch_size=64, verbose=0)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"{name} Test Accuracy: {round(test_acc * 100, 2)}%")
