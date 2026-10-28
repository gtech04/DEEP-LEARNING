import tensorflow as tf
from tensorflow.keras import layers, models, initializers, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

def create_model(initializer, dropout_rate=0.3, kernel_regularizer=regularizers.l2(0.01)):
    model = models.Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        
        layers.Dense(512, activation='relu',
                     kernel_initializer=initializer,
                     kernel_regularizer=kernel_regularizer),
        layers.Dropout(dropout_rate),
        
        layers.Dense(256, activation='relu',
                     kernel_initializer=initializer,
                     kernel_regularizer=kernel_regularizer),
        layers.Dropout(dropout_rate),
        
        layers.Dense(128, activation='relu',
                     kernel_initializer=initializer,
                     kernel_regularizer=kernel_regularizer),
        layers.Dropout(dropout_rate),
        
        layers.Dense(64, activation='relu',
                     kernel_initializer=initializer,
                     kernel_regularizer=kernel_regularizer),
        layers.Dropout(dropout_rate),
        
        layers.Dense(32, activation='relu',
                     kernel_initializer=initializer,
                     kernel_regularizer=kernel_regularizer),
        layers.Dropout(dropout_rate),
        
        layers.Dense(10, activation='softmax')
    ])
    return model

xavier_initializer = initializers.glorot_normal()
kaiming_initializer = initializers.he_normal()

xavier_model = create_model(xavier_initializer, dropout_rate=0.3)
kaiming_model = create_model(kaiming_initializer, dropout_rate=0.3)

xavier_model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

kaiming_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

xavier_history = xavier_model.fit(x_train, y_train_cat,
                                  epochs=10, batch_size=64,
                                  validation_split=0.2, verbose=1)

kaiming_history = kaiming_model.fit(x_train, y_train_cat,
                                    epochs=10, batch_size=64,
                                    validation_split=0.2, verbose=1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(xavier_history.history['accuracy'], label='Xavier Train')
plt.plot(xavier_history.history['val_accuracy'], label='Xavier Val')
plt.title('Xavier Initialization')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(kaiming_history.history['accuracy'], label='Kaiming Train')
plt.plot(kaiming_history.history['val_accuracy'], label='Kaiming Val')
plt.title('Kaiming Initialization')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

test_loss_x, test_acc_x = xavier_model.evaluate(x_test, y_test_cat)
test_loss_k, test_acc_k = kaiming_model.evaluate(x_test, y_test_cat)

print(f"\nXavier Test Accuracy: {round(test_acc_x * 100, 2)}%")
print(f"Kaiming Test Accuracy: {round(test_acc_k * 100, 2)}%")
