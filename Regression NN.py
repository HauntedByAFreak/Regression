import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
num_samples = 1200
num_features = 8
X_train = np.random.randn(num_samples, num_features)
y_train = 2 * X_train.sum(axis=1, keepdims=True) + np.random.randn(num_samples, 1)

train_size = int(0.6 * num_samples)
val_size = int(0.2 * num_samples)
test_size = num_samples - train_size - val_size

X_train_split = X_train[:train_size]
y_train_split = y_train[:train_size]

X_val_split = X_train[train_size:train_size + val_size]
y_val_split = y_train[train_size:train_size + val_size]

X_test_split = X_train[train_size + val_size:]
y_test_split = y_train[train_size + val_size:]



model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_features,)),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, batch_size=32, epochs=40, validation_split=0.2)

model.summary()


def measure_bias_variance(model, X, y, num_splits=5):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_errors = []
    test_errors = []

    for _ in range(num_splits):
        
        cloned_model = tf.keras.models.clone_model(model)
        learning_rate = 0.009
        cloned_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')


        history = cloned_model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2, verbose=0)

        train_error = history.history['loss'][-1]
        test_error = cloned_model.evaluate(X_test, y_test, verbose=0)

        train_errors.append(train_error)
        test_errors.append(test_error)

    
    bias = np.mean(train_errors) - np.mean(test_errors)
    variance = np.var(test_errors)
    total_error = bias + variance

    return bias, variance, total_error


bias, variance, total_error = measure_bias_variance(model, X_train, y_train)


print("Bias: ", bias)
print("Variance: ", variance)
print("Total Error: ", total_error)

# Plot learning curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
