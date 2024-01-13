import tensorflow as tf

constant_tensor = tf.constant([1, 2, 3])
print(constant_tensor.numpy())

a = tf.constant([2, 4, 6])
b = tf.constant([1, 2, 3])

result = tf.multiply(a, b)
print(result.numpy())

x = tf.constant([1, 2, 3, 4, 5, 6])
reshaped_tensor = tf.reshape(x, (2, 3))
print(reshaped_tensor.numpy())

matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
middle_column = matrix[:, 1]
print(middle_column.numpy())

weights = tf.Variable(tf.random.normal((3, 3)))
print(weights.numpy())

input_tensor = tf.constant([4.0, 9.0, 16.0])
sqrt_tensor = tf.sqrt(input_tensor)
print(sqrt_tensor.numpy())