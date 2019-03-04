import tensorflow as tf
import numpy as np

tf.__version__

# Convolution layer 1
filter_size_1 = 5
num_filters_1 = 16

# Convolution layer 2
filter_size_2 = 5
num_filters_2 = 36

# Fully connected layer
fc_size = 128

# Learning rate
learning_rate = 1e-4

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('/tmp/MNIST', one_hot = True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis = 1)

# Image size
img_size = 28

# Flattened image
img_size_flat = img_size * img_size

# Image shape (width, height)
img_shape = (img_size, img_size)

# No. of color channels
num_channels = 1

# No. of output classes
num_classes = 10

# Assign new weights
def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))

# Assign new biases
def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape = [length]))

# Define the convolutional layer
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling = True):
	# Format defined by Tensorflow API
	shape = [filter_size, filter_size, num_input_channels, num_filters]

	weights = new_weights(shape = shape)
	biases = new_biases(length = num_filters)

	# Create tensorflow operation for convolution
	layer = tf.nn.conv2d(input = input, filter = weights, strides = [1, 1, 1, 1], padding = 'SAME')

	# Add bias value
	layer += biases

	# Downsample if max_pooling is enabled
	if use_pooling:
		layer = tf.nn.max_pool(value = layer, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# Use activation
	layer = tf.nn.relu(layer)

	return layer, weights

# Flatten layer to get the total no. of features
def flatten_layer(layer):
	# Get shape of the input layer
	layer_shape = layer.get_shape()

	# Get no. of features from the shape
	num_features = layer_shape[1:4].num_elements()

	# Reshape to [num_images, num_features]
	layer_flat = tf.reshape(layer, [-1, num_features])

	return layer_flat, num_features

# Create a fully-connected (dense) layer
def new_fc_layer(inputs, num_inputs, num_outputs, use_relu = True):
	# Create new weights and biases
	weights = new_weights(shape = [num_inputs, num_outputs])
	biases = new_biases(length = num_outputs)

	# Create output fc layer
	layer = tf.add(tf.matmul(inputs, weights), biases)

	# If ReLU is enabled
	if use_relu:
		layer = tf.nn.relu(layer)

	return layer

# Create placeholders for X and y
x = tf.placeholder(tf.float32, shape = [None, img_size_flat], name = 'x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape = [None, num_classes], name = 'y')
y_true_cls = tf.argmax(y_true, axis = 1)

# Create network convolutional layers
conv_layer_1, conv_weights_1 = new_conv_layer(input = x_image,
											  num_input_channels = num_channels,
											  filter_size = filter_size_1,
											  num_filters = num_filters_1,
											  use_pooling = True)

conv_layer_2, conv_weights_2 = new_conv_layer(input = conv_layer_1,
											  num_input_channels = num_filters_1,
											  filter_size = filter_size_2,
											  num_filters = num_filters_2,
											  use_pooling = True)

# Flatten the output conv layer
layer_flat, num_features = flatten_layer(conv_layer_2)

# Create fully-connected layers
layer_fc_1 = new_fc_layer(inputs = layer_flat, num_inputs = num_features, num_outputs = fc_size, use_relu = True)

layer_fc_2 = new_fc_layer(inputs = layer_fc_1, num_inputs = fc_size, num_outputs = num_classes, use_relu = True)

y_pred = tf.nn.softmax(layer_fc_2)
y_pred_cls = tf.argmax(y_pred, axis = 1)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = layer_fc_2, labels = y_true))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)
train_batch_size = 64
total_iterations = 0

def optimize(num_iterations):
	global total_iterations

	for i in range(total_iterations, total_iterations + num_iterations):
		# Get the batch of training examples
		x_batch, y_true_batch = data.train.next_batch(train_batch_size)

		feed_dict = {x: x_batch, y_true: y_true_batch}

		# Run the optimizer
		session.run(train_op, feed_dict = feed_dict)

		# Print status every 100 iterations
		if i % 100 == 0:
			acc = session.run(accuracy, feed_dict = feed_dict)
			print("Iteration: " + str(i+1) + ", Accuracy: " + str(acc))

	total_iterations = total_iterations + num_iterations

test_batch_size = 256

def print_test_accuracy():
	num_test = len(data.test.images)

	cls_pred = np.zeros(shape = num_test, dtype = np.int)

	# Starting index for next batch
	i = 0

	while i < num_test:
		# Ending index for the batch
		j = min(i + test_batch_size, num_test)

		images = data.test.images[i:j, :]
		labels = data.test.labels[i:j, :]

		feed_dict = {x: images, y_true: labels}

		cls_pred[i:j] = session.run(y_pred_cls, feed_dict = feed_dict)

		# Set the start index of the next batch to the end index of current batch
		i = j

	cls_true = data.test.cls

	# No. of correct labels
	correct = (cls_true == cls_pred).sum()

	acc = float(correct) / num_test

	print("Accuracy on test set: " + str(acc))


print_test_accuracy()
optimize(num_iterations=1)
print_test_accuracy()
optimize(num_iterations=99) # Performed 1 iteration above.
print_test_accuracy()
optimize(num_iterations=900) # Performed 100 iterations above.
print_test_accuracy()
session.close()
