from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np
import pickle

IMAGE_SIZE = 39
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 583285
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 64809

def read_queue(filename_queue): 
	reader = tf.WholeFileReader()
	key, value = reader.read(filename_queue)
	
	image = tf.image.decode_jpeg(value, channels=1)

	#image = tf.image.resize_images(image, 392, 392)
	#image = tf.slice(image, [0,0,0], [392, 392, 1])
	image = tf.image.resize_images(image, 78, 78)
	image = tf.slice(image, [0,0,0], [78, 78, 1])

	return image  



def _generate_image_flop_batch(image, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and paths.
  Args:
    image: 3-D Tensor of [height, width, 1] of type.float32.
    path: 1-D Tensor of type string
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 1] size.
    path_batch: Paths. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, path_batch = tf.train.batch(
        [image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)


  #images = tf.reshape(images,[batch_size,392,392,1])
  images = tf.reshape(images,[batch_size,78,78,1])

  return images


def inputs(data_dir, batch_size):
  """Construct flop inputs
  Args:
    data_dir: Path to the  data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
  """
  with open(data_dir, 'r') as f:
    		filenames = [line.rstrip('\n') for line in f]



  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)
  print(filename_queue)

  # Read examples from files in the filename queue.
  image = read_queue(filename_queue)
  print(tf.shape(image))
  reshaped_image = tf.cast(image, tf.float32)

  #height = IMAGE_SIZE
  #width = IMAGE_SIZE
  

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.0001 #0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_flop_batch(reshaped_image,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
										 
										 



