# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""A demo for on-device backprop (transfer learning) of a classification model.

This demo runs a similar task as described in TF Poets tutorial, except that
learning happens on-device.
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0

Here are the steps:
1) mkdir -p /tmp/retrain/

2) curl http://download.tensorflow.org/example_images/flower_photos.tgz \
     | tar xz -C /tmp/retrain

3) bash examples/install_requirements.sh backprop_last_layer.py

4) Start training:

    python3 examples/backprop_last_layer.py \
      --data_dir /tmp/retrain/flower_photos \
      --embedding_extractor_path \
      test_data/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite

   Weights for retrained last layer will be saved to /tmp/retrain/output by
   default.

5) Run an inference with the new model:

    python3 examples/classify_image.py \
      --model /tmp/retrain/output/retrained_model_edgetpu.tflite \
      --label /tmp/retrain/output/label_map.txt
      --input test_data/sunflower.bmp

For more information, see
https://coral.ai/docs/edgetpu/retrain-classification-ondevice-backprop/
"""

import argparse
import contextlib
import os
import sys
import time

import tensorflow as tf

import numpy as np
from PIL import Image


@contextlib.contextmanager
def test_image(path):
  """Returns opened test image."""
  with open(path, 'rb') as f:
    with Image.open(f) as image:
      yield image


def save_label_map(label_map, out_path):
  """Saves label map to a file."""
  with open(out_path, 'w') as f:
    for key, val in label_map.items():
      f.write('%s %s\n' % (key, val))


def get_image_paths(data_dir):
  """Walks through data_dir and returns list of image paths and label map.

  Args:
    data_dir: string, path to data directory. It assumes data directory is
      organized as, - [CLASS_NAME_0] -- image_class_0_a.jpg --
      image_class_0_b.jpg -- ... - [CLASS_NAME_1] -- image_class_1_a.jpg -- ...

  Returns:
    A tuple of (image_paths, labels, label_map)
    image_paths: list of string, represents image paths
    labels: list of int, represents labels
    label_map: a dictionary (int -> string), e.g., 0->class0, 1->class1, etc.
  """
  classes = None
  image_paths = []
  labels = []

  class_idx = 0
  for root, dirs, files in os.walk(data_dir):
    if root == data_dir:
      # Each sub-directory in `data_dir`
      classes = dirs
    else:
      # Read each sub-directory
      assert classes[class_idx] in root
      print('Reading dir: %s, which has %d images' % (root, len(files)))
      for img_name in files:
        image_paths.append(os.path.join(root, img_name))
        labels.append(class_idx)
      class_idx += 1
  # print("error")
  return image_paths, labels, dict(zip(range(class_idx), classes))


def shuffle_and_split(image_paths, labels, val_percent=0.1, test_percent=0.1):
  """Shuffles and splits data into train, validation, and test sets.

  Args:
    image_paths: list of string, of dim num_data
    labels: list of int of length num_data
    val_percent: validation data set percentage.
    test_percent: test data set percentage.

  Returns:
    Two dictionaries (train_and_val_dataset, test_dataset).
    train_and_val_dataset has the following fields.
      'data_train': data_train
      'labels_train': labels_train
      'data_val': data_val
      'labels_val': labels_val
    test_dataset has the following fields.
      'data_test': data_test
      'labels_test': labels_test
  """
  image_paths = np.array(image_paths)
  labels = np.array(labels)
  perm = np.random.permutation(image_paths.shape[0])
  image_paths = image_paths[perm]
  labels = labels[perm]

  num_total = image_paths.shape[0]
  num_val = int(num_total * val_percent)
  num_test = int(num_total * test_percent)
  num_train = num_total - num_val - num_test

  # Printing for details of dataset size
  print("-" * 50)
  print(f"Dataset split summary:")
  print(f"Total images: {num_total}")
  print(f"Training set: {num_train} images ({num_train/num_total*100:.1f}%)")
  print(f"Validation set: {num_val} images ({num_val/num_total*100:.1f}%)")
  print(f"Test set: {num_test} images ({num_test/num_total*100:.1f}%)")
  print("-" * 50)

  train_and_val_dataset = {}
  train_and_val_dataset['data_train'] = image_paths[0:num_train]
  train_and_val_dataset['labels_train'] = labels[0:num_train]
  train_and_val_dataset['data_val'] = image_paths[num_train:num_train + num_val]
  train_and_val_dataset['labels_val'] = labels[num_train:num_train + num_val]
  test_dataset = {}
  test_dataset['data_test'] = image_paths[num_train + num_val:]
  test_dataset['labels_test'] = labels[num_train + num_val:]
  return train_and_val_dataset, test_dataset


def extract_embeddings(image_paths, interpreter, batch_size):
  """Uses model to process images as embeddings.

  Reads image, resizes and feeds to model to get feature embeddings. Original
  image is discarded to keep maximum memory consumption low.

  Args:
    image_paths: ndarray, represents a list of image paths.
    interpreter: TFLite interpreter, wraps embedding extractor model.

  Returns:
    ndarray of length image_paths.shape[0] of embeddings.
  """
  batch_images = []
  embeddings = []
  input_details = interpreter.get_input_details()
  input_batch_shape = input_details[0]['shape'] # [1, 224, 224, 3]
  input_batch_shape[0] = batch_size  # [batch_size, 224, 224, 3]

  print(input_batch_shape)
  interpreter.resize_tensor_input(0, input_batch_shape)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  
  input_index = input_details[0]['index']
  output_index = output_details[0]['index']
  
  interpreter.resize_tensor_input(input_index, input_batch_shape)
  interpreter.allocate_tensors()

  for idx, path in enumerate(image_paths):
    with test_image(path) as img:
        batch_images.append(img.resize((224,224), Image.NEAREST))
      

    # If batch is full or it's the last image
    if len(batch_images) == batch_size or idx == len(image_paths) - 1:

        input_batch = np.stack(batch_images, axis=0)  # Shape: [batch_size, H, W, C]

        if idx == len(image_paths) - 1:
            interpreter.resize_tensor_input(0, input_batch.shape)
            interpreter.allocate_tensors()

        # Resize and run inference
        interpreter.set_tensor(input_index, input_batch)
        interpreter.invoke()

        output = interpreter.get_tensor(output_index)
        embeddings.append(output)

        # Clear batch
        batch_images = []
    

  embeddings = np.concatenate(embeddings, axis=0)

  return embeddings


def train(model_path, data_dir, output_dir, batch_size=8):
  """Trains a softmax regression model given data and embedding extractor.

  Args:
    model_path: string, path to embedding extractor.
    data_dir: string, directory that contains training data.
    output_dir: string, directory to save retrained tflite model and label map.
  """
  image_paths, labels, label_map = get_image_paths(data_dir)
  train_and_val_dataset, test_dataset = shuffle_and_split(image_paths, labels)

  interpreter = tf.lite.Interpreter(model_path, num_threads=4) # The place to change threads

  # print('error here 3')
  print('Extract embeddings for data_train')
  t0 = time.perf_counter()
  train_and_val_dataset['data_train'] = extract_embeddings(
      train_and_val_dataset['data_train'], interpreter, batch_size)
  t1 = time.perf_counter()
  print('Feature extractor for training dataset takes %.2f seconds' % (t1 - t0))

  print('Extract embeddings for data_val')
  t2 = time.perf_counter()
  train_and_val_dataset['data_val'] = extract_embeddings(
      train_and_val_dataset['data_val'], interpreter, batch_size)
  t3 = time.perf_counter()
  print('Feature extractor for validation dataset takes %.2f seconds' % (t3 - t2))

  t3 = time.perf_counter()
  print('Extract embeddings for data_test')
  test_embeddings = extract_embeddings(test_dataset['data_test'], interpreter, batch_size)
  
  t4 = time.perf_counter()
  print('Feature extractor for test dataset takes %.2f seconds' % (t4 - t3))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--embedding_extractor_path',
      required=True,
      help='Path to embedding extractor tflite model.')
  parser.add_argument('--data_dir', required=True, help='Directory to data.')
  parser.add_argument(
      '--output_dir',
      default='/tmp/retrain/output',
      help='Path to directory to save retrained model and label map.')
  parser.add_argument(
      '--batch_size',
      default=8,
      help='Batch size.')
  args = parser.parse_args()

  if not os.path.exists(args.data_dir):
    sys.exit('%s does not exist!' % args.data_dir)

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

#   train('./mobilenet_cut_avgpool.tflite', './flower_photos', './out')
  train(args.embedding_extractor_path, args.data_dir, args.output_dir, args.batch_size)


if __name__ == '__main__':
  main()