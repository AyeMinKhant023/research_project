import os
import sys
import time
import csv
import json
import random
import contextlib
from typing import Dict, List, Tuple, Optional

import flwr as fl
import numpy as np
from PIL import Image
from softmax_regression import SoftmaxRegression, create_classification_head


@contextlib.contextmanager
def test_image(path):
    """Returns opened test image."""
    with open(path, 'rb') as f:
        with Image.open(f) as image:
            yield image


def get_image_paths(data_dir):
    """Walks through data_dir in deterministic sorted order and returns image paths and label map."""
    classes = None
    image_paths = []
    labels = []

    class_idx = 0
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        if root == data_dir:
            classes = list(dirs)
        else:
            assert classes[class_idx] in root
            valid_files = sorted(f for f in files if not f.startswith('.'))
            print('Reading dir: %s, which has %d images' % (root, len(valid_files)))
            for img_name in valid_files:
                image_paths.append(os.path.join(root, img_name))
                labels.append(class_idx)
            class_idx += 1

    return image_paths, labels, dict(zip(range(class_idx), classes))


def shuffle_and_split(image_paths, labels, val_percent=0.0, test_percent=0.2, seed: Optional[int] = 42):
    """Shuffles and splits data into train, validation, and test sets (80-20 split by default)."""
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    if seed is not None and seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(image_paths.shape[0])
    else:
        perm = np.random.permutation(image_paths.shape[0])
    image_paths = image_paths[perm]
    labels = labels[perm]

    num_total = image_paths.shape[0]
    num_val = int(num_total * val_percent)
    num_test = int(num_total * test_percent)
    num_train = num_total - num_val - num_test

    train_and_val_dataset = {
        'data_train': image_paths[0:num_train],
        'labels_train': labels[0:num_train],
        'data_val': image_paths[num_train:num_train + num_val],
        'labels_val': labels[num_train:num_train + num_val],
    }

    test_dataset = {
        'data_test': image_paths[num_train + num_val:],
        'labels_test': labels[num_train + num_val:],
    }

    return train_and_val_dataset, test_dataset


def extract_embeddings_tflite(image_paths, interpreter, input_size=None):
    """Extract float32 embeddings using TFLite interpreter, auto-adapting to MobileNetV1 & InceptionV3."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    auto_input_size = (int(input_shape[2]), int(input_shape[1]))  # width, height
    target_size = auto_input_size if auto_input_size[0] > 0 else (input_size or (224, 224))
    input_dtype = input_details[0]['dtype']

    out_shape = output_details[0]['shape']
    feature_dim = int(np.prod(out_shape))
    out_dtype = output_details[0]['dtype']
    out_q = output_details[0].get('quantization', (0.0, 0))
    out_scale = float(out_q[0]) if out_q else 0.0
    out_zp = int(out_q[1]) if out_q else 0
    need_dequant = np.issubdtype(out_dtype, np.integer) and out_scale > 0.0

    embeddings = np.empty((len(image_paths), feature_dim), dtype=np.float32)
    for idx, path in enumerate(image_paths):
        with test_image(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_resized = img.resize(target_size, Image.NEAREST)
            if input_dtype == np.uint8:
                img_array = np.array(img_resized, dtype=np.uint8)
            else:
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index']).reshape(-1)
            if need_dequant:
                embeddings[idx, :] = out_scale * (output.astype(np.int64) - out_zp)
            else:
                embeddings[idx, :] = output.astype(np.float32)
    return embeddings


class FederatedClient(fl.client.NumPyClient):
    def __init__(
        self,
        embedding_extractor_path: str,
        data_dir: str,
        num_classes: int,
        input_size: Tuple[int, int] = (224, 224),
        num_head_layers: int = 0,
        hidden_dim: int = 512,
        num_threads: int = 4,
        local_epochs: int = 10,
        track_accuracy: bool = False,
        accuracy_csv: Optional[str] = None,
        seed: int = 42,
        val_percent: float = 0.0,
        test_percent: float = 0.2,
        embedding_cache: Optional[str] = None,
    ):
        self.embedding_extractor_path = os.path.abspath(os.path.expanduser(embedding_extractor_path))
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_head_layers = num_head_layers
        self.hidden_dim = hidden_dim
        self.num_threads = num_threads
        self.local_epochs = local_epochs
        self.track_accuracy = track_accuracy
        self.accuracy_csv = accuracy_csv
        self.seed = seed if (seed is not None and seed >= 0) else None
        self.val_percent = val_percent
        self.test_percent = test_percent
        self.embedding_cache = os.path.expanduser(embedding_cache) if embedding_cache else None

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.test_embeddings = None
        self.test_labels = None
        self.round_history = []
        self.round_train_times = []
        self._last_fit_metrics = {}

        if self.accuracy_csv:
            os.makedirs(os.path.dirname(os.path.abspath(self.accuracy_csv)), exist_ok=True)
            with open(self.accuracy_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "train_loss", "train_acc", "test_loss", "test_acc", "round_train_sec", "cum_train_sec"])

        self.interpreter = None
        if not (self.embedding_cache and os.path.exists(self.embedding_cache)):
            self.load_model()
        self.load_data()

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.feature_dim = self.train_embeddings.shape[1]
        self.head = create_classification_head(
            self.feature_dim,
            self.num_classes,
            num_head_layers=self.num_head_layers,
            hidden_dim=self.hidden_dim,
            weight_scale=None,
            reg=0.0,
            seed=self.seed,
        )
        p_counts = self.head.count_parameters()

        print("Mac Client initialized with:")
        print(f"- Feature dimension: {self.feature_dim}")
        print(f"- Number of classes: {self.num_classes}")
        print(f"- Head hidden layers: {self.num_head_layers} (width={self.hidden_dim})")
        print(f"- Head parameters: {p_counts['total']:,}")
        print(f"- Training samples: {len(self.train_embeddings)}")
        print(f"- Test samples: {len(self.test_embeddings) if self.test_embeddings is not None else 0}")

    def load_model(self):
        """Load the embedding extractor model."""
        if not os.path.exists(self.embedding_extractor_path):
            raise FileNotFoundError(f"Model file not found: {self.embedding_extractor_path}")

        start = time.time()
        try:
            from tflite_runtime.interpreter import Interpreter as InterpreterCls
        except ImportError:
            import tensorflow as tf
            InterpreterCls = tf.lite.Interpreter

        self.interpreter = InterpreterCls(model_path=self.embedding_extractor_path, num_threads=self.num_threads)
        self.interpreter.allocate_tensors()
        self.model_type = 'tflite'
        end = time.time()
        print(f"Time to load model on Mac: {end - start:.4f} seconds")

    def load_data(self):
        """Load and preprocess local data with 80-20 split and optional embedding cache."""
        print("Loading local data on Mac...")
        image_paths, labels, label_map = get_image_paths(self.data_dir)

        if self.embedding_cache:
            if os.path.exists(self.embedding_cache):
                t_c = time.time()
                cached = np.load(self.embedding_cache)
                all_embeddings = cached["embeddings"].astype(np.float32)
                all_labels = cached["labels"]
                print(f"[CACHE] Loaded {len(all_embeddings)} cached embeddings from {self.embedding_cache} in {time.time() - t_c:.3f}s")
            else:
                print(f"[CACHE] Extracting all {len(image_paths)} embeddings on Mac to populate {self.embedding_cache}...")
                t0 = time.time()
                all_embeddings = extract_embeddings_tflite(image_paths, self.interpreter, self.input_size)
                all_labels = np.array(labels)
                print(f"Time to extract all embeddings on Mac: {time.time() - t0:.2f}s")
                os.makedirs(os.path.dirname(os.path.abspath(self.embedding_cache)), exist_ok=True)
                tmp_cache = f"{self.embedding_cache}.{os.getpid()}.tmp.npz"
                np.savez(tmp_cache, embeddings=all_embeddings, labels=all_labels)
                os.replace(tmp_cache, self.embedding_cache)

            train_and_val_dataset, test_dataset = shuffle_and_split(
                all_embeddings,
                all_labels,
                val_percent=self.val_percent,
                test_percent=self.test_percent,
                seed=self.seed,
            )
            self.train_embeddings = train_and_val_dataset["data_train"]
            self.train_labels = train_and_val_dataset["labels_train"]
            self.test_embeddings = test_dataset["data_test"]
            self.test_labels = test_dataset["labels_test"]
        else:
            train_and_val_dataset, test_dataset = shuffle_and_split(
                image_paths,
                labels,
                val_percent=self.val_percent,
                test_percent=self.test_percent,
                seed=self.seed,
            )
            start = time.time()
            print("Extracting training embeddings on Mac...")
            self.train_embeddings = extract_embeddings_tflite(
                train_and_val_dataset['data_train'], self.interpreter, self.input_size
            )
            self.train_labels = train_and_val_dataset['labels_train']
            print(f"Time to extract training embeddings on Mac: {time.time() - start:.2f} seconds")

            eval_paths = test_dataset['data_test'] if len(test_dataset['data_test']) > 0 else train_and_val_dataset['data_val']
            eval_labels = test_dataset['labels_test'] if len(test_dataset['data_test']) > 0 else train_and_val_dataset['labels_val']
            if len(eval_paths) > 0:
                self.test_embeddings = extract_embeddings_tflite(eval_paths, self.interpreter, self.input_size)
                self.test_labels = eval_labels

        self.val_embeddings = self.test_embeddings if self.test_embeddings is not None else np.array([])
        self.val_labels = self.test_labels if self.test_labels is not None else np.array([])
        self.dataset = {
            'data_train': self.train_embeddings,
            'labels_train': self.train_labels,
            'data_val': self.val_embeddings,
            'labels_val': self.val_labels,
        }

    def get_parameters(self, config):
        return self.head.get_weights()

    def set_parameters(self, parameters):
        self.head.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        learning_rate = config.get("learning_rate", 1e-2)
        batch_size = config.get("batch_size", 200)
        local_epochs = config.get("local_epochs", getattr(self, "local_epochs", 10))
        server_round = config.get("server_round", len(self.round_train_times) + 1)

        n_train = len(self.train_embeddings)
        iter_per_epoch = max(1, int(np.ceil(n_train / batch_size)))
        num_iter = local_epochs * iter_per_epoch
        round_seed = (self.seed + int(server_round)) if self.seed is not None else None

        start = time.time()
        self.head.train_with_sgd(
            self.dataset,
            num_iter,
            learning_rate,
            batch_size=batch_size,
            seed=round_seed,
        )
        round_t = time.time() - start
        self.round_train_times.append(round_t)

        metrics = {}
        if self.track_accuracy:
            train_loss = self.head.compute_loss(self.train_embeddings, self.train_labels)
            train_acc = self.head.get_accuracy(self.train_embeddings, self.train_labels)
            self._last_fit_metrics = {
                "round": int(server_round),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "round_train_sec": float(round_t),
                "cum_train_sec": float(sum(self.round_train_times)),
            }
            metrics = {"train_loss": float(train_loss), "train_acc": float(train_acc)}

        return self.get_parameters(config), len(self.train_embeddings), metrics

    def evaluate(self, parameters, config):
        if not self.track_accuracy or self.test_embeddings is None or len(self.test_embeddings) == 0:
            return 0.0, len(self.train_embeddings), {}

        self.set_parameters(parameters)
        server_round = int(config.get("server_round", len(self.round_history) + 1))
        test_loss = self.head.compute_loss(self.test_embeddings, self.test_labels)
        test_acc = self.head.get_accuracy(self.test_embeddings, self.test_labels)

        rec = {
            "round": server_round,
            "train_loss": round(self._last_fit_metrics.get("train_loss", 0.0), 6),
            "train_acc": round(self._last_fit_metrics.get("train_acc", 0.0), 6),
            "test_loss": round(float(test_loss), 6),
            "test_acc": round(float(test_acc), 6),
            "round_train_sec": round(self._last_fit_metrics.get("round_train_sec", 0.0), 4),
            "cum_train_sec": round(self._last_fit_metrics.get("cum_train_sec", 0.0), 4),
        }
        self.round_history.append(rec)

        if self.accuracy_csv:
            with open(self.accuracy_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    rec["round"],
                    rec["train_loss"],
                    rec["train_acc"],
                    rec["test_loss"],
                    rec["test_acc"],
                    rec["round_train_sec"],
                    rec["cum_train_sec"],
                ])
                f.flush()

        print(f"[ROUND_METRICS] {json.dumps(rec)}")
        return float(test_loss), len(self.test_embeddings), {"accuracy": float(test_acc)}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_extractor_path", required=True,
                        help="Path to embedding extractor model (.tflite)")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing local training data")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes in the dataset")
    parser.add_argument("--server_address", default="localhost:8080",
                        help="Server address (default: localhost:8080)")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Number of CPU threads for TFLite Interpreter")
    parser.add_argument("--num_head_layers", type=int, default=0,
                        help="Number of hidden layers in classification head (0=baseline linear, 3/6/9/12=MLP)")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden layer width in classification head (default: 512)")
    parser.add_argument("--local_epochs", type=int, default=10,
                        help="Number of local SGD epochs per round (default: 10)")
    parser.add_argument("--input_size", type=int, nargs=2, default=[224, 224],
                        help="Input image size (height width) (default: 224 224)")
    parser.add_argument("--track_accuracy", action="store_true",
                        help="Evaluate global head on held-out test split every round")
    parser.add_argument("--accuracy_csv", default=None,
                        help="Optional path to save per-round test accuracy CSV")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic split, init, and SGD (-1 for non-deterministic)")
    parser.add_argument("--val_percent", type=float, default=0.0,
                        help="Validation split fraction (default: 0.0 for 80-20 split)")
    parser.add_argument("--test_percent", type=float, default=0.2,
                        help="Test split fraction (default: 0.2 for 80-20 split)")
    parser.add_argument("--embedding_cache", default=None,
                        help="Optional path to .npz file to cache all extracted image embeddings across seeds")

    args = parser.parse_args()

    client = FederatedClient(
        args.embedding_extractor_path,
        args.data_dir,
        args.num_classes,
        input_size=tuple(args.input_size),
        num_head_layers=args.num_head_layers,
        hidden_dim=args.hidden_dim,
        num_threads=args.num_threads,
        local_epochs=args.local_epochs,
        track_accuracy=args.track_accuracy,
        accuracy_csv=args.accuracy_csv,
        seed=args.seed,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
        embedding_cache=args.embedding_cache,
    )

    print(f"Starting Mac federated learning client, connecting to {args.server_address}")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()
