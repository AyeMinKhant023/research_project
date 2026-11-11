import os
# Limit OpenBLAS spin-wait threads for small mini-batch GEMMs on Pi 5:
# 1 thread avoids multi-core spin-wait overhead on (200x512) matrices and cuts training current from ~2.8A to ~1.3A
# (TFLite CPU feature extraction still uses --num_threads=4 independently).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import flwr as fl
import numpy as np
import csv
import json
import time
import psutil
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from softmax_regression import SoftmaxRegression, create_classification_head
import contextlib
from typing import Dict, List, Tuple, Optional


def log_phase_marker(phase_name, timestamp=None):
    """Log phase markers to phases.txt in fnirsi-usb-power-data-logger."""
    if timestamp is None:
        timestamp = time.time()

    base_dir = os.path.dirname(__file__)
    phase_file_path = os.path.abspath(
        os.path.join(base_dir, "..", "fnirsi-usb-power-data-logger", "phases.txt")
    )
    os.makedirs(os.path.dirname(phase_file_path), exist_ok=True)

    mode = "w" if phase_name == "START_LOAD_MODEL" else "a"
    with open(phase_file_path, mode) as f:
        f.write(f"#phase {timestamp:.3f} {phase_name}\n")

    print(f"[PHASE MARKER] {phase_name} at timestamp {timestamp:.3f} (saved to {phase_file_path})")


@contextlib.contextmanager
def test_image(path):
    """Returns opened test image."""
    with open(path, 'rb') as f:
        with Image.open(f) as image:
            yield image


def get_image_paths(data_dir):
    """Walks through data_dir in deterministic sorted order and returns image paths and labels."""
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


def shuffle_and_split(image_paths, labels, val_percent=0.1, test_percent=0.1, seed: Optional[int] = 42):
    """Shuffles and splits data into train, validation, and test sets (deterministic when seed is set)."""
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    if seed is not None and seed >= 0:
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


def log_resource_usage(label, func, *args, **kwargs):
    process = psutil.Process()
    mem_before = process.memory_info().rss

    start_time = time.time()
    result = func(*args, **kwargs)
    mem_now = process.memory_info().rss
    end_time = time.time()

    mem_after = process.memory_info().rss
    ram_usage_psutil = (mem_after - mem_before) / (1024 * 1024)
    ram_usage_psutil_absolute = mem_now / (1024 * 1024)

    print(f"\n[RESOURCE] Usage for {label}")
    print(f"  Time: {end_time - start_time:.2f} sec")
    print(f"  Peak RAM Usage (psutil): {ram_usage_psutil:.2f} MB")
    print(f"  Absolute Peak RAM Usage (psutil): {ram_usage_psutil_absolute:.2f} MB")

    return result


def extract_embeddings(image_paths, interpreter, is_edgetpu=False):
    """Extract float32 embeddings on either EdgeTPU (pycoral) or CPU TFLite, auto-adapting to MobileNet & InceptionV3."""
    if is_edgetpu:
        input_size = common.input_size(interpreter)
        feature_dim = int(classify.num_classes(interpreter))
        embeddings = np.empty((len(image_paths), feature_dim), dtype=np.float32)
        for idx, path in enumerate(image_paths):
            with test_image(path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                common.set_input(interpreter, img.resize(input_size, Image.NEAREST))
                interpreter.invoke()
                embeddings[idx, :] = classify.get_scores(interpreter)
        return embeddings
    else:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        input_size = (int(input_shape[2]), int(input_shape[1]))  # width, height
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
                img_resized = img.resize(input_size, Image.NEAREST)
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
        num_threads: int = 4,
        num_head_layers: int = 0,
        hidden_dim: int = 256,
        local_epochs: int = 10,
        track_accuracy: bool = False,
        accuracy_csv: Optional[str] = None,
        seed: int = 42,
    ):
        self.embedding_extractor_path = os.path.expanduser(embedding_extractor_path)
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.num_threads = num_threads
        self.num_head_layers = num_head_layers
        self.hidden_dim = hidden_dim
        self.local_epochs = local_epochs
        self.track_accuracy = track_accuracy
        self.accuracy_csv = accuracy_csv
        self.seed = seed
        self.is_edgetpu = "edgetpu" in os.path.basename(self.embedding_extractor_path).lower()

        self.test_embeddings = None
        self.test_labels = None
        self.round_history = []
        self._last_fit_metrics = {}

        if self.accuracy_csv:
            os.makedirs(os.path.dirname(os.path.abspath(self.accuracy_csv)), exist_ok=True)
            with open(self.accuracy_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "train_loss", "train_acc", "test_loss", "test_acc", "round_train_sec", "cum_train_sec"])

        # Log start of model loading
        log_phase_marker("START_LOAD_MODEL")

        if self.is_edgetpu:
            from pycoral.utils.edgetpu import make_interpreter
            start = time.time()

            def load_interpreter(model_path):
                interpreter = make_interpreter(model_path, device=':0')
                interpreter.allocate_tensors()
                return interpreter

            self.interpreter = log_resource_usage(
                "Usage for Load Feature Extractor (EdgeTPU Interpreter)",
                load_interpreter,
                self.embedding_extractor_path,
            )
            end = time.time()
            print(f"[RUNTIME] Time for Load Feature Extractor (EdgeTPU Interpreter): {end - start:.4f} seconds")
        else:
            try:
                from tflite_runtime.interpreter import Interpreter as InterpreterCls
            except ImportError:
                import tensorflow as tf
                InterpreterCls = tf.lite.Interpreter

            start = time.time()

            def load_tflite_interpreter(model_path):
                interpreter = InterpreterCls(model_path=model_path, num_threads=self.num_threads)
                interpreter.allocate_tensors()
                return interpreter

            self.interpreter = log_resource_usage(
                f"Usage for Load Feature Extractor (TFLite Interpreter, {self.num_threads} threads)",
                load_tflite_interpreter,
                self.embedding_extractor_path,
            )
            end = time.time()
            print(f"[RUNTIME] Time for Load Feature Extractor (TFLite Interpreter, {self.num_threads} threads): {end - start:.4f} seconds")

        log_phase_marker("END_LOAD_MODEL")

        self.round_train_times = []

        # Load and preprocess data
        self.load_data()

        # Initialize head (0 layers = linear Softmax baseline, >0 = MLP with ReLU hidden layers)
        self.feature_dim = self.train_embeddings.shape[1]
        self.head = create_classification_head(
            self.feature_dim,
            self.num_classes,
            num_head_layers=self.num_head_layers,
            hidden_dim=self.hidden_dim,
            weight_scale=None,
            reg=0.0,
        )
        p_counts = self.head.count_parameters()

        print("Client initialized with:")
        print(f"- Feature dimension: {self.feature_dim}")
        print(f"- Number of classes: {self.num_classes}")
        print(f"- Head hidden layers: {self.num_head_layers} (width={self.hidden_dim})")
        print(f"- Head parameters: {p_counts['total']:,} (weights={p_counts['weights']:,}, biases={p_counts['biases']:,})")
        print(f"- Training samples: {len(self.train_embeddings)}")
        if self.test_embeddings is not None:
            print(f"- Test samples (accuracy tracking): {len(self.test_embeddings)}")

    def load_data(self):
        """Load and preprocess local data."""
        print("Loading local data...")
        image_paths, labels, label_map = get_image_paths(self.data_dir)
        train_and_val_dataset, test_dataset = shuffle_and_split(
            image_paths, labels, val_percent=0.1, test_percent=0.1, seed=self.seed
        )

        # Log start of feature extraction (training images only)
        log_phase_marker("START_EXTRACT_FEATURES")

        print("Extracting training embeddings...")
        start = time.time()
        self.train_embeddings = log_resource_usage(
            "Usage for Extract Train Embeddings (Perform)",
            extract_embeddings,
            train_and_val_dataset['data_train'],
            self.interpreter,
            self.is_edgetpu,
        )
        end = time.time()
        print(f"[RUNTIME] Time for Extract Train Embeddings (Perform): {end - start:.2f} seconds")
        self.train_labels = train_and_val_dataset['labels_train']

        log_phase_marker("END_EXTRACT_FEATURES")

        # Extract held-out test embeddings AFTER END_EXTRACT_FEATURES so extraction timing/energy stays pure
        if self.track_accuracy and len(test_dataset['data_test']) > 0:
            print(f"Extracting {len(test_dataset['data_test'])} held-out test embeddings for per-round accuracy tracking...")
            t_start = time.time()
            self.test_embeddings = extract_embeddings(
                test_dataset['data_test'], self.interpreter, self.is_edgetpu
            )
            self.test_labels = test_dataset['labels_test']
            print(f"[ACCURACY] Test embeddings ready in {time.time() - t_start:.2f}s (shape={self.test_embeddings.shape})")

        self.dataset = {
            'data_train': self.train_embeddings,
            'labels_train': self.train_labels,
            'data_val': self.test_embeddings if self.test_embeddings is not None else np.array([]),
            'labels_val': self.test_labels if self.test_labels is not None else np.array([]),
        }

    def get_parameters(self, config):
        """Return current head parameters."""
        return self.head.get_weights()

    def set_parameters(self, parameters):
        """Set head parameters received from server."""
        self.head.set_weights(parameters)

    def fit(self, parameters, config):
        """Train the head locally using local data."""
        self.set_parameters(parameters)

        learning_rate = config.get("learning_rate", 1e-2)
        batch_size = config.get("batch_size", 200)
        local_epochs = config.get("local_epochs", getattr(self, "local_epochs", 10))
        server_round = config.get("server_round", len(self.round_train_times) + 1)

        n_train = len(self.train_embeddings)
        iter_per_epoch = max(1, int(np.ceil(n_train / batch_size)))
        num_iter = local_epochs * iter_per_epoch

        print(f"Starting local training: {n_train} samples, {local_epochs} epochs ({num_iter} iterations, batch_size={batch_size}, lr={learning_rate})")

        log_phase_marker("START_TRAIN_HEAD")

        start = time.time()
        log_resource_usage(
            "Usage for Train Classification Head",
            self.head.train_with_sgd,
            self.dataset,
            num_iter,
            learning_rate,
            batch_size=batch_size,
        )
        end = time.time()
        round_t = end - start
        self.round_train_times.append(round_t)

        log_phase_marker("END_TRAIN_HEAD")

        print(f"[ROUND {len(self.round_train_times)}] Train Head Time: {round_t:.4f} seconds (cumulative: {sum(self.round_train_times):.4f}s)")
        print(f"[RUNTIME] Time for  Train Classification Head (Round {len(self.round_train_times)}): {round_t:.4f} seconds")

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
        """Evaluate global head on held-out test embeddings when accuracy tracking is enabled."""
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
    log_phase_marker("START_FL_CLIENT")
    global_start_time = time.time()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_extractor_path", required=True,
                        help="Path to embedding extractor tflite model")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing local training data")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes in the dataset")
    parser.add_argument("--server_address", default="localhost:8080",
                        help="Server address (default: localhost:8080)")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Number of threads for CPU TFLite Interpreter (1, 2, or 4)")
    parser.add_argument("--num_head_layers", type=int, default=0,
                        help="Number of hidden layers in classification head (0=linear, 3/6/9/12=MLP)")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Number of neurons per hidden layer (default: 256; use 512 for v2 benchmarks)")
    parser.add_argument("--local_epochs", type=int, default=10,
                        help="Number of local epochs over the training dataset per FL round (default: 10)")
    parser.add_argument("--track_accuracy", action="store_true",
                        help="Evaluate global head on held-out test split every round and log [ROUND_METRICS]")
    parser.add_argument("--accuracy_csv", default=None,
                        help="Optional path to save per-round test accuracy CSV on client")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic train/val/test split (-1 for non-deterministic)")
    parser.add_argument("--benchmark_head_only", action="store_true",
                        help="Directly execute local head training without requiring Flower server connection")

    args = parser.parse_args()
    if args.num_head_layers < 0:
        parser.error("--num_head_layers must be >= 0")

    client = FederatedClient(
        args.embedding_extractor_path,
        args.data_dir,
        args.num_classes,
        num_threads=args.num_threads,
        num_head_layers=args.num_head_layers,
        hidden_dim=args.hidden_dim,
        local_epochs=args.local_epochs,
        track_accuracy=args.track_accuracy,
        accuracy_csv=args.accuracy_csv,
        seed=args.seed,
    )

    if args.benchmark_head_only:
        n_train = len(client.train_embeddings)
        batch_size = 200
        iter_per_epoch = max(1, int(np.ceil(n_train / batch_size)))
        num_iter = args.local_epochs * iter_per_epoch
        print(f"[BENCHMARK] Executing on-device classification head training ({args.local_epochs} epochs = {num_iter} iterations)...")
        log_phase_marker("START_TRAIN_HEAD")
        start = time.time()
        log_resource_usage(
            "Usage for Train Classification Head",
            client.head.train_with_sgd,
            client.dataset,
            num_iter=num_iter,
            learning_rate=1e-2,
            batch_size=batch_size,
        )
        end = time.time()
        train_sec = end - start
        log_phase_marker("END_TRAIN_HEAD")
        print(f"[RUNTIME] Time for  Train Classification Head: {train_sec:.4f} seconds")
        global_end_time = time.time()
        print(f"[RUNTIME] Total Runtime (Full Client Run): {global_end_time - global_start_time:.2f} seconds")
        return

    log_phase_marker("START_FEDERATED_LEARNING")

    print(f"Starting federated learning client, connecting to {args.server_address}")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )

    log_phase_marker("END_FEDERATED_LEARNING")

    total_head_time = sum(client.round_train_times)
    n_rounds = len(client.round_train_times)
    avg_per_round = total_head_time / max(1, n_rounds)
    print(f"\n[RUNTIME] Time for  Train Classification Head: {total_head_time:.4f} seconds ({n_rounds} rounds total, avg {avg_per_round:.4f}s/round)")

    if client.round_history:
        final_rec = client.round_history[-1]
        print(f"[ACCURACY] Final Round {final_rec['round']}: Train Acc = {final_rec['train_acc']:.4f}, Test Acc = {final_rec['test_acc']:.4f}, Test Loss = {final_rec['test_loss']:.4f}")

    global_end_time = time.time()
    log_phase_marker("END_FL_CLIENT")

    total_runtime = global_end_time - global_start_time
    print(f"[RUNTIME] Total Runtime (Full Client Run): {total_runtime:.2f} seconds")

    with open("mydata.txt", "a") as f:
        f.write(f"#runtime {int(global_end_time)} {total_runtime:.2f}\n")


if __name__ == "__main__":
    main()
