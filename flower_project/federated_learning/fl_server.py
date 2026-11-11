import flwr as fl
import numpy as np
from typing import Dict, List, Tuple, Optional
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History


class FederatedServer:
    def __init__(self, feature_dim: int, num_classes: int, min_clients: int = 2, num_head_layers: int = 0, hidden_dim: int = 512):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.min_clients = min_clients
        self.num_head_layers = num_head_layers
        self.hidden_dim = hidden_dim

        # Initialize global head parameters
        self.global_weights = self._initialize_parameters()

        print(f"Server initialized with:")
        print(f"- Feature dimension: {feature_dim}")
        print(f"- Number of classes: {num_classes}")
        print(f"- Minimum clients: {min_clients}")
        print(f"- Head layers: {num_head_layers} (width={hidden_dim})")

    def _initialize_parameters(self):
        """Initialize global head parameters."""
        from softmax_regression import create_classification_head
        head = create_classification_head(
            self.feature_dim,
            self.num_classes,
            num_head_layers=self.num_head_layers,
            hidden_dim=self.hidden_dim,
        )
        return head.get_weights()

    def get_evaluate_fn(self):
        """Return evaluation function for global model evaluation."""
        def evaluate(server_round: int, parameters: List[np.ndarray], config: Dict[str, str]) -> Optional[Tuple[float, Dict[str, float]]]:
            print(f"Global evaluation at round {server_round}")
            return 0.0, {"global_accuracy": 0.0}

        return evaluate

    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate evaluation metrics from clients safely."""
        acc_entries = [(num_examples, m["accuracy"]) for num_examples, m in metrics if "accuracy" in m]
        total_examples = sum(num_examples for num_examples, _ in acc_entries)
        if total_examples == 0:
            return {}
        weighted_acc = sum(num_examples * acc for num_examples, acc in acc_entries) / total_examples
        return {"accuracy": weighted_acc}
    
    def get_strategy(self):
        """Return federated averaging strategy."""
        return FedAvg(
            fraction_fit=1.0,  # Use all available clients for training
            fraction_evaluate=1.0,  # Use all available clients for evaluation
            min_fit_clients=self.min_clients,
            min_evaluate_clients=self.min_clients,
            min_available_clients=self.min_clients,
            evaluate_metrics_aggregation_fn=self.weighted_average,
            on_fit_config_fn=self.get_fit_config,
            on_evaluate_config_fn=self.get_evaluate_config,
            initial_parameters=None,  # Dynamic: automatically adopts client model shape (0, 1, 2, 3, 4 layers)
        )
    
    def get_fit_config(self, server_round: int):
        """Return training configuration for clients."""
        config = {
            "server_round": server_round,
            "learning_rate": 1e-2,
            "batch_size": 200,
            "local_epochs": 10,
        }
        return config
    
    def get_evaluate_config(self, server_round: int):
        """Return evaluation configuration for clients."""
        return {"server_round": server_round}


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dim", type=int, required=True,
                       help="Feature dimension from embedding extractor")
    parser.add_argument("--num_classes", type=int, default=10,
                       help="Number of classes in the dataset (default: 10 for CIFAR-10)")
    parser.add_argument("--min_clients", type=int, default=1,
                       help="Minimum number of clients (default: 1)")
    parser.add_argument("--num_rounds", type=int, default=40,
                       help="Number of federated learning rounds (default: 40)")
    parser.add_argument("--num_head_layers", type=int, default=0,
                       help="Number of hidden layers in classification head (0=baseline linear, 3/6/9/12=MLP)")
    parser.add_argument("--hidden_dim", type=int, default=512,
                       help="Hidden layer width in classification head (default: 512)")
    parser.add_argument("--auto_restart", action="store_true",
                       help="Automatically keep server running across multiple benchmark rounds")
    parser.add_argument("--server_address", default="0.0.0.0:8080",
                       help="Server address (default: 0.0.0.0:8080)")
    
    args = parser.parse_args()
    
    # Create federated server
    fed_server = FederatedServer(
        args.feature_dim,
        args.num_classes,
        args.min_clients,
        num_head_layers=args.num_head_layers,
        hidden_dim=args.hidden_dim,
    )
    
    # Get strategy
    strategy = fed_server.get_strategy()
    
    print(f"Starting federated learning server on {args.server_address}")
    print(f"Waiting for {args.min_clients} clients to connect...")
    print(f"Will run for {args.num_rounds} rounds")
    
    # Start federated learning server
    round_count = 0
    while True:
        round_count += 1
        print(f"[FL SERVER] >>> Listening on {args.server_address} (Waiting for client run #{round_count})...")
        try:
            strategy = fed_server.get_strategy()
            fl.server.start_server(
                server_address=args.server_address,
                config=fl.server.ServerConfig(num_rounds=args.num_rounds),
                strategy=strategy,
            )
            print("[FL SERVER] ✅ FL Round completed!")
        except Exception as e:
            print(f"[FL SERVER] Connection finished or reset: {e}")

        if not args.auto_restart:
            break
        import time
        time.sleep(1)


if __name__ == "__main__":
    main()
