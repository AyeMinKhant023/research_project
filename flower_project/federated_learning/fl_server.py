import flwr as fl
import numpy as np
from typing import Dict, List, Tuple, Optional
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History


class FederatedServer:
    def __init__(self, feature_dim: int, num_classes: int, min_clients: int = 2):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.min_clients = min_clients
        
        # Initialize global head parameters
        self.global_weights = self._initialize_parameters()
        
        print(f"Server initialized with:")
        print(f"- Feature dimension: {feature_dim}")
        print(f"- Number of classes: {num_classes}")
        print(f"- Minimum clients: {min_clients}")
    
    def _initialize_parameters(self):
        """Initialize global head parameters."""
        # Initialize weights and biases similar to SoftmaxRegression
        weight_scale = 5e-2
        W = np.random.randn(self.feature_dim, self.num_classes) * weight_scale
        b = np.zeros(self.num_classes)
        return [W.flatten(), b.flatten()]
    
    def get_evaluate_fn(self):
        """Return evaluation function for global model evaluation."""
        def evaluate(server_round: int, parameters: List[np.ndarray], config: Dict[str, str]) -> Optional[Tuple[float, Dict[str, float]]]:
            print(f"Global evaluation at round {server_round}")
            # In a real scenario, you might want to evaluate on a global test set
            # For now, we'll return a dummy evaluation
            return 0.0, {"global_accuracy": 0.0}
        
        return evaluate
    
    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate evaluation metrics from clients."""
        total_examples = sum(num_examples for num_examples, _ in metrics)
        
        if total_examples == 0:
            return {}
        
        # Calculate weighted average of accuracies
        weighted_acc = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
        
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
            initial_parameters=fl.common.ndarrays_to_parameters(self.global_weights),
        )
    
    def get_fit_config(self, server_round: int):
        """Return training configuration for clients."""
        config = {
            "server_round": server_round,
            "learning_rate": 1e-2,
            "batch_size": 100,
            "num_iter": 100,  # Local training iterations
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
    parser.add_argument("--num_classes", type=int, required=True,
                       help="Number of classes in the dataset")
    parser.add_argument("--min_clients", type=int, default=2,
                       help="Minimum number of clients (default: 2)")
    parser.add_argument("--num_rounds", type=int, default=10,
                       help="Number of federated learning rounds (default: 10)")
    parser.add_argument("--server_address", default="0.0.0.0:8080",
                       help="Server address (default: 0.0.0.0:8080)")
    
    args = parser.parse_args()
    
    # Create federated server
    fed_server = FederatedServer(
        args.feature_dim,
        args.num_classes,
        args.min_clients
    )
    
    # Get strategy
    strategy = fed_server.get_strategy()
    
    print(f"Starting federated learning server on {args.server_address}")
    print(f"Waiting for {args.min_clients} clients to connect...")
    print(f"Will run for {args.num_rounds} rounds")
    
    # Start federated learning server
    history = fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
    
    # Print final results
    print("\n" + "="*50)
    print("FEDERATED LEARNING COMPLETED")
    print("="*50)
    
    if history.metrics_distributed:
        final_accuracy = history.metrics_distributed['accuracy'][-1][1]
        print(f"Final global accuracy: {final_accuracy:.4f}")
    
    print(f"Training completed after {args.num_rounds} rounds")


if __name__ == "__main__":
    main()