import sys
import flatbuffers
import tflite.Model  # From generated flatbuffers schema

def load_model(path):
    with open(path, 'rb') as f:
        buf = f.read()
    return tflite.Model.Model.GetRootAsModel(buf, 0)

def get_operator_types(model):
    types = []
    for i in range(model.OperatorCodesLength()):
        types.append(model.OperatorCodes(i).BuiltinCode())
    return types

def print_model_info(model, label):
    print(f"\n[{label}]")
    print("Number of subgraphs:", model.SubgraphsLength())
    for i in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(i)
        print(f" Subgraph {i}:")
        print("  Number of layers (operators):", subgraph.OperatorsLength())

def compare_models(model1, model2):
    print("\n📊 Comparison Summary:")

    # (1) Compare number of layers
    sub1 = model1.Subgraphs(0)
    sub2 = model2.Subgraphs(0)
    print("\n(1) Number of layers:")
    print("CPU:", sub1.OperatorsLength())
    print("TPU:", sub2.OperatorsLength())

    # (2) Compare operator types
    print("\n(2) Operator types (as builtin codes):")
    ops1 = get_operator_types(model1)
    ops2 = get_operator_types(model2)
    print("CPU Operators:", ops1)
    print("TPU Operators:", ops2)

    # (3) Compare tensor count
    print("\n(3) Tensors in subgraph:")
    print("CPU:", sub1.TensorsLength())
    print("TPU:", sub2.TensorsLength())
    if sub1.TensorsLength() != sub2.TensorsLength():
        print("❗ Tensor count mismatch!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 compare_models.py model_cpu.tflite model_tpu.tflite")
        sys.exit(1)

    model_cpu = load_model(sys.argv[1])
    model_tpu = load_model(sys.argv[2])

    print_model_info(model_cpu, "CPU Model")
    print_model_info(model_tpu, "TPU Model")
    compare_models(model_cpu, model_tpu)

