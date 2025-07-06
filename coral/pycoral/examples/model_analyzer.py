import argparse
import contextlib
import os
import sys
import time
import glob
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from PIL import Image
import tensorflow as tf

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter


def count_tflite_parameters(tflite_model_path):
    """Count parameters in a TFLite model using a more robust method."""
    try:
        # Method 1: Try to count from tensor details
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        total_params = 0
        tensor_details = interpreter.get_tensor_details()
        
        print(f"\nAnalyzing model: {tflite_model_path}")
        print("Tensor details:")
        
        for i, tensor in enumerate(tensor_details):
            shape = tensor['shape']
            name = tensor['name']
            size = np.prod(shape) if len(shape) > 0 else 0
            
            # More comprehensive parameter detection
            is_weight = any(keyword in name.lower() for keyword in [
                'weight', 'kernel', 'bias', 'conv', 'dense', 'linear', 
                'depthwise', 'pointwise', 'scale', 'offset', 'beta', 'gamma'
            ])
            
            print(f"  Tensor {i}: {name} - Shape: {shape} - Size: {size} - Is Weight: {is_weight}")
            
            if is_weight and size > 0:
                total_params += size
        
        print(f"\nMethod 1 - Total parameters: {total_params:,}")
        
        # Method 2: Alternative approach - count all tensors with certain characteristics
        total_params_alt = 0
        for tensor in tensor_details:
            shape = tensor['shape']
            if len(shape) > 1:  # Multi-dimensional tensors are likely weights
                total_params_alt += np.prod(shape)
        
        print(f"Method 2 - Total parameters (alternative count): {total_params_alt:,}")
        
        # Method 3: Try to get model size information
        try:
            with open(tflite_model_path, 'rb') as f:
                model_content = f.read()
                model_size_bytes = len(model_content)
                print(f"Model file size: {model_size_bytes:,} bytes ({model_size_bytes/1024/1024:.2f} MB)")
        except:
            print("Could not determine model file size")
            
        return max(total_params, total_params_alt)  # Return the higher count
        
    except Exception as e:
        print(f"Error counting parameters: {e}")
        return 0


@contextlib.contextmanager
def test_image(path):
    """Returns opened test image."""
    with open(path, 'rb') as f:
        with Image.open(f) as image:
            yield image


def get_image_paths(data_dir):
    """Walks through data_dir and returns list of image paths and label map."""
    classes = None
    image_paths = []
    labels = []

    class_idx = 0
    for root, dirs, files in os.walk(data_dir):
        if root == data_dir:
            classes = dirs
        else:
            assert classes[class_idx] in root
            print('Reading dir: %s, which has %d images' % (root, len(files)))
            for img_name in files:
                image_paths.append(os.path.join(root, img_name))
                labels.append(class_idx)
            class_idx += 1
    
    return image_paths, labels, dict(zip(range(class_idx), classes))


def shuffle_and_split(image_paths, labels, val_percent=0.1, test_percent=0.1):
    """Shuffles and splits data into train, validation, and test sets."""
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    perm = np.random.permutation(image_paths.shape[0])
    image_paths = image_paths[perm]
    labels = labels[perm]

    num_total = image_paths.shape[0]
    num_val = int(num_total * val_percent)
    num_test = int(num_total * test_percent)
    num_train = num_total - num_val - num_test

    print(f"\n" + "="*60)
    print(f"DATASET SPLIT SUMMARY")
    print(f"="*60)
    print(f"Total images: {num_total}")
    print(f"Training set: {num_train} images ({num_train/num_total*100:.1f}%)")
    print(f"Validation set: {num_val} images ({num_val/num_total*100:.1f}%)")
    print(f"Test set: {num_test} images ({num_test/num_total*100:.1f}%)")
    print(f"="*60)

    train_and_val_dataset = {}
    train_and_val_dataset['data_train'] = image_paths[0:num_train]
    train_and_val_dataset['labels_train'] = labels[0:num_train]
    train_and_val_dataset['data_val'] = image_paths[num_train:num_train + num_val]
    train_and_val_dataset['labels_val'] = labels[num_train:num_train + num_val]
    test_dataset = {}
    test_dataset['data_test'] = image_paths[num_train + num_val:]
    test_dataset['labels_test'] = labels[num_train + num_val:]
    return train_and_val_dataset, test_dataset


def extract_embeddings(image_paths, interpreter):
    """Uses model to process images as embeddings."""
    input_size = common.input_size(interpreter)
    feature_dim = classify.num_classes(interpreter)
    embeddings = np.empty((len(image_paths), feature_dim), dtype=np.float32)
    for idx, path in enumerate(image_paths):
        with test_image(path) as img:
            common.set_input(interpreter, img.resize(input_size, Image.NEAREST))
            interpreter.invoke()
            embeddings[idx, :] = classify.get_scores(interpreter)
    return embeddings


def analyze_single_model(model_path, data_dir, output_dir):
    """Analyzes a single model and returns the results."""
    print(f"\n" + "="*80)
    print(f"MODEL ANALYSIS: {os.path.basename(model_path)}")
    print(f"="*80)
    
    # Count parameters in the model
    total_params = count_tflite_parameters(model_path)
    
    # Get image paths and labels (only once for all models)
    image_paths, labels, label_map = get_image_paths(data_dir)
    train_and_val_dataset, test_dataset = shuffle_and_split(image_paths, labels)
    
    # Initialize interpreter
    print(f"\n" + "-"*60)
    print(f"INITIALIZING MODEL")
    print(f"-"*60)
    interpreter = make_interpreter(model_path, device=':0')
    interpreter.allocate_tensors()
    
    # Record total runtime
    total_start_time = time.perf_counter()
    
    # Extract embeddings for training set
    print(f"\n" + "-"*60)
    print(f"EXTRACTING EMBEDDINGS")
    print(f"-"*60)
    print('Extract embeddings for data_train')
    t0 = time.perf_counter()
    train_and_val_dataset['data_train'] = extract_embeddings(
        train_and_val_dataset['data_train'], interpreter)
    t1 = time.perf_counter()
    train_runtime = t1 - t0
    print('Feature extractor for training dataset takes %.2f seconds' % train_runtime)

    print('Extract embeddings for data_val')
    t2 = time.perf_counter()
    train_and_val_dataset['data_val'] = extract_embeddings(
        train_and_val_dataset['data_val'], interpreter)
    t3 = time.perf_counter()
    val_runtime = t3 - t2
    print('Feature extractor for validation dataset takes %.2f seconds' % val_runtime)

    print('Extract embeddings for data_test')
    t4 = time.perf_counter()
    test_embeddings = extract_embeddings(test_dataset['data_test'], interpreter)
    t5 = time.perf_counter()
    test_runtime = t5 - t4
    print('Feature extractor for test dataset takes %.2f seconds' % test_runtime)
    
    total_end_time = time.perf_counter()
    total_runtime = total_end_time - total_start_time
    
    # Print final summary for this model
    print(f"\n" + "="*80)
    print(f"FINAL SUMMARY REPORT")
    print(f"="*80)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Total Samples: {len(image_paths):,}")
    print(f"Training Samples: {len(train_and_val_dataset['data_train']):,}")
    print(f"Validation Samples: {len(train_and_val_dataset['data_val']):,}")
    print(f"Test Samples: {len(test_dataset['data_test']):,}")
    print(f"")
    print(f"RUNTIME BREAKDOWN:")
    print(f"  Training set extraction: {train_runtime:.2f} seconds")
    print(f"  Validation set extraction: {val_runtime:.2f} seconds")
    print(f"  Test set extraction: {test_runtime:.2f} seconds")
    print(f"  Total runtime: {total_runtime:.2f} seconds")
    print(f"="*80)
    
    # Create shared result folder and save detailed results
    model_name = os.path.splitext(os.path.basename(model_path))[0]  # Remove .tflite extension
    result_folder = os.path.join(output_dir, "result of all models")
    os.makedirs(result_folder, exist_ok=True)

    # Save detailed results to txt file
    results_file = os.path.join(result_folder, f'result of {model_name}.txt')
    with open(results_file, 'w') as f:
        f.write(f"="*80 + "\n")
        f.write(f"MODEL ANALYSIS AND TRAINING REPORT\n")
        f.write(f"="*80 + "\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Output directory: {result_folder}\n")
        f.write(f"Analysis date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")
        f.write(f"="*80 + "\n")
        f.write(f"FINAL SUMMARY REPORT\n")
        f.write(f"="*80 + "\n")
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Total Samples: {len(image_paths):,}\n")
        f.write(f"Training Samples: {len(train_and_val_dataset['data_train']):,}\n")
        f.write(f"Validation Samples: {len(train_and_val_dataset['data_val']):,}\n")
        f.write(f"Test Samples: {len(test_dataset['data_test']):,}\n")
        f.write(f"\n")
        f.write(f"RUNTIME BREAKDOWN:\n")
        f.write(f"  Training set extraction: {train_runtime:.2f} seconds\n")
        f.write(f"  Validation set extraction: {val_runtime:.2f} seconds\n")
        f.write(f"  Test set extraction: {test_runtime:.2f} seconds\n")
        f.write(f"  Total runtime: {total_runtime:.2f} seconds\n")
        f.write(f"="*80 + "\n")
        f.write(f"\n")
        f.write(f"DATASET SPLIT SUMMARY:\n")
        f.write(f"Total images: {len(image_paths)}\n")
        f.write(f"Training set: {len(train_and_val_dataset['data_train'])} images ({len(train_and_val_dataset['data_train'])/len(image_paths)*100:.1f}%)\n")
        f.write(f"Validation set: {len(train_and_val_dataset['data_val'])} images ({len(train_and_val_dataset['data_val'])/len(image_paths)*100:.1f}%)\n")
        f.write(f"Test set: {len(test_dataset['data_test'])} images ({len(test_dataset['data_test'])/len(image_paths)*100:.1f}%)\n")
        f.write(f"\n")
        f.write(f"MODEL TECHNICAL DETAILS:\n")
        f.write(f"Model file size: {os.path.getsize(model_path):,} bytes ({os.path.getsize(model_path)/1024/1024:.2f} MB)\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"="*80 + "\n")
    
    print(f"Individual results saved to: {results_file}")
    
    # Return results dictionary
    return {
        'model_name': os.path.basename(model_path),
        'model_path': model_path,
        'total_parameters': total_params,
        'total_samples': len(image_paths),
        'training_samples': len(train_and_val_dataset['data_train']),
        'validation_samples': len(train_and_val_dataset['data_val']),
        'test_samples': len(test_dataset['data_test']),
        'train_runtime': train_runtime,
        'val_runtime': val_runtime,
        'test_runtime': test_runtime,
        'total_runtime': total_runtime,
        'result_folder': result_folder
    }


def create_plots(results_df, output_dir):
    """Create plots for the analysis results."""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Parameters vs Total Runtime (Main plot requested by professor)
    ax1.scatter(results_df['total_parameters'], results_df['total_runtime'], 
               c='blue', alpha=0.7, s=100)
    ax1.set_xlabel('Total Parameters')
    ax1.set_ylabel('Total Runtime (seconds)')
    ax1.set_title('Parameters vs Runtime')
    ax1.grid(True, alpha=0.3)
    
    # Add model names as labels
    for i, row in results_df.iterrows():
        ax1.annotate(row['model_name'], 
                    (row['total_parameters'], row['total_runtime']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    # Plot 2: Parameters vs Training Runtime
    ax2.scatter(results_df['total_parameters'], results_df['train_runtime'], 
               c='red', alpha=0.7, s=100)
    ax2.set_xlabel('Total Parameters')
    ax2.set_ylabel('Training Runtime (seconds)')
    ax2.set_title('Parameters vs Training Runtime')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Model Comparison - Runtime Breakdown
    models = results_df['model_name']
    x_pos = np.arange(len(models))
    
    ax3.bar(x_pos, results_df['train_runtime'], label='Training', alpha=0.7)
    ax3.bar(x_pos, results_df['val_runtime'], bottom=results_df['train_runtime'], 
            label='Validation', alpha=0.7)
    ax3.bar(x_pos, results_df['test_runtime'], 
            bottom=results_df['train_runtime'] + results_df['val_runtime'], 
            label='Test', alpha=0.7)
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Runtime (seconds)')
    ax3.set_title('Runtime Breakdown by Model')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameters Distribution
    ax4.bar(models, results_df['total_parameters'], alpha=0.7, color='green')
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Total Parameters')
    ax4.set_title('Model Parameters Comparison')
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'model_analysis_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")
    
    # Also create a focused plot just for parameters vs runtime (what professor specifically asked for)
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['total_parameters'], results_df['total_runtime'], 
               c='blue', alpha=0.7, s=100)
    plt.xlabel('Total Parameters')
    plt.ylabel('Total Runtime (seconds)')
    plt.title('Model Parameters vs Runtime Analysis')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(results_df['total_parameters'], results_df['total_runtime'], 1)
    p = np.poly1d(z)
    plt.plot(results_df['total_parameters'], p(results_df['total_parameters']), 
             "r--", alpha=0.8, label=f'Trend line')
    
    # Add model names as labels
    for i, row in results_df.iterrows():
        plt.annotate(row['model_name'], 
                    (row['total_parameters'], row['total_runtime']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.legend()
    plt.tight_layout()
    
    # Save the focused plot
    focused_plot_path = os.path.join(output_dir, 'parameters_vs_runtime.png')
    plt.savefig(focused_plot_path, dpi=300, bbox_inches='tight')
    print(f"Parameters vs Runtime plot saved to: {focused_plot_path}")
    
    plt.show()


def batch_analyze_models(model_paths, data_dir, output_dir):
    """Analyzes multiple models and creates summary report with plots."""
    
    all_results = []
    
    print(f"\n" + "="*80)
    print(f"BATCH MODEL ANALYSIS")
    print(f"="*80)
    print(f"Found {len(model_paths)} models to analyze")
    print(f"Models: {[os.path.basename(p) for p in model_paths]}")
    print(f"Output directory: {output_dir}")
    print(f"Individual results will be saved in separate 'result_of_[model_name]' folders")
    
    # Analyze each model
    for i, model_path in enumerate(model_paths, 1):
        print(f"\n" + "="*80)
        print(f"ANALYZING MODEL {i}/{len(model_paths)}: {os.path.basename(model_path)}")
        print(f"="*80)
        
        try:
            result = analyze_single_model(model_path, data_dir, output_dir)
            all_results.append(result)
            
            # Print individual summary
            print(f"\n" + "-"*60)
            print(f"SUMMARY FOR {result['model_name']}")
            print(f"-"*60)
            print(f"Parameters: {result['total_parameters']:,}")
            print(f"Total Samples: {result['total_samples']:,}")
            print(f"Total Runtime: {result['total_runtime']:.2f} seconds")
            print(f"Results saved to: {result['result_folder']}")
            print(f"-"*60)
            
        except Exception as e:
            print(f"Error analyzing {model_path}: {e}")
            continue
    
    # Create DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Print comprehensive summary
    print(f"\n" + "="*80)
    print(f"COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"="*80)
    print(f"Total Models Analyzed: {len(all_results)}")
    print(f"Total Samples per Model: {results_df['total_samples'].iloc[0] if len(results_df) > 0 else 'N/A'}")
    print(f"\nParameter Range: {results_df['total_parameters'].min():,} - {results_df['total_parameters'].max():,}")
    print(f"Runtime Range: {results_df['total_runtime'].min():.2f} - {results_df['total_runtime'].max():.2f} seconds")
    print(f"\nModel Performance Summary:")
    print(f"{'Model':<25} {'Parameters':<15} {'Runtime (s)':<15} {'Samples':<10}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['model_name']:<25} {row['total_parameters']:<15,} {row['total_runtime']:<15.2f} {row['total_samples']:<10,}")
    
    # Save comprehensive summary to txt file
    summary_file = os.path.join(output_dir, 'COMPREHENSIVE_ANALYSIS_SUMMARY.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE BATCH ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Analysis date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Models Analyzed: {len(all_results)}\n")
        f.write(f"Data Directory: {data_dir}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Total Samples per Model: {results_df['total_samples'].iloc[0] if len(results_df) > 0 else 'N/A'}\n")
        f.write(f"\n")
        f.write(f"STATISTICS:\n")
        f.write(f"Parameter Range: {results_df['total_parameters'].min():,} - {results_df['total_parameters'].max():,}\n")
        f.write(f"Runtime Range: {results_df['total_runtime'].min():.2f} - {results_df['total_runtime'].max():.2f} seconds\n")
        f.write(f"Average Runtime: {results_df['total_runtime'].mean():.2f} seconds\n")
        f.write(f"Average Parameters: {results_df['total_parameters'].mean():,.0f}\n")
        f.write(f"\n")
        f.write("="*80 + "\n")
        f.write("DETAILED MODEL PERFORMANCE SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"{'Model':<25} {'Parameters':<15} {'Runtime (s)':<15} {'Samples':<10}\n")
        f.write("-" * 70 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['model_name']:<25} {row['total_parameters']:<15,} {row['total_runtime']:<15.2f} {row['total_samples']:<10,}\n")
        f.write("="*80 + "\n")
        f.write("\n")
        f.write("RUNTIME BREAKDOWN BY MODEL:\n")
        f.write("-" * 50 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"\n{row['model_name']}:\n")
            f.write(f"  Training extraction: {row['train_runtime']:.2f} seconds\n")
            f.write(f"  Validation extraction: {row['val_runtime']:.2f} seconds\n")
            f.write(f"  Test extraction: {row['test_runtime']:.2f} seconds\n")
            f.write(f"  Total runtime: {row['total_runtime']:.2f} seconds\n")
        f.write("="*80 + "\n")
        f.write("\n")
        f.write("INDIVIDUAL RESULT LOCATIONS:\n")
        f.write("-" * 50 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['model_name']}: {row['result_folder']}\n")
        f.write("="*80 + "\n")
    
    print(f"\nComprehensive summary saved to: {summary_file}")
    
    # Save detailed results to CSV
    csv_path = os.path.join(output_dir, 'batch_analysis_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Detailed CSV results saved to: {csv_path}")
    
    # Create a simple text summary for quick reference
    quick_summary_file = os.path.join(output_dir, 'QUICK_SUMMARY.txt')
    with open(quick_summary_file, 'w') as f:
        f.write("QUICK REFERENCE SUMMARY\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total Models: {len(all_results)}\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\nFAST FACTS:\n")
        f.write(f"• Fastest Model: {results_df.loc[results_df['total_runtime'].idxmin(), 'model_name']} ({results_df['total_runtime'].min():.2f}s)\n")
        f.write(f"• Slowest Model: {results_df.loc[results_df['total_runtime'].idxmax(), 'model_name']} ({results_df['total_runtime'].max():.2f}s)\n")
        f.write(f"• Smallest Model: {results_df.loc[results_df['total_parameters'].idxmin(), 'model_name']} ({results_df['total_parameters'].min():,} params)\n")
        f.write(f"• Largest Model: {results_df.loc[results_df['total_parameters'].idxmax(), 'model_name']} ({results_df['total_parameters'].max():,} params)\n")
        f.write(f"\nSee 'COMPREHENSIVE_ANALYSIS_SUMMARY.txt' for detailed results\n")
    
    print(f"Quick summary saved to: {quick_summary_file}")
    
    # Create plots
    if len(results_df) > 0:
        create_plots(results_df, output_dir)
    
    print(f"\n" + "="*80)
    print(f"BATCH ANALYSIS COMPLETE")
    print(f"="*80)
    print(f"📁 Main output directory: {output_dir}")
    print(f"📁 Individual model results in: result_of_[model_name]/ folders")
    print(f"📄 Summary files:")
    print(f"   • COMPREHENSIVE_ANALYSIS_SUMMARY.txt")
    print(f"   • QUICK_SUMMARY.txt")
    print(f"   • batch_analysis_results.csv")
    print(f"📊 Plots: parameters_vs_runtime.png, model_analysis_plots.png")
    print(f"="*80)
    
    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embedding_extractor_path',
        help='Path to single embedding extractor tflite model.')
    parser.add_argument(
        '--models_directory',
        help='Directory containing multiple .tflite models to analyze.')
    parser.add_argument(
        '--model_pattern',
        default='*.tflite',
        help='Pattern to match model files (default: *.tflite).')
    parser.add_argument('--data_dir', required=True, help='Directory to data.')
    parser.add_argument(
        '--output_dir',
        default='/tmp/retrain/output',
        help='Path to directory to save results.')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        sys.exit('%s does not exist!' % args.data_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Determine which models to analyze
    model_paths = []
    
    if args.embedding_extractor_path:
        # Single model analysis
        if os.path.exists(args.embedding_extractor_path):
            model_paths = [args.embedding_extractor_path]
        else:
            sys.exit(f'Model file {args.embedding_extractor_path} does not exist!')
    
    elif args.models_directory:
        # Batch analysis
        if os.path.exists(args.models_directory):
            pattern = os.path.join(args.models_directory, args.model_pattern)
            model_paths = glob.glob(pattern)
            if not model_paths:
                sys.exit(f'No models found matching pattern {pattern}')
        else:
            sys.exit(f'Models directory {args.models_directory} does not exist!')
    
    else:
        sys.exit('Either --embedding_extractor_path or --models_directory must be provided!')

    # Run the analysis
    if len(model_paths) == 1:
        # Single model analysis
        result = analyze_single_model(model_paths[0], args.data_dir, args.output_dir)
        print(f"\nSingle model analysis complete. Results saved to {args.output_dir}")
    else:
        # Batch analysis
        results_df = batch_analyze_models(model_paths, args.data_dir, args.output_dir)
        print(f"\nBatch analysis complete. Results and plots saved to {args.output_dir}")


if __name__ == '__main__':
    main()