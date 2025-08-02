import os
import torch
import argparse
import numpy as np
from model import SASRec
from utils import data_partition

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def extract_embeddings():
    """
    Extract item and positional embeddings from a trained SASRec model
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract embeddings from SASRec model')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--train_dir', required=True, help='Training directory')
    parser.add_argument('--model_path', required=True, help='Path to saved model state dict')
    parser.add_argument('--output_dir', default='embeddings', help='Output directory for embeddings')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=200, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--norm_first', action='store_true', default=False)
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Loading dataset: {args.dataset}")
    
    # Load dataset to get user and item numbers
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    print(f"Number of users: {usernum}")
    print(f"Number of items: {itemnum}")
    
    # Initialize the model
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    # Load the trained model state
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    print(f"Loading model from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)))
    model.eval()
    
    # Extract embeddings
    print("Extracting embeddings...")
    
    with torch.no_grad():
        # Extract item embeddings
        item_embeddings = model.item_emb.weight.data.cpu().numpy()
        print(f"Item embedding shape: {item_embeddings.shape}")
        
        # Extract positional embeddings
        pos_embeddings = model.pos_emb.weight.data.cpu().numpy()
        print(f"Positional embedding shape: {pos_embeddings.shape}")
    
    # Save embeddings
    item_emb_path = os.path.join(args.output_dir, f'{args.dataset}_item_embeddings.npy')
    pos_emb_path = os.path.join(args.output_dir, f'{args.dataset}_pos_embeddings.npy')
    
    np.save(item_emb_path, item_embeddings)
    np.save(pos_emb_path, pos_embeddings)
    
    print(f"Item embeddings saved to: {item_emb_path}")
    print(f"Positional embeddings saved to: {pos_emb_path}")
    
    # Save embedding metadata
    metadata = {
        'dataset': args.dataset,
        'usernum': usernum,
        'itemnum': itemnum,
        'hidden_units': args.hidden_units,
        'maxlen': args.maxlen,
        'item_embedding_shape': item_embeddings.shape,
        'pos_embedding_shape': pos_embeddings.shape,
        'model_path': args.model_path
    }
    
    metadata_path = os.path.join(args.output_dir, f'{args.dataset}_embedding_metadata.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Metadata saved to: {metadata_path}")
    
    # Optional: Save embeddings in text format for easier inspection
    if item_embeddings.shape[0] < 10000:  # Only for smaller datasets
        item_emb_txt_path = os.path.join(args.output_dir, f'{args.dataset}_item_embeddings.txt')
        pos_emb_txt_path = os.path.join(args.output_dir, f'{args.dataset}_pos_embeddings.txt')
        
        np.savetxt(item_emb_txt_path, item_embeddings, fmt='%.6f')
        np.savetxt(pos_emb_txt_path, pos_embeddings, fmt='%.6f')
        
        print(f"Item embeddings (text) saved to: {item_emb_txt_path}")
        print(f"Positional embeddings (text) saved to: {pos_emb_txt_path}")

def load_embeddings(dataset_name, output_dir='embeddings'):
    """
    Helper function to load saved embeddings
    """
    item_emb_path = os.path.join(output_dir, f'{dataset_name}_item_embeddings.npy')
    pos_emb_path = os.path.join(output_dir, f'{dataset_name}_pos_embeddings.npy')
    
    if not os.path.exists(item_emb_path) or not os.path.exists(pos_emb_path):
        raise FileNotFoundError("Embedding files not found. Please run extract_embeddings first.")
    
    item_embeddings = np.load(item_emb_path)
    pos_embeddings = np.load(pos_emb_path)
    
    return item_embeddings, pos_embeddings

def analyze_embeddings(dataset_name, output_dir='embeddings'):
    """
    Basic analysis of extracted embeddings
    """
    item_embeddings, pos_embeddings = load_embeddings(dataset_name, output_dir)
    
    print(f"=== Embedding Analysis for {dataset_name} ===")
    print(f"Item embeddings shape: {item_embeddings.shape}")
    print(f"Item embedding statistics:")
    print(f"  Mean: {np.mean(item_embeddings):.6f}")
    print(f"  Std: {np.std(item_embeddings):.6f}")
    print(f"  Min: {np.min(item_embeddings):.6f}")
    print(f"  Max: {np.max(item_embeddings):.6f}")
    
    print(f"\nPositional embeddings shape: {pos_embeddings.shape}")
    print(f"Positional embedding statistics:")
    print(f"  Mean: {np.mean(pos_embeddings):.6f}")
    print(f"  Std: {np.std(pos_embeddings):.6f}")
    print(f"  Min: {np.min(pos_embeddings):.6f}")
    print(f"  Max: {np.max(pos_embeddings):.6f}")
    
    # Check for padding embeddings (should be zero)
    print(f"\nPadding embeddings check:")
    print(f"Item padding embedding (index 0) norm: {np.linalg.norm(item_embeddings[0]):.6f}")
    print(f"Position padding embedding (index 0) norm: {np.linalg.norm(pos_embeddings[0]):.6f}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--analyze':
        # Analysis mode
        if len(sys.argv) < 3:
            print("Usage for analysis: python extract_embeddings.py --analyze <dataset_name> [output_dir]")
            sys.exit(1)
        
        dataset_name = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else 'embeddings'
        analyze_embeddings(dataset_name, output_dir)
    else:
        # Extraction mode
        extract_embeddings()
