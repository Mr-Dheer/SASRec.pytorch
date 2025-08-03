

#NOTE Use this to run the script in the terminal, also see below in the code it is written, which things are required to give in the terminal, which are default.
"""
python extract_user_embedding_perplexity.py \
    --dataset Magazine_Subscriptions \
    --train_dir default \
    --model_path Magazine_Subscriptions_default/SASRec.epoch=1000.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth \
    --output_dir user_embeddings \
    --hidden_units 50 \
    --maxlen 200 \
    --num_blocks 2 \
    --num_heads 1 \
    --device cuda \
    --embedding_type last_hidden \
    --min_sequence_length 3
"""


import os
import torch
import argparse
import numpy as np
from model import SASRec
from utils import data_partition
import pickle
from tqdm import tqdm

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def extract_user_embeddings():
    """
    Extract user embeddings from a trained SASRec model using user sequences
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract user embeddings from SASRec model')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--train_dir', required=True, help='Training directory')
    parser.add_argument('--model_path', required=True, help='Path to saved model state dict')
    parser.add_argument('--output_dir', default='user_embeddings', help='Output directory for user embeddings')
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
    parser.add_argument('--embedding_type', default='last_hidden', choices=['last_hidden', 'mean_pool', 'max_pool'],
                       help='Type of user embedding to extract')
    parser.add_argument('--min_sequence_length', default=3, type=int, 
                       help='Minimum sequence length for a user to be included')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Loading dataset: {args.dataset}")
    
    # Load dataset
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
    
    print("Extracting user embeddings...")
    
    user_embeddings = {}
    user_sequences = {}
    valid_users = 0
    
    with torch.no_grad():
        for user_id in tqdm(range(1, usernum + 1), desc="Processing users"):
            # Get user's training sequence
            if user_id not in user_train or len(user_train[user_id]) < args.min_sequence_length:
                continue
            
            # Prepare sequence (same as in evaluation)
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            
            # Add validation item if exists
            if user_id in user_valid and len(user_valid[user_id]) > 0:
                seq[idx] = user_valid[user_id][0]
                idx -= 1
            
            # Add training sequence in reverse order
            for item in reversed(user_train[user_id]):
                seq[idx] = item
                idx -= 1
                if idx == -1:
                    break
            
            seq_reshaped = seq.reshape(1, -1)  # Keep as numpy array
            log_feats = model.log2feats(seq_reshaped)
            
            # Extract user embedding based on specified type
            if args.embedding_type == 'last_hidden':
                # Use the last non-padding position's hidden state
                non_zero_positions = (seq != 0).nonzero()[0] if (seq != 0).any() else [args.maxlen - 1]
                last_pos = non_zero_positions[-1] if len(non_zero_positions) > 0 else args.maxlen - 1
                user_emb = log_feats[0, last_pos, :].cpu().numpy()
                
            elif args.embedding_type == 'mean_pool':
                # Mean pooling over non-padding positions
                mask = (seq != 0)
                if mask.any():
                    masked_feats = log_feats[0, mask, :]
                    user_emb = masked_feats.mean(dim=0).cpu().numpy()
                else:
                    user_emb = log_feats[0, -1, :].cpu().numpy()  # fallback to last position
                    
            elif args.embedding_type == 'max_pool':
                # Max pooling over non-padding positions
                mask = (seq != 0)
                if mask.any():
                    masked_feats = log_feats[0, mask, :]
                    user_emb = masked_feats.max(dim=0)[0].cpu().numpy()
                else:
                    user_emb = log_feats[0, -1, :].cpu().numpy()  # fallback to last position
            
            user_embeddings[user_id] = user_emb
            user_sequences[user_id] = seq
            valid_users += 1
    
    print(f"Extracted embeddings for {valid_users} users")
    
    # Convert to arrays for easier handling
    user_ids = list(user_embeddings.keys())
    embedding_matrix = np.array([user_embeddings[uid] for uid in user_ids])
    
    print(f"User embedding matrix shape: {embedding_matrix.shape}")
    
    # Save user embeddings
    embeddings_path = os.path.join(args.output_dir, f'{args.dataset}_user_embeddings_{args.embedding_type}.npy')
    user_ids_path = os.path.join(args.output_dir, f'{args.dataset}_user_ids_{args.embedding_type}.npy')
    sequences_path = os.path.join(args.output_dir, f'{args.dataset}_user_sequences_{args.embedding_type}.pkl')
    
    np.save(embeddings_path, embedding_matrix)
    np.save(user_ids_path, np.array(user_ids))
    
    # Save sequences for reference
    with open(sequences_path, 'wb') as f:
        pickle.dump(user_sequences, f)
    
    print(f"User embeddings saved to: {embeddings_path}")
    print(f"User IDs saved to: {user_ids_path}")
    print(f"User sequences saved to: {sequences_path}")
    
    # Save metadata
    metadata = {
        'dataset': args.dataset,
        'total_users': usernum,
        'valid_users': valid_users,
        'embedding_type': args.embedding_type,
        'hidden_units': args.hidden_units,
        'maxlen': args.maxlen,
        'min_sequence_length': args.min_sequence_length,
        'embedding_shape': embedding_matrix.shape,
        'model_path': args.model_path
    }
    
    metadata_path = os.path.join(args.output_dir, f'{args.dataset}_user_embedding_metadata_{args.embedding_type}.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Metadata saved to: {metadata_path}")

def load_user_embeddings(dataset_name, embedding_type='last_hidden', output_dir='user_embeddings'):
    """
    Helper function to load saved user embeddings
    """
    embeddings_path = os.path.join(output_dir, f'{dataset_name}_user_embeddings_{embedding_type}.npy')
    user_ids_path = os.path.join(output_dir, f'{dataset_name}_user_ids_{embedding_type}.npy')
    
    if not os.path.exists(embeddings_path) or not os.path.exists(user_ids_path):
        raise FileNotFoundError("User embedding files not found. Please run extract_user_embeddings first.")
    
    user_embeddings = np.load(embeddings_path)
    user_ids = np.load(user_ids_path)
    
    # Create a dictionary mapping user_id -> embedding
    user_emb_dict = {uid: emb for uid, emb in zip(user_ids, user_embeddings)}
    
    return user_embeddings, user_ids, user_emb_dict

def analyze_user_embeddings(dataset_name, embedding_type='last_hidden', output_dir='user_embeddings'):
    """
    Basic analysis of extracted user embeddings
    """
    user_embeddings, user_ids, user_emb_dict = load_user_embeddings(dataset_name, embedding_type, output_dir)
    
    print(f"=== User Embedding Analysis for {dataset_name} ({embedding_type}) ===")
    print(f"Number of users: {len(user_ids)}")
    print(f"Embedding dimension: {user_embeddings.shape[1]}")
    print(f"User embedding statistics:")
    print(f"  Mean: {np.mean(user_embeddings):.6f}")
    print(f"  Std: {np.std(user_embeddings):.6f}")
    print(f"  Min: {np.min(user_embeddings):.6f}")
    print(f"  Max: {np.max(user_embeddings):.6f}")
    
    # Compute pairwise similarities for a sample of users
    if len(user_ids) > 1:
        sample_size = min(100, len(user_ids))
        sample_embeddings = user_embeddings[:sample_size]
        
        # Compute cosine similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(sample_embeddings)
        
        # Remove diagonal (self-similarity)
        np.fill_diagonal(sim_matrix, np.nan)
        
        print(f"\nSimilarity analysis (sample of {sample_size} users):")
        print(f"  Mean cosine similarity: {np.nanmean(sim_matrix):.6f}")
        print(f"  Std cosine similarity: {np.nanstd(sim_matrix):.6f}")
        print(f"  Max cosine similarity: {np.nanmax(sim_matrix):.6f}")
        print(f"  Min cosine similarity: {np.nanmin(sim_matrix):.6f}")

def get_user_embedding_for_e2p(user_id, dataset_name, embedding_type='last_hidden', output_dir='user_embeddings'):
    """
    Get a specific user's embedding for use in E2P pipeline
    """
    _, _, user_emb_dict = load_user_embeddings(dataset_name, embedding_type, output_dir)
    
    if user_id not in user_emb_dict:
        raise ValueError(f"User {user_id} not found in embeddings")
    
    return user_emb_dict[user_id]

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--analyze':
        # Analysis mode
        if len(sys.argv) < 3:
            print("Usage for analysis: python extract_user_embeddings.py --analyze <dataset_name> [embedding_type] [output_dir]")
            sys.exit(1)
        
        dataset_name = sys.argv[2]
        embedding_type = sys.argv[3] if len(sys.argv) > 3 else 'last_hidden'
        output_dir = sys.argv[4] if len(sys.argv) > 4 else 'user_embeddings'
        analyze_user_embeddings(dataset_name, embedding_type, output_dir)
    else:
        # Extraction mode
        extract_user_embeddings()
