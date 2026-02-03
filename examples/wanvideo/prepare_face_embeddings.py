"""
Face Embedding Preparation Script

Quick utility to extract and save face embeddings from reference images.
This is the first step before training.
"""

import torch
import argparse
from pathlib import Path
from diffsynth.utils.face_embedding import (
    FaceEmbeddingExtractor,
    FaceEmbeddingDataset,
    extract_face_embeddings
)


def prepare_face_embeddings(
    face_images_dir: str,
    output_dir: str,
    identity_name: str = "my_identity",
    encoder_type: str = "clip",
    aggregation: str = "weighted_mean",
    device: str = "cuda",
):
    """
    Extract and save face embeddings from reference images.
    
    Args:
        face_images_dir: Directory with 10-15 face reference images
        output_dir: Where to save embeddings
        identity_name: Name for this identity
        encoder_type: "clip" or "dinov3"
        aggregation: "mean", "weighted_mean", or "median"
        device: Device to use
    """
    print("=" * 80)
    print("FACE EMBEDDING EXTRACTION")
    print("=" * 80)
    
    face_images_dir = Path(face_images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    print(f"\n[1/3] Scanning for face images in: {face_images_dir}")
    image_paths = list(face_images_dir.glob("*.jpg")) + \
                  list(face_images_dir.glob("*.png")) + \
                  list(face_images_dir.glob("*.jpeg"))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {face_images_dir}")
    
    print(f"  Found {len(image_paths)} images")
    
    if len(image_paths) < 3:
        print("  ⚠ Warning: Less than 3 images. Recommend 10-15 for best results.")
    elif len(image_paths) > 20:
        print("  ⚠ Warning: More than 20 images. Using first 20 for efficiency.")
        image_paths = image_paths[:20]
    
    # Extract embeddings
    print(f"\n[2/3] Extracting embeddings using {encoder_type}...")
    identity_embedding, all_embeddings = extract_face_embeddings(
        image_paths=image_paths,
        encoder_type=encoder_type,
        device=device,
        aggregation=aggregation,
    )
    
    print(f"  ✓ Identity embedding shape: {identity_embedding.shape}")
    print(f"  ✓ All embeddings shape: {all_embeddings.shape}")
    
    # Create dataset and save
    print(f"\n[3/3] Saving embeddings...")
    dataset = FaceEmbeddingDataset(name=identity_name)
    
    for i, (emb, path) in enumerate(zip(all_embeddings, image_paths)):
        dataset.add_embedding(emb, path)
    
    dataset.identity_embedding = identity_embedding
    dataset.save(output_dir)
    
    # Also save the aggregated embedding separately for easy loading
    torch.save(identity_embedding, output_dir / "identity_embedding.pt")
    
    print(f"  ✓ Saved to: {output_dir}")
    print(f"  ✓ Identity embedding: {output_dir / 'identity_embedding.pt'}")
    print(f"  ✓ Full dataset: {output_dir / 'metadata.pt'}")
    
    print("\n" + "=" * 80)
    print("✓ Face embedding extraction complete!")
    print("\nNext steps:")
    print("  1. Train the face LoRA using train_face_lora.py")
    print("  2. Generate videos using face_identity_inference.py")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract face embeddings from reference images"
    )
    
    parser.add_argument(
        "--face_images_dir",
        type=str,
        required=True,
        help="Directory containing 10-15 clear face images"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/face_embeddings",
        help="Output directory for embeddings"
    )
    
    parser.add_argument(
        "--identity_name",
        type=str,
        default="my_identity",
        help="Name for this face identity"
    )
    
    parser.add_argument(
        "--encoder",
        type=str,
        default="clip",
        choices=["clip", "dinov3"],
        help="Encoder type (clip recommended)"
    )
    
    parser.add_argument(
        "--aggregation",
        type=str,
        default="weighted_mean",
        choices=["mean", "weighted_mean", "median"],
        help="How to aggregate multiple face embeddings"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    prepare_face_embeddings(
        face_images_dir=args.face_images_dir,
        output_dir=args.output_dir,
        identity_name=args.identity_name,
        encoder_type=args.encoder,
        aggregation=args.aggregation,
        device=args.device,
    )
