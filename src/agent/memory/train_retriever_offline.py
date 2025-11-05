"""
Offline training script for Soft Q Retriever
Train the retriever model based on sample files
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

path = Path(__file__).parent.parent.parent.resolve()  # /root/DREAM/src
if str(path) not in sys.path:
    sys.path.append(str(path))

from agent.memory.memory_manager import MemoryManager
from agent.plan.online_predict import RCRankPredictor


def load_samples(samples_path):
    """Load samples from JSONL file (Experience format)"""
    samples = []
    if not os.path.exists(samples_path):
        print(f"Error: Sample file does not exist: {samples_path}")
        return samples

    line_count = 0
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                # Validate sample format: should contain query_info, case_info, reward
                if "query_info" in sample and "case_info" in sample and "reward" in sample:
                    samples.append(sample)
                else:
                    print(f"Warning: Line {line_count} has incorrect format, missing required fields")
                    continue
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_count}: {e}")
                continue

    print(f"Successfully loaded {len(samples)} valid samples (read {line_count} lines total)")
    return samples


def train_retriever_from_samples(
    samples_path,
    model_save_path,
    memory_manager_config,
    epochs=10,
    batch_size=32,
    learning_rate=1e-4,
):
    """
    Train retriever from sample file

    Args:
        samples_path: Path to sample file (JSONL format)
        model_save_path: Path to save the trained model
        memory_manager_config: MemoryManager configuration dictionary
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    # Load samples
    samples = load_samples(samples_path)
    if len(samples) == 0:
        print("Error: No available sample data")
        return

    # Initialize MemoryManager and Retriever
    planner_config = memory_manager_config.get("PLANNER_CONFIG", {})
    if not planner_config:
        planner_config = {"device": "cpu"}
    predictor = RCRankPredictor(planner_config)
    memory_manager = MemoryManager(memory_manager_config, predictor=predictor)
    retriever = memory_manager.retriever

    # Update training parameters
    retriever.batch_size = batch_size
    retriever.learning_rate = learning_rate
    retriever.optimizer = torch.optim.Adam(retriever.network.parameters(), lr=learning_rate)

    # Training loop
    print(f"Starting training, {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        # Shuffle data
        import random

        random.shuffle(samples)

        # Batch training
        valid_batches = 0
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i : i + batch_size]

            if len(batch_samples) < batch_size:
                continue  # Skip incomplete batch (last incomplete batch)

            # Prepare training data
            features_list = []
            rewards_list = []

            for sample in batch_samples:
                query_info = sample.get("query_info", {})
                case_info = sample.get("case_info", {})
                root_causes = case_info.get("root_causes", [])
                reward = sample.get("reward", 0)
                reward_binary = 1 if reward == 1 else 0

                features = retriever._compute_features(query_info, case_info, root_causes)
                features_list.append(features)
                rewards_list.append(reward_binary)

            # Ensure all features have consistent dimensions
            if len(features_list) > 0:
                feature_dim = features_list[0].shape[0]
                # Filter out features with inconsistent dimensions
                valid_features = []
                valid_rewards = []
                for feat, rew in zip(features_list, rewards_list):
                    if feat.shape[0] == feature_dim:
                        valid_features.append(feat)
                        valid_rewards.append(rew)

                features_list = valid_features
                rewards_list = valid_rewards

            # Convert to tensors
            features_tensor = torch.stack(features_list)
            rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32).to(retriever.device)

            # Forward pass
            predictions = retriever.network(features_tensor).squeeze()

            # Compute loss
            loss = F.mse_loss(predictions, rewards_tensor)

            # Backward pass
            retriever.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (optional, prevents gradient explosion)
            torch.nn.utils.clip_grad_norm_(retriever.network.parameters(), max_norm=1.0)

            retriever.optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            valid_batches += 1

        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}, Valid Batches: {valid_batches}")

        # # Update target network
        # if (epoch + 1) % 5 == 0:
        #     retriever.target_network.load_state_dict(retriever.network.state_dict())
        #     print(f"Updated target network (Epoch {epoch + 1})")

    # Save model
    retriever.save_model(model_save_path)
    print(f"Training completed! Model saved to: {model_save_path}")
    print(f"Processed {len(samples)} samples over {epochs} epochs")


def main():
    parser = argparse.ArgumentParser(description="Offline training for Soft Q Retriever")
    parser.add_argument(
        "--samples_path",
        type=str,
        required=True,
        help="Path to sample file (JSONL format)",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        required=True,
        help="Path to save the trained model",
    )
    parser.add_argument("--config_path", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    # Load configuration (JSON format)
    with open(args.config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    memory_manager_config = configs.get("MEMORY_MANAGER_CONFIG")

    # If PLANNER_CONFIG is not in configs, add default configuration
    if "PLANNER_CONFIG" not in configs:
        configs["PLANNER_CONFIG"] = {"device": "cpu"}

    # Add PLANNER_CONFIG to memory_manager_config if needed
    if "PLANNER_CONFIG" not in memory_manager_config:
        memory_manager_config["PLANNER_CONFIG"] = configs["PLANNER_CONFIG"]

    # Start training
    train_retriever_from_samples(
        samples_path=args.samples_path,
        model_save_path=args.model_save_path,
        memory_manager_config=memory_manager_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


# cd /root/DREAM/src/agent/memory
# python train_retriever_offline.py \
#     --samples_path experience/retriever_samples.jsonl \
#     --model_save_path model/soft_q_retriever.pt \
#     --config_path ../../../config/base_config.json \
#     --epochs 10 \
#     --batch_size 32 \
#     --learning_rate 0.0001

if __name__ == "__main__":
    main()
