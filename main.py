import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_scheduler
import numpy as np
import json
import os
from tqdm import tqdm
import torch.backends.cudnn as cudnn

# Enable cudNN benchmarking for faster performance
cudnn.benchmark = True

# Define the Coconut-style reasoning layer
class LatentReasoningLayer(nn.Module):
    def __init__(self, hidden_dim, num_latent_tokens=1):
        super(LatentReasoningLayer, self).__init__()
        self.num_latent_tokens = num_latent_tokens
        self.latent_projection = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # Generate latent tokens by projecting hidden states
        latent_tokens = []
        current_state = hidden_states[:, -1, :]  # Use the last token as the initial state
        for _ in range(self.num_latent_tokens):
            current_state = self.activation(self.latent_projection(current_state))
            latent_tokens.append(current_state.unsqueeze(1))  # Add a new latent token dimension

        return torch.cat(latent_tokens, dim=1)  # Concatenate latent tokens


# Define the Coconut-enhanced Qwen2.5 model
class CoconutQwen(nn.Module):
    def __init__(self, base_model_name, hidden_dim, num_latent_tokens=1):
        super(CoconutQwen, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, output_hidden_states=True)
        self.latent_layer = LatentReasoningLayer(hidden_dim, num_latent_tokens)
        self.lm_head = self.base_model.lm_head  # Reference to the language model head

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the base model in language mode
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract hidden states from the base model's last layer
        hidden_states = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)

        # Generate latent tokens from the reasoning layer
        latent_tokens = self.latent_layer(hidden_states)  # Shape: (batch_size, num_latent_tokens, hidden_dim)

        # Concatenate latent tokens back into the sequence for final language generation
        extended_hidden_states = torch.cat([hidden_states, latent_tokens], dim=1)

        # Reuse the base model's language head to generate output tokens from extended states
        logits = self.lm_head(extended_hidden_states)

        loss = None
        if labels is not None:
            # Calculate loss only for the original sequence (ignore latent tokens)
            logits = logits[:, :hidden_states.size(1), :]  # Trim to original sequence length
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return {"loss": loss, "logits": logits}

    def generate(self, input_ids, attention_mask=None, max_length=200, **kwargs):
        # For text generation during inference, we need a custom generate function
        # that incorporates our latent reasoning tokens

        batch_size = input_ids.shape[0]
        current_input_ids = input_ids
        current_attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)

        for _ in range(max_length - input_ids.shape[1]):
            # Get model outputs
            with torch.no_grad():
                outputs = self.base_model(input_ids=current_input_ids,
                                         attention_mask=current_attention_mask,
                                         output_hidden_states=True)

                # Extract hidden states and generate latent tokens
                hidden_states = outputs.hidden_states[-1]
                latent_tokens = self.latent_layer(hidden_states)

                # Process hidden states with latent tokens to get next token prediction
                extended_hidden_states = torch.cat([hidden_states, latent_tokens[:, -1:, :]], dim=1)
                next_token_logits = self.lm_head(extended_hidden_states[:, -1:, :])

                # Sample next token (greedy for simplicity)
                next_token = torch.argmax(next_token_logits, dim=-1)

            # Append new token to sequence
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones((batch_size, 1), device=current_input_ids.device, dtype=torch.long)
            ], dim=1)

            # Check for end of sequence token
            if (next_token == tokenizer.eos_token_id).all():
                break

        return current_input_ids

    def save_pretrained(self, save_directory):
        # Save both the base model and custom layers
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save base model
        self.base_model.save_pretrained(os.path.join(save_directory, "base_model"))

        # Save custom latent layer parameters
        torch.save(self.latent_layer.state_dict(), os.path.join(save_directory, "latent_layer.pt"))

        # Save config
        config = {
            "hidden_dim": self.latent_layer.latent_projection.in_features,
            "num_latent_tokens": self.latent_layer.num_latent_tokens,
            "base_model_name": os.path.join(save_directory, "base_model")
        }

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, model_path):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)

        model = cls(config["base_model_name"], config["hidden_dim"], config["num_latent_tokens"])
        model.latent_layer.load_state_dict(torch.load(os.path.join(model_path, "latent_layer.pt")))

        return model

    def get_transformer_layers(self):
        """Get a list of transformer layers from the base model"""
        # This function may need to be adjusted based on the exact architecture of Qwen2.5
        # We're assuming a standard transformer architecture with layers accessible via model.model.layers
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            return self.base_model.model.layers
        # Alternative structure in some models
        elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'layers'):
            return self.base_model.transformer.layers
        else:
            # Try to find transformer layers in the model structure
            # This is a fallback and might need adjustment for specific models
            for name, module in self.base_model.named_children():
                if "block" in name or "layers" in name:
                    return module
        raise ValueError("Could not find transformer layers in the model structure")


# Load GSM8K dataset using Hugging Face's datasets library
def preprocess_gsm8k(example):
    # Combine question and answer into a single prompt
    prompt = f"<think> {example['question']} <answer> {example['answer']}"
    return {"input_text": prompt}


# Layer-wise training function
def train_model_layerwise(model, tokenizer, device, layers_per_stage=4, num_epochs=3, batch_size=8):
    print("Starting layer-wise training...")

    # Load and preprocess GSM8K dataset
    dataset = load_dataset('gsm8k','main' , split="train")
    dataset = dataset.map(preprocess_gsm8k)

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["input_text"],
                                  padding="max_length",
                                  truncation=True,
                                  max_length=512),
        batched=True  # Optimizes tokenization speed
    )

    # Prepare DataLoader for training
    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch])
        attention_mask = torch.tensor([item["attention_mask"] for item in batch])
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss calculation
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,         # Increase number of workers
        pin_memory=True        # Enable pinning memory for faster GPU transfers
    )

    # Get transformer layers
    try:
        transformer_layers = model.get_transformer_layers()
    except ValueError as e:
        print(f"Error getting transformer layers: {e}")
        print("Falling back to training the entire model at once")
        transformer_layers = []

    # Calculate the total number of stages based on layers_per_stage
    num_layers = len(transformer_layers) if transformer_layers else 0
    num_stages = (num_layers + layers_per_stage - 1) // layers_per_stage if num_layers > 0 else 1

    # First, freeze all transformer layers
    if num_layers > 0:
        for layer in transformer_layers:
            for param in layer.parameters():
                param.requires_grad = False

    # Always train the latent reasoning layer and LM head
    for param in model.latent_layer.parameters():
        param.requires_grad = True

    # Train in stages
    for stage in range(num_stages):
        print(f"\n--- Training Stage {stage+1}/{num_stages} ---")

        # Calculate start and end indices for this stage
        start_idx = stage * layers_per_stage
        end_idx = min((stage + 1) * layers_per_stage, num_layers)

        # If we have transformer layers, unfreeze only the current stage's layers
        if num_layers > 0:
            print(f"Unfreezing layers {start_idx} to {end_idx-1}")
            for i, layer in enumerate(transformer_layers):
                for param in layer.parameters():
                    param.requires_grad = (i >= start_idx and i < end_idx)
        else:
            # If we couldn't identify transformer layers, just train all parameters
            print("Training all parameters (couldn't identify layer structure)")
            for param in model.parameters():
                param.requires_grad = True

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params:,}")

        # Training setup for this stage
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
        num_training_steps = len(train_dataloader) * num_epochs

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        scaler = torch.cuda.amp.GradScaler()  # Initialize the scaler

        # Training loop for this stage
        model.train()
        progress_bar = tqdm(range(num_training_steps))

        for epoch in range(num_epochs):
            total_loss = 0.0

            for batch in train_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs["loss"]

                scaler.scale(loss).backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                progress_bar.update(1)
                for param in model.parameters():
                    param.grad = None  # Reset gradients

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

        # Save checkpoint after each stage
        save_dir = f"./coconut_qwen_stage_{stage+1}"
        model.save_pretrained(save_dir)
        print(f"Stage {stage+1} completed. Model checkpoint saved to {save_dir}")

        # After finishing a stage, freeze those layers
        if num_layers > 0:
            for i in range(start_idx, end_idx):
                if i < len(transformer_layers):
                    for param in transformer_layers[i].parameters():
                        param.requires_grad = False

    # Save final fine-tuned model for inference
    model.save_pretrained("./coconut_qwen_final")
    tokenizer.save_pretrained("./coconut_qwen_final")
    print("Layer-wise training completed and final model saved.")


# Main execution for training
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = CoconutQwen("Qwen/Qwen2.5-7B-Instruct", hidden_dim=3584, num_latent_tokens=2).to(device)

    # Train the model using layer-wise training
    # You can adjust the layers_per_stage parameter to control how many layers are trained at once
    train_model_layerwise(model, tokenizer, device, layers_per_stage=4, num_epochs=3)
  
