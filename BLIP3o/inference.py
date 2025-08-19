from dataclasses import dataclass
import torch
from PIL import Image
from transformers import AutoTokenizer
from blip3o.model.language_model.blip3o_qwen_inference import blip3oQwenForInferenceLM, blip3oQwenConfig
import os
import numpy as np


@dataclass
class FuturePredictionConfig:
    model_path: str = "/scratch/bc4227/INT-ACT-1/models/Pretrain"
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    # generation config
    future_step: int = 10
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = None

class FuturePredictionInference:
    def __init__(self, config: FuturePredictionConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._load_models()
        
    def _load_models(self):
        """Load the future prediction model and tokenizer."""
        print(f"Loading model from {self.config.model_path}...")
        
        # Load config and model
        model_config = blip3oQwenConfig.from_pretrained(self.config.model_path)
        self.model = blip3oQwenForInferenceLM.from_pretrained(
            self.config.model_path, 
            config=model_config,
            torch_dtype=self.config.dtype
        ).to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
        # Set model to evaluation mode
        self.model.eval()
        print("Model loaded successfully!")

    def predict_future(self, instruction: str, current_image: Image.Image) -> Image.Image:
        """
        Predict future image given current image and instruction.
        
        Args:
            instruction: Task instruction (e.g., "pick up the red block")
            current_image: PIL Image of current timestep
            
        Returns:
            PIL Image of predicted future scene
        """
        print(f"Predicting future for instruction: '{instruction}'")
        print(f"Current image size: {current_image.size}")
        
        with torch.no_grad():
            future_images = self.model.predict_future_image(
                instruction=instruction,
                current_image=current_image,
                tokenizer=self.tokenizer,
                future_step=self.config.future_step,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                return_tensor=False  # Return PIL Images
            )
        
        if len(future_images) > 0:
            print(f"Successfully generated {len(future_images)} future image(s)")
            return future_images[0]  # Return the first generated image
        else:
            print("Warning: No future images were generated")
            return None

    def predict_future_batch(self, instructions: list, current_images: list) -> list:
        """
        Predict future images for a batch of inputs.
        
        Args:
            instructions: List of task instructions
            current_images: List of PIL Images for current timesteps
            
        Returns:
            List of PIL Images of predicted future scenes
        """
        assert len(instructions) == len(current_images), "Instructions and images must have same length"
        
        future_images = []
        for instruction, current_image in zip(instructions, current_images):
            future_image = self.predict_future(instruction, current_image)
            future_images.append(future_image)
        
        return future_images


def create_dummy_image(size=(384, 384)):
    """Create a dummy image for testing when no real image is available."""
    # Create a simple scene with colored blocks
    img_array = np.zeros((*size, 3), dtype=np.uint8)
    
    # Add a "table" (brown rectangle at bottom)
    img_array[size[0]//2:, :] = [139, 69, 19]  # Brown color
    
    # Add some "blocks" (colored rectangles)
    block_size = size[0] // 8
    
    # Red block
    start_x, start_y = size[0]//2 - block_size, size[1]//3
    img_array[start_x:start_x+block_size, start_y:start_y+block_size] = [255, 0, 0]
    
    # Blue block  
    start_x, start_y = size[0]//2 - block_size, size[1]//2
    img_array[start_x:start_x+block_size, start_y:start_y+block_size] = [0, 0, 255]
    
    # Green block
    start_x, start_y = size[0]//2 - block_size, 2*size[1]//3
    img_array[start_x:start_x+block_size, start_y:start_y+block_size] = [0, 255, 0]
    
    return Image.fromarray(img_array)


def main():
    config = FuturePredictionConfig()
    inference = FuturePredictionInference(config)

    # Example tasks and images for future prediction
    tasks = [
        {
            "instruction": "put small spoon from basket to tray",
            "image_path": "/vast/bc4227/datasets/bridge_processed/episode0000000/frame000.jpg", 
        },
        {
            "instruction": "put the blue cube on the right side of the table on top of the rectangular block", 
            "image_path": "/vast/bc4227/datasets/bridge_processed/episode0000001/frame013.jpg",
        },
        {
            "instruction": "put the red object into the pot",
            "image_path": "/vast/bc4227/datasets/bridge_processed/episode0000002/frame030.jpg", 
        }
    ]

    output_dir = "future_predictions"
    os.makedirs(output_dir, exist_ok=True)

    for idx, task in enumerate(tasks):
        instruction = task["instruction"]
        image_path = task["image_path"]
        
        # Load current image
        if image_path and os.path.exists(image_path):
            current_image = Image.open(image_path).convert('RGB')
            print(f"Loaded image from: {image_path}")
        else:
            current_image = create_dummy_image()
            print("Using dummy image for testing")
        
        # Save current image for reference
        current_save_path = os.path.join(output_dir, f"current_{idx:02d}.png")
        current_image.save(current_save_path)
        print(f"Saved current image: {current_save_path}")
        
        try:
            # Predict future image
            future_image = inference.predict_future(instruction, current_image)
            
            if future_image is not None:
                # Save future prediction
                future_save_path = os.path.join(output_dir, f"future_{idx:02d}.png")
                future_image.save(future_save_path)
                print(f"Saved future prediction: {future_save_path}")
                
                # Save instruction as text file
                instruction_path = os.path.join(output_dir, f"instruction_{idx:02d}.txt")
                with open(instruction_path, 'w') as f:
                    f.write(f"Instruction: {instruction}\n")
                    f.write(f"Future step: {config.future_step}\n")
                print(f"Saved instruction: {instruction_path}")
            else:
                print(f"Failed to generate future image for task {idx}")
                
        except Exception as e:
            print(f"Error processing task {idx}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()  