# Create Your Own LLM: Fine-Tuning GPT-2 on Custom Text Data

This repository provides a Jupyter notebook that fine-tunes GPT-2 on a custom text dataset using the Hugging Face Transformers library. The project, **Create Your Own LLM**, enables you to adapt GPT-2 to generate domain-specific text based on your own dataset.

## Features
- **Hugging Face Transformers:** Uses the `transformers` library to load and fine-tune GPT-2.
- **Dataset Tokenization and Padding:** Custom tokenization and dynamic padding using the `AutoTokenizer` and `DataCollatorForLanguageModeling`.
- **Training with Scheduler:** Includes a linear learning rate scheduler with warmup steps using `get_linear_schedule_with_warmup`.
- **Supports GPU Training:** Automatically detects GPU availability and moves the model to GPU if available.
- **Saving the Fine-Tuned Model:** After training, both the model and tokenizer are saved for future use.

## Requirements

Before you begin, ensure you have the following installed:

- Python 3.7+
- Jupyter Notebook or JupyterLab
- PyTorch
- Hugging Face Transformers
- Datasets
- TQDM

You can install the necessary dependencies with the following commands:

```bash
pip install torch transformers datasets tqdm notebook
```

## Getting Started
1. Clone the Repository
Start by cloning the repository to your local machine:

```bash
git clone https://github.com/yourusername/create-your-own-llm.git
cd create-your-own-llm
```

2. Prepare Your Dataset
Replace the placeholder dataset `Your_Data.txt` with your own custom dataset in the same directory. Your dataset should be in plain text format with one example per line, which will be tokenized and fed into GPT-2 for training.

3. Run the Jupyter Notebook
Since this project uses a Jupyter Notebook file (.ipynb), follow these steps to run the notebook.
Open a terminal and navigate to the directory where the notebook is located:

```bash
jupyter notebook
```

A web interface will open in your default browser. From the interface, locate and open the fine_tune_gpt2.ipynb notebook file.
Run the notebook cells sequentially by clicking Cell -> Run All, or run them one by one to step through the process.

4. Adjusting Hyperparameters
You can easily modify key hyperparameters like batch_size, epochs, learning_rate, and max_length directly within the notebook:

```bash
# Set up the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3
batch_size = 2
max_length = 512
```

You can adjust these to fit your dataset size and hardware capabilities. Larger datasets may require more epochs, while smaller batches may be necessary for limited GPU memory.

5. Save and Use the Fine-Tuned Model
Once training is complete, the fine-tuned model and tokenizer will be saved to the ./fine_tuned_model directory. You can load and use them for inference in the future by following these steps:

Loading the fine-tuned model:
```bash
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')
model = AutoModelForCausalLM.from_pretrained('./fine_tuned_model')

# Example usage for text generation
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

output = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
```

This script demonstrates how to use the fine-tuned model to generate new text based on a provided prompt.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
