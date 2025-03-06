# Atom of Thoughts (AOT) Reasoning

An implementation of the Atom of Thoughts (AOT) reasoning technique that helps language models break down complex questions into simpler components for more accurate answers.

## Requirements

- Python 3.8+
- Ollama installed and running locally

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aot-reasoning.git
cd aot-reasoning

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Chat

```bash
python aot_reasoning.py --model llama3
```

This starts an interactive terminal chat where you can ask questions and get responses using the AOT reasoning technique.

### API Server

```bash
python aot_reasoning.py --mode api --port 8000 --model llama3
```

This starts an OpenAI-compatible API server on port 8000 that you can use with any OpenAI client library.

## Command Line Arguments

- `--model`: Ollama model to use (default: llama3)
- `--mode`: Run mode - "terminal" or "api" (default: terminal)
- `--port`: Port for API server (default: 8000)

## References
(arXiv:2502.12018)[https://doi.org/10.48550/arXiv.2502.12018]
