# HealthBot: Your Personal Health Assistant

HealthBot is an AI-powered chatbot designed to assist users with health-related questions and advice. It provides 
information on fitness, nutrition, mental health, and general well-being.

## Features

- Personalized chat experience with user authentication
- Persistent chat history for each user
- AI-powered responses using advanced language models
- Web-based interface using Gradio

## Installation

1. Clone the repository:
```commandline 
git clone https://github.com/JakeFurtaw/HealthReelDemo
```
2. Install the required dependencies
```commandline
pip install -r requirements.txt
```
3. Set up the necessary environment variables or configuration files (see Configuration section).

## Usage

To run the HealthBot application:
```commandline
python appy.py
```
This will launch the Gradio interface in your default web browser.

## Configuration

The project uses a `config.py` file for various settings. Make sure to set up the following configurations:

- `EMBEDDING_MODEL_NAME`: The name of the embedding model to use
- `LLM_MODEL_NAME`: The name of the language model to use
- `CHAT_STORAGE_PATH`: The path to store chat history
- `TOKEN_LIMIT`: The token limit for chat memory
- `GRADIO_THEME`: The Gradio theme to use for the interface
- `CHATBOT_HEIGHT`: The height of the chat interface

## Project Structure

- `app.py`: Main entry point of the application
- `HealthBot.py`: Contains the Gradio interface implementation
- `chat.py`: Handles chat functionality and storage
- `models.py`: Sets up the embedding model, language model, and chat engine
- `utils.py`: Contains utility functions for the project

## Dependencies

- gradio
- llama_index
- langchain
- torch
- huggingface_hub

For a complete list of dependencies, refer to the `requirements.txt` file.

## Disclaimer

HealthBot is designed to provide general health information and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.