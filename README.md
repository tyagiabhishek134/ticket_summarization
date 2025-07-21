# Ticket Summarization System

This project implements a ticket summarization and Q&A system using transformer models. It provides a REST API for summarizing text and answering questions about tickets.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask application:
```bash
python app.py
```

## API Endpoints

### 1. Summarize Ticket
- **Endpoint**: `/summarize`
- **Method**: POST
- **Input**: JSON with `text` field
- **Output**: JSON with `summary` field

Example:
```bash
curl -X POST http://localhost:5000/summarize \
     -H "Content-Type: application/json" \
     -d '{"text": "Your ticket text here"}'
```

### 2. Question Answering
- **Endpoint**: `/qa`
- **Method**: POST
- **Input**: JSON with `context` and `question` fields
- **Output**: JSON with `answer` field

Example:
```bash
curl -X POST http://localhost:5000/qa \
     -H "Content-Type: application/json" \
     -d '{"context": "Your ticket text here", "question": "What is the issue?"}'
```

## Models Used
- Summarization: facebook/bart-large-cnn
- Question Answering: deepset/roberta-base-squad2
