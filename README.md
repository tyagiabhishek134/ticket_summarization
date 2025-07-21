<<<<<<< HEAD
# Ticket Summarization

This project implements an intelligent ticket analysis system that uses transformer models for summarization and question answering on IT support tickets.

## Features

- Ticket parsing and relationship detection
- Vector-based semantic search
- LLM-powered ticket summarization
- Question answering capabilities
- Flask API endpoints for integration

## Technologies Used

- Python 3.x
- transformers library for NLP tasks
- Flask for API endpoints
- BART model for summarization
- RoBERTa model for question answering
- sentence-transformers for semantic search

## Project Structure

```
ticket_summarization/
├── app.py                 # Flask application
├── requirements.txt       # Project dependencies
├── Data/                 # Data files
│   ├── it_support_ticket_data.xlsx
│   └── parsed_tickets_output.json
├── src/                  # Source code
│   ├── model.py         # ML model implementations
│   ├── vector_search.py # Semantic search functionality
│   ├── process_sample_tickets.py
│   └── parsing/         # Ticket parsing logic
│       ├── parser.py
│       └── T_template.yaml
└── tests/               # Test files
    └── test_parser.py
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/tyagiabhishek134/ticket_summarization.git
cd ticket_summarization
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Process sample tickets:
```bash
python src/process_sample_tickets.py
```

3. Perform vector search:
```bash
python src/vector_search.py
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Models Used
- Summarization: facebook/bart-large-cnn
- Question Answering: deepset/roberta-base-squad2
=======
# ticket_summarization
Built a rule-based IT support ticket retrieval system using only inter-ticket relations. Parsed tickets into structured fields and linked them via explicit references and text similarity. Used embeddings for semantic search and retrieved relevant tickets to assist LLM-based summarization
>>>>>>> c6708fad8a3850313d8dc14af4549387ddd68de8
