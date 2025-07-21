from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.model import TicketSummarizer
import logging
from logging.handlers import RotatingFileHandler
import os
import time

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)

app = Flask(__name__)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Ticket Summarization Service startup')

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"]
)

# Initialize summarizer
summarizer = TicketSummarizer()

def validate_text_length(text: str, min_length: int = 50, max_length: int = 5000) -> bool:
    """Validate text length is within acceptable bounds"""
    return min_length <= len(text.strip()) <= max_length

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

@app.route('/summarize', methods=['POST'])
@limiter.limit("5 per minute")
def summarize_ticket():
    """
    Endpoint to summarize ticket text
    
    Expected JSON body:
    {
        "text": "ticket text to summarize"
    }
    """
    start_time = time.time()
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        if not validate_text_length(text):
            return jsonify({'error': 'Text length must be between 50 and 5000 characters'}), 400

        summary = summarizer.summarize(text)
        if summary is None:
            return jsonify({'error': 'Failed to generate summary'}), 500

        response_time = time.time() - start_time
        app.logger.info(f'Summarization completed in {response_time:.2f}s')
        
        return jsonify({
            'summary': summary,
            'processing_time': f'{response_time:.2f}s'
        })

    except Exception as e:
        app.logger.error(f'Error in summarize_ticket: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/qa', methods=['POST'])
@limiter.limit("10 per minute")
def answer_question():
    """
    Endpoint to answer questions about ticket text
    
    Expected JSON body:
    {
        "context": "ticket text for context",
        "question": "question to answer"
    }
    """
    start_time = time.time()
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        context = data.get('context', '').strip()
        question = data.get('question', '').strip()

        if not context or not question:
            return jsonify({'error': 'Both context and question must be provided'}), 400

        if not validate_text_length(context):
            return jsonify({'error': 'Context length must be between 50 and 5000 characters'}), 400

        result = summarizer.answer_question(context, question)
        if 'error' in result:
            return jsonify({'error': result['error']}), 500

        response_time = time.time() - start_time
        app.logger.info(f'Question answering completed in {response_time:.2f}s')
        
        return jsonify({
            'answer': result['answer'],
            'confidence': result['confidence'],
            'processing_time': f'{response_time:.2f}s'
        })

    except Exception as e:
        app.logger.error(f'Error in answer_question: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
