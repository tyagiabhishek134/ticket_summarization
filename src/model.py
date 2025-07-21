from transformers import pipeline
import torch
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicketSummarizer:
    def __init__(self):
        """
        Initialize the summarizer with BART and RoBERTa models.
        Models are loaded only when needed to save memory.
        """
        self._summarizer = None
        self._qa_pipeline = None

    @property
    def summarizer(self):
        if self._summarizer is None:
            logger.info("Loading summarization model...")
            self._summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return self._summarizer

    @property
    def qa_pipeline(self):
        if self._qa_pipeline is None:
            logger.info("Loading question-answering model...")
            self._qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        return self._qa_pipeline

    def summarize(self, text: str) -> Optional[str]:
        """
        Summarize the input text using BART model
        
        Args:
            text (str): The input text to summarize
            
        Returns:
            Optional[str]: The summarized text or None if an error occurs
            
        Raises:
            ValueError: If input text is empty or too short
        """
        try:
            if not text or len(text.strip()) < 50:
                raise ValueError("Input text is too short. Minimum 50 characters required.")
            
            summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return None

    def answer_question(self, context: str, question: str) -> Dict[str, Any]:
        """
        Answer questions about the ticket using a question-answering model
        
        Args:
            context (str): The context text to search for answers
            question (str): The question to answer
            
        Returns:
            Dict[str, Any]: Dictionary containing answer and confidence score
            
        Raises:
            ValueError: If context or question is empty
        """
        try:
            if not context or not question:
                raise ValueError("Both context and question must be provided")
            
            result = self.qa_pipeline(question=question, context=context)
            return {
                'answer': result['answer'],
                'confidence': round(float(result['score']), 3)
            }
        except Exception as e:
            logger.error(f"Error in question answering: {str(e)}")
            return {'error': str(e)}
