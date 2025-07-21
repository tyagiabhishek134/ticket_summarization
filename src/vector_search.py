import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
import yaml
from transformers import pipeline
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicketVectorSearch:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize vector search with a sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.ticket_embeddings = None
        self.tickets = None
        self.ticket_index = {}  # Map ticket IDs to their data
        
        # Initialize T5 model for text generation
        self.generator = pipeline('text2text-generation', model='t5-small', device=-1)  # -1 for CPU
    
    def _prepare_ticket_text(self, ticket: Dict) -> str:
        """
        Convert ticket data into a searchable text representation
        
        Args:
            ticket: Dictionary containing ticket data
            
        Returns:
            String representation of ticket
        """
        original = ticket.get('original_data', {})
        # Include more context for better semantic matching
        return f"""
        Title: {original.get('Title', '')}
        Description: {original.get('Description', '')}
        Category: {original.get('Category', '')}
        Priority: {original.get('Priority', '')}
        Status: {original.get('Status', '')}
        Resolution: {original.get('Resolution Notes', '')}
        Created: {original.get('Created Date', '')}
        Closed: {original.get('Closed Date', '')}
        """
    
    def _extract_yaml_data(self, yaml_str: str) -> Dict:
        """Extract structured data from YAML string"""
        try:
            return yaml.safe_load(yaml_str)
        except:
            return {}
    
    def load_tickets(self, json_path: str):
        """
        Load tickets from JSON file and create embeddings
        
        Args:
            json_path: Path to the JSON file containing processed tickets
        """
        logger.info(f"Loading tickets from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Extract tickets list and build index
        self.tickets = data.get('tickets', [])
        for ticket in self.tickets:
            ticket_id = ticket.get('original_data', {}).get('Ticket ID')
            if ticket_id:
                self.ticket_index[ticket_id] = ticket
        
        # Create text representations
        texts = [self._prepare_ticket_text(ticket) for ticket in self.tickets]
        
        # Generate embeddings
        logger.info("Generating ticket embeddings...")
        self.ticket_embeddings = self.model.encode(texts, convert_to_tensor=True)
        logger.info(f"Generated embeddings for {len(self.tickets)} tickets")
    
    def _get_related_tickets_info(self, ticket: Dict) -> List[Dict]:
        """Get information about related tickets"""
        related_info = []
        parsed = self._extract_yaml_data(ticket.get('parsed_structure', ''))
        if parsed and 'relations' in parsed:
            related_ids = parsed['relations'].get('related_tickets', [])
            for rel_id in related_ids:
                if rel_id in self.ticket_index:
                    rel_ticket = self.ticket_index[rel_id]
                    rel_data = rel_ticket.get('original_data', {})
                    related_info.append({
                        'ticket_id': rel_id,
                        'title': rel_data.get('Title', ''),
                        'status': rel_data.get('Status', ''),
                        'priority': rel_data.get('Priority', '')
                    })
        return related_info
    
    def generate_analysis_prompt(self, query: str, search_results: List[Tuple[Dict, float]]) -> str:
        """
        Generate an analysis prompt for the LLM
        
        Args:
            query: Original search query
            search_results: List of (ticket, score) tuples from search
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Based on the IT support tickets below, please provide:
1. A summary of the common issues and their relationships
2. Pattern analysis of priorities and categories
3. Recommendations for issue prevention
4. Suggested process improvements

Query: "{query}"

Relevant Tickets:
"""
        
        for ticket, score in search_results:
            original = ticket.get('original_data', {})
            related_tickets = self._get_related_tickets_info(ticket)
            
            prompt += f"""
Ticket {original.get('Ticket ID')} (Relevance Score: {score:.2f})
------------------------------
Title: {original.get('Title')}
Description: {original.get('Description')}
Category: {original.get('Category')}
Priority: {original.get('Priority')}
Status: {original.get('Status')}
Created: {original.get('Created Date')}
Closed: {original.get('Closed Date')}
Resolution: {original.get('Resolution Notes')}

Related Tickets:
{yaml.dump({'related': related_tickets}, default_flow_style=False)}
"""
        
        prompt += """
Please analyze the above tickets and provide:
1. Key patterns and trends
2. Root cause analysis
3. Suggested preventive measures
4. Process improvement recommendations

Format your response in a clear, structured manner with sections for each aspect of the analysis."""

        return prompt
    
    def generate_summary_prompt(self, query: str, search_results: List[Tuple[Dict, float]]) -> str:
        """Generate a concise summary prompt"""
        prompt = f"""Summarize the key points from these IT support tickets related to: "{query}"

Relevant Information:
"""
        for ticket, score in search_results:
            original = ticket.get('original_data', {})
            prompt += f"""
- Ticket {original.get('Ticket ID')}: {original.get('Title')}
  {original.get('Description')}
  Status: {original.get('Status')} | Priority: {original.get('Priority')}
"""
        
        prompt += "\nProvide a concise summary focusing on:\n1. Common themes\n2. Critical issues\n3. Resolution patterns"
        return prompt
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Search for tickets similar to the query
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (ticket, similarity_score) tuples
        """
        if self.ticket_embeddings is None:
            raise ValueError("No tickets loaded. Call load_tickets first.")
            
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Calculate similarities
        cos_scores = np.inner(query_embedding, self.ticket_embeddings)
        
        # Get top-k results
        top_indices = np.argsort(cos_scores)[-top_k:][::-1]
        
        return [(self.tickets[idx], float(cos_scores[idx])) for idx in top_indices]
    
    def get_llm_response(self, query: str, results: List[Tuple[dict, float]]) -> str:
        """
        Generate a response using BART model based on the query and relevant tickets.
        
        Args:
            query: User's question
            results: List of (ticket, similarity_score) tuples
        
        Returns:
            str: Generated response from the LLM
        """
        # Create a common template for all queries
        prompt = f"""Based on the IT support tickets provided below, help solve the following issue:

USER QUERY:
{query}

RELEVANT SUPPORT TICKETS:
"""
        
        # Add relevant tickets to the prompt
        for ticket, score in results:
            original = ticket.get('original_data', {})
            
            resolution = original.get('Resolution Notes', 'No resolution notes available')
            if resolution and resolution.lower() != 'nan' and resolution != 'None':
                resolution_info = f"Resolution: {resolution}"
            else:
                resolution_info = "Resolution: Not available"
                
            prompt += f"""
[Ticket {original.get('Ticket ID')} - Relevance Score: {score:.2f}]
----------------------------------------
Title: {original.get('Title')}
Description: {original.get('Description')}
Category: {original.get('Category')}
Status: {original.get('Status')}
{resolution_info}

"""
        
        prompt += """
Analyze the tickets and answer in this exact format:

1. IMMEDIATE SOLUTION
* First, try these steps:
* If that doesn't work:
* If still blocked:

2. LIKELY CAUSES
* Main reason:
* Other possibilities:

3. PREVENTION
* Best practice:
* Settings to check:

4. WHEN TO CALL IT
* Escalate if:
* Provide this info:"""
        
        # Generate response using the pipeline with improved parameters
        result = self.generator(
            prompt, 
            max_new_tokens=256,
            min_length=100,
            num_beams=5,
            do_sample=False,
            temperature=0.7,
            early_stopping=True
        )[0]['generated_text']
        
        # Format the response
        sections = [
            "IMMEDIATE SOLUTION:",
            "ROOT CAUSE ANALYSIS:",
            "PREVENTIVE MEASURES:",
            "ESCALATION GUIDELINES:"
        ]
        
        formatted_response = "\n" + "="*80 + "\n"
        formatted_response += "ANALYSIS RESULTS\n"
        formatted_response += "="*80 + "\n\n"
        
        # Try to extract and format sections
        current_section = ""
        for line in result.split('\n'):
            if any(section in line for section in sections):
                current_section = line
                formatted_response += f"\n{current_section}\n{'='*len(current_section)}\n"
            else:
                formatted_response += line + "\n"
        
        return formatted_response

def main():
    # Example usage
    vector_search = TicketVectorSearch()
    
    # Load and embed tickets
    json_path = Path(__file__).parent.parent / "Data" / "parsed_tickets_output.json"
    vector_search.load_tickets(str(json_path))
    
    # Example query
    query = "why after vpn login its showing me IP is blocked please tell me what to do?"
    
    # Print query with clear formatting
    logger.info("\n" + "="*80)
    logger.info("USER QUERY")
    logger.info("="*80)
    logger.info(query)
    logger.info("="*80 + "\n")
    
    # Get search results
    results = vector_search.search(query, top_k=3)
    
    # Print matched tickets
    logger.info("RELEVANT TICKETS FOUND")
    logger.info("-"*80)
    for ticket, score in results:
        original = ticket.get('original_data', {})
        logger.info(f"Ticket: {original.get('Ticket ID')} (Relevance: {score:.2f})")
        logger.info(f"Title: {original.get('Title')}")
        logger.info(f"Category: {original.get('Category')}")
        logger.info("-"*80)
    
    # Get and print LLM response
    response = vector_search.get_llm_response(query, results)
    logger.info("\nGENERATED RESPONSE")
    logger.info("="*80)
    logger.info(response)
    logger.info("="*80)

if __name__ == "__main__":
    main()

