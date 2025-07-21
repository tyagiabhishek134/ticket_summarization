import re
from typing import Dict, List, Optional, Any
import yaml
from transformers import pipeline
import logging
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TicketNode:
    """Represents a node in the ticket parse tree"""
    section: str
    content: Any
    children: List['TicketNode'] = None
    relations: Dict[str, str] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.relations is None:
            self.relations = {}

class TicketParser:
    def __init__(self, template_path: str):
        """
        Initialize the ticket parser with template and LLM
        
        Args:
            template_path: Path to the YAML template file
        """
        self.template = self._load_template(template_path)
        # Commented out LLM initialization to focus on rule-based parsing only
        # self.llm = pipeline("text2text-generation", model="google/flan-t5-base")
        
        # Compile regex patterns for rule-based parsing
        self.patterns = {
            'code_snippets': re.compile(r'```[\s\S]*?```|`[\s\S]*?`'),
            'error_messages': re.compile(r'(?:error|exception|failed|failure|error message|issue|problem)[:]\s*([^\n]+(?:\n(?!\n).+)*)', re.IGNORECASE | re.MULTILINE),
            'stack_traces': re.compile(r'(?:stack trace|traceback):([\s\S]*?)(?=\n\n|\Z)', re.IGNORECASE),
            'ticket_id': re.compile(r'Ticket\s+ID\s*:\s*(\S+)', re.IGNORECASE),
            'title': re.compile(r'Title\s*:\s*(.+?)(?=\n|$)', re.IGNORECASE),
            'priority': re.compile(r'Priority\s*:\s*(.+?)(?=\n|$)', re.IGNORECASE),
            'status': re.compile(r'Status\s*:\s*(.+?)(?=\n|$)', re.IGNORECASE),
            'created_date': re.compile(r'Created\s+Date\s*:\s*(.+?)(?=\n|$)', re.IGNORECASE),
            'closed_date': re.compile(r'Closed\s+Date\s*:\s*(.+?)(?=\n|$)', re.IGNORECASE),
            'assigned_to': re.compile(r'Assigned\s+To\s*:\s*(.+?)(?=\n|$)', re.IGNORECASE),
            'category': re.compile(r'Category\s*:\s*(.+?)(?=\n|$)', re.IGNORECASE),
            'description': re.compile(r'Description\s*:\s*(.+?)(?=\n\n|\n[A-Z]|$)', re.IGNORECASE | re.DOTALL),
            'technical_details': re.compile(r'Technical\s+Details:\s*\n(.+?)(?=\n\n|\n[A-Z]|$)', re.IGNORECASE | re.DOTALL),
            'steps_to_reproduce': re.compile(r'(?:Steps\s+to\s+Reproduce|Steps\s+Taken):\s*\n(.+?)(?=\n\n|\n[A-Z]|$)', re.IGNORECASE | re.DOTALL),
            'environment': re.compile(r'Environment:\s*\n(.*?)(?=\n\n|\n[A-Z]|$)', re.IGNORECASE | re.DOTALL),
            'related_tickets': re.compile(r'Related\s+Tickets\s*:\s*(.+?)(?=\n|$)', re.IGNORECASE),
            'resolution_notes': re.compile(r'Resolution\s+Notes\s*:\s*(.+?)(?=\n\n|\n[A-Z]|$)', re.IGNORECASE | re.DOTALL)
        }

    def _load_template(self, template_path: str) -> Dict:
        """Load and validate the YAML template"""
        try:
            with open(template_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading template: {str(e)}")
            raise

    def _rule_based_parse(self, text: str) -> Dict[str, List[str]]:
        """
        Apply rule-based parsing for structured elements
        
        Args:
            text: Input ticket text
            
        Returns:
            Dictionary of extracted elements
        """
        results = {}
        # First pass: Extract all fields with their values
        for field, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Clean up the matches - strip whitespace and remove empty strings
                cleaned_matches = [m.strip() for m in matches if m and m.strip()]
                if cleaned_matches:
                    if field in ['ticket_id', 'priority', 'status', 'created_date', 'closed_date', 
                               'assigned_to', 'category', 'title']:
                        # For metadata fields, just take the first match
                        results[field] = cleaned_matches[0]
                    else:
                        # For content fields, keep all matches
                        results[field] = cleaned_matches

        # Special handling for certain fields
        if 'description' in results:
            # Use description for problem statement if not already extracted
            if 'problem_statement' not in results:
                results['problem_statement'] = results['description']

        if 'technical_details' in results:
            # Look for error messages and stack traces in technical details
            error_pattern = re.compile(r'error:.*?(?=\n|$)', re.IGNORECASE)
            errors = error_pattern.findall(str(results['technical_details']))
            if errors:
                results['error_messages'] = errors

        return results

    def _llm_based_parse(self, text: str, section: str) -> str:
        """Use LLM to extract sections that need natural language understanding"""
        # Extract context
        title_match = self.patterns['title'].search(text)
        desc_match = self.patterns['description'].search(text)
        tech_match = self.patterns['technical_details'].search(text)
        res_match = self.patterns['resolution_notes'].search(text)
        
        # Clean and prepare context
        title = title_match.group(1) if title_match else 'Unknown'
        desc = desc_match.group(1) if desc_match else ''
        tech = tech_match.group(1) if tech_match else ''
        res = res_match.group(1) if res_match else ''

        # Build prompts with clear instruction focus
        prompts = {
            'problem_statement': 
                f"Ticket: {title}\nDescription: {desc}\n\n"
                "Extract the core IT support issue. What specific problem needs to be fixed?\n"
                "Be concise and focus on the technical problem.\n",

            'expected_behavior':
                f"Description: {desc}\n\n"
                "What is the expected normal behavior?\n"
                "Start with 'System should' or 'User should'.\n",

            'actual_behavior':
                f"Description: {desc}\nTechnical Details: {tech}\n\n"
                "What is currently happening? Describe the error or issue being experienced.\n"
                "Be specific about any error messages or symptoms.\n",

            'reproduction_steps':
                f"Description: {desc}\nTechnical Details: {tech}\n\n"
                "List the steps to reproduce this issue.\n"
                "Number each step. Be specific about actions taken.\n",

            'attempted_solutions':
                f"Technical Details: {tech}\nResolution Notes: {res}\n\n"
                "What troubleshooting or fixes have been attempted?\n"
                "List each solution tried and its outcome.\n"
        }

        prompt = prompts.get(section, f"Extract the {section} from: {desc}")

        try:
            # Generate response
            response = self.llm(
                prompt,
                max_new_tokens=150,
                num_return_sequences=1,
                do_sample=False
            )
            
            result = response[0]['generated_text'].strip()
            
            # Clean up response
            if not result or result.lower() in ['none', 'nan', 'n/a', 'unknown']:
                return None
                
            # Remove any prompt text that might have been echoed
            result = result.replace(prompt, '').strip()
            if not result:
                return None
                
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM parsing for {section}: {str(e)}")
            return None

    def _build_tree(self, parsed_data: Dict) -> TicketNode:
        """
        Build a tree structure from parsed ticket data
        
        Args:
            parsed_data: Dictionary of parsed ticket data
            
        Returns:
            Root node of the ticket tree
        """
        # Initialize root node
        root = TicketNode("ticket_root", {})

        # Create metadata node with scalar values
        metadata_content = {}
        for field in ['ticket_id', 'title', 'priority', 'status', 
                     'created_date', 'closed_date', 'assigned_to', 'category']:
            if field in parsed_data:
                value = parsed_data[field]
                if isinstance(value, list):
                    value = value[0] if value else None
                metadata_content[field] = str(value) if value else None

        metadata_node = TicketNode("metadata", metadata_content)
        root.children.append(metadata_node)

        # Create content node with descriptive fields
        content = {}
        for field in ['description', 'technical_details', 'steps_to_reproduce']:
            if field in parsed_data and parsed_data[field]:
                value = parsed_data[field]
                if isinstance(value, list):
                    value = '\n'.join(str(v) for v in value if v and str(v).lower() != 'nan')
                content[field] = str(value)
        content_node = TicketNode("content", content)
        root.children.append(content_node)

        # Create analysis node with basic problem statement from description
        analysis = {}
        if 'problem_statement' in parsed_data:
            value = parsed_data['problem_statement']
            if isinstance(value, list):
                value = '\n'.join(str(v) for v in value if v and str(v).lower() != 'nan')
            analysis['problem_statement'] = str(value)
        analysis_node = TicketNode("analysis", analysis)
        root.children.append(analysis_node)
        
        # Comment out LLM-based analysis fields
        """
        for field in ['expected_behavior', 'actual_behavior',
                     'reproduction_steps', 'attempted_solutions']:
            if field in parsed_data and parsed_data[field]:
                value = parsed_data[field]
                if isinstance(value, list):
                    value = '\n'.join(str(v) for v in value if v and str(v).lower() != 'nan')
                analysis[field] = str(value)
        """

        # Create supplementary node
        supplementary = {}
        for field in ['environment', 'related_tickets', 'resolution_notes']:
            if field in parsed_data and parsed_data[field]:
                value = parsed_data[field]
                if isinstance(value, list):
                    value = '\n'.join(str(v) for v in value if v and str(v).lower() != 'nan')
                supplementary[field] = str(value)
        supp_node = TicketNode("supplementary", supplementary)
        root.children.append(supp_node)

        return root

    def _detect_relationships(self, tickets: List[Dict]) -> Dict[str, List[str]]:
        """
        Detect relationships between tickets based on similarities in category, title and description
        
        Args:
            tickets: List of ticket dictionaries
            
        Returns:
            Dictionary mapping ticket IDs to lists of related ticket IDs
        """
        relationships = {}
        
        # Group tickets by category for initial filtering
        category_groups = {}
        for ticket in tickets:
            category = ticket.get('category', '').lower()
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(ticket)
        
        # For each ticket, find related tickets
        for ticket in tickets:
            ticket_id = ticket.get('ticket_id', '')
            if not ticket_id:
                continue
                
            relationships[ticket_id] = []
            title = ticket.get('title', '').lower()
            description = ' '.join(ticket.get('description', [''])).lower()
            category = ticket.get('category', '').lower()
            priority = ticket.get('priority', '').lower()
            
            # Check tickets in same and related categories
            related_categories = {category}
            if 'network' in category:
                related_categories.update(['vpn issue', 'wifi issue'])
            elif 'authentication' in category:
                related_categories.update(['access request', 'login issue'])
            
            for cat in related_categories:
                for other in category_groups.get(cat, []):
                    other_id = other.get('ticket_id', '')
                    if other_id and other_id != ticket_id:
                        other_title = other.get('title', '').lower() 
                        other_desc = ' '.join(other.get('description', [''])).lower()
                        other_priority = other.get('priority', '').lower()
                        
                        # Calculate similarity scores
                        # 1. Title similarity
                        title_words = set(title.split())
                        other_title_words = set(other_title.split())
                        title_similarity = len(title_words.intersection(other_title_words)) / max(len(title_words), len(other_title_words))
                        
                        # 2. Description similarity
                        desc_words = set(description.split())
                        other_desc_words = set(other_desc.split())
                        desc_similarity = len(desc_words.intersection(other_desc_words)) / max(len(desc_words), len(other_desc_words))
                        
                        # Ticket is related if:
                        # - Titles are very similar (similarity > 0.7)
                        # - Descriptions are very similar (similarity > 0.8)
                        # - Same category and priority with moderately similar descriptions (similarity > 0.5)
                        if (title_similarity > 0.7 or
                            desc_similarity > 0.8 or
                            (category == cat and priority == other_priority and desc_similarity > 0.5)):
                            relationships[ticket_id].append(other_id)
                        
        return relationships

    def _build_ticket_structure(self, parsed_data: Dict[str, Any], relationships: Dict[str, List[str]]) -> TicketNode:
        """
        Build hierarchical ticket structure based on template
        
        Args:
            parsed_data: Dictionary of parsed fields
            relationships: Dictionary mapping ticket IDs to related ticket IDs
            
        Returns:
            Root TicketNode containing full ticket structure
        """
        # Create root node
        root = TicketNode(section="ticket", content={}, children=[])
        
        # Create metadata node
        metadata = TicketNode(section="metadata", content={}, children=[])
        for field in self.template['ticket_structure']['metadata']:
            if field in parsed_data:
                metadata.content[field] = parsed_data[field]
        root.children.append(metadata)
        
        # Create content node
        content = TicketNode(section="content", content={}, children=[])
        
        # Add description section
        desc = TicketNode(section="description", content={}, children=[])
        for field in self.template['ticket_structure']['content']['description']:
            if field in parsed_data:
                desc.content[field] = parsed_data[field]
        content.children.append(desc)
        
        # Add technical details section
        tech = TicketNode(section="technical_details", content={}, children=[])
        for field in self.template['ticket_structure']['content']['technical_details']:
            if field in parsed_data:
                tech.content[field] = parsed_data[field]
        content.children.append(tech)
        
        # Add steps section
        steps = TicketNode(section="steps", content={}, children=[])
        for field in self.template['ticket_structure']['content']['steps']:
            if field in parsed_data:
                steps.content[field] = parsed_data[field]
        content.children.append(steps)
        
        # Add environment section
        env = TicketNode(section="environment", content={}, children=[])
        for field in self.template['ticket_structure']['content']['environment']:
            if field in parsed_data:
                env.content[field] = parsed_data[field]
        content.children.append(env)
        
        root.children.append(content)
        
        # Add relations
        relations = TicketNode(section="relations", content={}, children=[])
        ticket_id = parsed_data.get('ticket_id')
        if ticket_id and ticket_id in relationships:
            relations.content['related_tickets'] = relationships[ticket_id]
        else:
            relations.content['related_tickets'] = []
        root.children.append(relations)
        
        return root

    def visualize_tree(self, node: TicketNode, level: int = 0) -> dict:
        """
        Convert a TicketNode tree to a dictionary for visualization
        
        Args:
            node: TicketNode to convert
            level: Current indentation level
            
        Returns:
            Dictionary representation of the tree
        """
        result = {
            node.section: {
                "content": node.content
            }
        }
        
        if node.children:
            for child in node.children:
                child_dict = self.visualize_tree(child, level + 1)
                result[node.section].update(child_dict)
        
        if node.relations:
            result[node.section]["relations"] = node.relations
            
        return result

    def parse(self, ticket_text: str) -> TicketNode:
        """
        Parse a ticket using hybrid approach
        
        Args:
            ticket_text: Raw ticket text
            
        Returns:
            TicketNode representing the parsed ticket structure
        """
        try:
            logger.info("Starting ticket parsing...")
            logger.debug(f"Raw ticket text:\n{ticket_text}\n{'-'*50}")
            
            # Rule-based parsing for structured elements
            rule_based_results = self._rule_based_parse(ticket_text)
            logger.info("Rule-based parsing results:")
            for field, value in rule_based_results.items():
                logger.info(f"{field}: {value}")
            
            # Using rule_based_results directly without LLM processing
            parsed_data = rule_based_results.copy()
            
            # Comment out LLM-based parsing
            """
            llm_sections = [
                'problem_statement',
                'expected_behavior',
                'actual_behavior',
                'reproduction_steps',
                'attempted_solutions'
            ]
            
            logger.info("\nStartting LLM-based parsing...")
            for section in llm_sections:
                logger.info(f"Processing section: {section}")
                content = self._llm_based_parse(ticket_text, section)
                if content and content != 'None':
                    parsed_data[section] = content
                    logger.info(f"Extracted {section}:\n{content}\n")
                else:
                    logger.info(f"No content found for {section}")
            """
            
            # Build the tree structure
            logger.info("Building parse tree...")
            tree = self._build_tree(parsed_data)
            logger.info("Parsing complete!")
            return tree
            
        except Exception as e:
            logger.error(f"Error parsing ticket: {str(e)}")
            raise

    def parse(self, text: str, all_tickets: List[Dict] = None) -> Dict:
        """
        Parse a ticket text into structured format
        
        Args:
            text: Raw ticket text
            all_tickets: Optional list of all tickets for relationship detection
            
        Returns:
            Structured ticket data following template
        """
        # Get rule-based parsing results
        logger.info("Starting ticket parsing...")
        parsed_data = self._rule_based_parse(text)
        logger.info("Rule-based parsing results:")
        for key, value in parsed_data.items():
            logger.info(f"{key}: {value}")
            
        # Detect relationships if we have all tickets
        relationships = {}
        if all_tickets and 'ticket_id' in parsed_data:
            relationships = self._detect_relationships(all_tickets)
            
        # Build structured tree
        logger.info("Building parse tree...")
        ticket_tree = self._build_ticket_structure(parsed_data, relationships)
        logger.info("Parsing complete!")
        
        return ticket_tree

    def visualize_tree(self, node: TicketNode, level: int = 0) -> str:
        """
        Create a text visualization of the ticket tree
        
        Args:
            node: Current node to visualize
            level: Current indentation level
            
        Returns:
            String representation of the tree
        """
        indent = "  " * level
        result = [f"{indent}{node.section}:"]
        
        if node.content:
            for key, value in node.content.items():
                result.append(f"{indent}  {key}: {value}")
        
        if node.relations:
            result.append(f"{indent}Relations:")
            for rel_type, rel_value in node.relations.items():
                result.append(f"{indent}  {rel_type}: {rel_value}")
        
        for child in node.children:
            result.append(self.visualize_tree(child, level + 1))
            
        return "\n".join(result)
