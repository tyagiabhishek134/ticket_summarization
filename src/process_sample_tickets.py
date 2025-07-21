import pandas as pd
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from parsing.parser import TicketParser
import logging
import json
from typing import Dict, List
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_resolution_time(tickets_df: pd.DataFrame) -> Dict:
    """
    Analyze resolution times for tickets by category and priority
    
    Args:
        tickets_df: DataFrame containing ticket data
        
    Returns:
        Dictionary containing resolution time statistics
    """
    # Convert date columns to datetime
    tickets_df['Created Date'] = pd.to_datetime(tickets_df['Created Date'])
    tickets_df['Closed Date'] = pd.to_datetime(tickets_df['Closed Date'])
    
    # Calculate resolution time in hours for closed tickets
    closed_tickets = tickets_df[tickets_df['Status'] == 'Closed'].copy()
    closed_tickets['Resolution Time'] = (closed_tickets['Closed Date'] - closed_tickets['Created Date']).dt.total_seconds() / 3600
    
    analysis = {
        'overall': {
            'avg_resolution_time': closed_tickets['Resolution Time'].mean(),
            'median_resolution_time': closed_tickets['Resolution Time'].median(),
            'min_resolution_time': closed_tickets['Resolution Time'].min(),
            'max_resolution_time': closed_tickets['Resolution Time'].max(),
            'total_tickets': len(tickets_df),
            'closed_tickets': len(closed_tickets),
            'open_tickets': len(tickets_df) - len(closed_tickets)
        },
        'by_category': {},
        'by_priority': {},
        'long_running_tickets': []
    }
    
    # Analysis by category
    for category in closed_tickets['Category'].unique():
        cat_tickets = closed_tickets[closed_tickets['Category'] == category]
        if not cat_tickets.empty:
            analysis['by_category'][category] = {
                'avg_resolution_time': cat_tickets['Resolution Time'].mean(),
                'median_resolution_time': cat_tickets['Resolution Time'].median(),
                'ticket_count': len(cat_tickets)
            }
    
    # Analysis by priority
    for priority in closed_tickets['Priority'].unique():
        prio_tickets = closed_tickets[closed_tickets['Priority'] == priority]
        if not prio_tickets.empty:
            analysis['by_priority'][priority] = {
                'avg_resolution_time': prio_tickets['Resolution Time'].mean(),
                'median_resolution_time': prio_tickets['Resolution Time'].median(),
                'ticket_count': len(prio_tickets)
            }
    
    # Identify long-running tickets (> 2x category average)
    for _, ticket in closed_tickets.iterrows():
        category_avg = analysis['by_category'][ticket['Category']]['avg_resolution_time']
        if ticket['Resolution Time'] > 2 * category_avg:
            analysis['long_running_tickets'].append({
                'ticket_id': ticket['Ticket ID'],
                'category': ticket['Category'],
                'priority': ticket['Priority'],
                'resolution_time': ticket['Resolution Time'],
                'category_avg': category_avg
            })
    
    return analysis

def analyze_ticket_priority(tickets_df: pd.DataFrame) -> Dict:
    """
    Analyze and suggest ticket priorities based on keywords, categories, and patterns
    
    Args:
        tickets_df: DataFrame containing ticket data
        
    Returns:
        Dictionary containing priority analysis and suggestions
    """
    # Priority keywords mapping
    priority_keywords = {
        'Critical': [
            'crash', 'error', 'fail', 'down', 'urgent', 'emergency', 'broken',
            'security', 'breach', 'attack', 'phishing', 'malware', 'data loss',
            'unable to access', 'production', 'critical'
        ],
        'High': [
            'important', 'high', 'bug', 'issue', 'problem', 'blocked', 'cannot',
            'login', 'access', 'permission', 'slow', 'performance'
        ],
        'Medium': [
            'request', 'update', 'change', 'modify', 'add', 'install',
            'configure', 'setup', 'maintenance'
        ],
        'Low': [
            'question', 'inquiry', 'how to', 'documentation', 'information',
            'minor', 'cosmetic', 'enhancement'
        ]
    }
    
    # Category default priorities
    category_priorities = {
        'Security Alert': 'Critical',
        'Authentication': 'High',
        'Network Issue': 'High',
        'Performance': 'High',
        'Software Bug': 'High',
        'Access Request': 'Medium',
        'Email Issue': 'Medium',
        'VPN Issue': 'Medium',
        'Printer Issue': 'Low'
    }
    
    analysis = {
        'priority_distribution': {},
        'category_priority_mapping': {},
        'suggested_priority_changes': [],
        'priority_trends': {},
        'keyword_impact': defaultdict(int)
    }
    
    # Analyze current priority distribution
    priority_counts = tickets_df['Priority'].value_counts()
    analysis['priority_distribution'] = priority_counts.to_dict()
    
    # Analyze priorities by category
    for category in tickets_df['Category'].unique():
        cat_tickets = tickets_df[tickets_df['Category'] == category]
        analysis['category_priority_mapping'][category] = {
            'current_distribution': cat_tickets['Priority'].value_counts().to_dict(),
            'suggested_default': category_priorities.get(category, 'Medium')
        }
    
    # Analyze each ticket for potential priority adjustments
    for idx, ticket in tickets_df.iterrows():
        current_priority = ticket['Priority']
        category = ticket['Category']
        description = str(ticket['Description']).lower()
        title = str(ticket['Title']).lower()
        
        # Calculate priority score based on keywords
        priority_scores = {
            level: sum(1 for keyword in keywords 
                      if keyword.lower() in description or keyword.lower() in title)
            for level, keywords in priority_keywords.items()
        }
        
        # Track keyword impacts
        for level, keywords in priority_keywords.items():
            for keyword in keywords:
                if keyword.lower() in description or keyword.lower() in title:
                    analysis['keyword_impact'][keyword] += 1
        
        # Get suggested priority
        suggested_priority = max(priority_scores.items(), key=lambda x: x[1])[0]
        if priority_scores[suggested_priority] == 0:
            suggested_priority = category_priorities.get(category, 'Medium')
        
        # Record if suggestion differs from current
        if suggested_priority != current_priority:
            analysis['suggested_priority_changes'].append({
                'ticket_id': ticket['Ticket ID'],
                'current_priority': current_priority,
                'suggested_priority': suggested_priority,
                'category': category,
                'reason': f"Based on keywords and category patterns. Found {priority_scores[suggested_priority]} keywords matching {suggested_priority} priority."
            })
    
    # Analyze priority trends over time
    tickets_df['Created Date'] = pd.to_datetime(tickets_df['Created Date'])
    tickets_df['month'] = tickets_df['Created Date'].dt.strftime('%Y-%m')
    priority_trends = tickets_df.groupby(['month', 'Priority']).size().unstack(fill_value=0)
    analysis['priority_trends'] = priority_trends.to_dict()
    
    return analysis

def node_to_dict(node):
    """Convert a TicketNode to a dictionary"""
    if not node:
        return None
        
    result = {
        "section": node.section,
        "content": node.content.copy() if node.content else {}
    }
    
    if node.children:
        result["children"] = {child.section: node_to_dict(child) for child in node.children}
    
    if node.relations:
        result["relations"] = node.relations.copy()
        
    return result

def process_sample_tickets(excel_path: str, template_path: str, num_samples: int = 20) -> Dict:
    """
    Process sample tickets from Excel file and return parsed results with analysis
    
    Args:
        excel_path: Path to the Excel file
        template_path: Path to the YAML template
        num_samples: Number of samples to process
        
    Returns:
        Dictionary containing parsed tickets and analysis results
    """
    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        logger.info(f"Loaded {len(df)} tickets from Excel file")
        
        # Take first num_samples rows
        df = df.head(num_samples)
        
        # Perform resolution time analysis
        resolution_analysis = analyze_resolution_time(df)
        logger.info("Completed resolution time analysis")
        
        # Perform priority analysis
        priority_analysis = analyze_ticket_priority(df)
        logger.info("Completed priority analysis")
        
        # Initialize parser
        parser = TicketParser(template_path)
        
        # First pass: collect all tickets for relationship detection
        all_tickets = []
        for idx, row in df.iterrows():
            ticket_dict = {
                'ticket_id': row.get('Ticket ID', f'TCKT{idx:04d}'),
                'title': row.get('Title', 'No Title'),
                'description': [row.get('Description', 'No Description')],
                'priority': row.get('Priority', 'Unknown'),
                'category': row.get('Category', 'Uncategorized'),
            }
            all_tickets.append(ticket_dict)
        
        parsed_tickets = []
        for idx, row in df.iterrows():
            try:
                # Format ticket text with clear section markers
                ticket_text = (
                    f"Ticket ID: {row.get('Ticket ID', f'TCKT{idx:04d}')}\n"
                    f"Title: {row.get('Title', 'No Title')}\n"
                    f"Description: {row.get('Description', 'No Description')}\n"
                    f"Priority: {row.get('Priority', 'Unknown')}\n"
                    f"Status: {row.get('Status', 'Unknown')}\n"
                    f"Created Date: {row.get('Created Date', 'Unknown')}\n"
                    f"Closed Date: {row.get('Closed Date', 'None')}\n"
                    f"Assigned To: {row.get('Assigned To', 'Unassigned')}\n"
                    f"Category: {row.get('Category', 'Uncategorized')}\n"
                    f"Resolution Notes: {row.get('Resolution Notes', 'None')}"
                )
                
                # Parse ticket with relationship detection
                parsed_tree = parser.parse(ticket_text, all_tickets)
                
                # Convert original data to dictionary
                original_data = {}
                for k, v in row.to_dict().items():
                    if pd.isna(v):  # Handle NaN/NaT values
                        original_data[k] = None
                    elif isinstance(v, (pd.Timestamp, pd.DatetimeTZDtype)):
                        try:
                            original_data[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            original_data[k] = None
                    else:
                        original_data[k] = v
                         
                ticket_dict = {
                    'ticket_number': idx + 1,
                    'original_data': original_data,
                    'parsed_structure': parser.visualize_tree(parsed_tree)
                }
                
                parsed_tickets.append(ticket_dict)
                logger.info(f"Successfully parsed ticket {idx + 1}")
                
            except Exception as e:
                logger.error(f"Error processing ticket {idx + 1}: {str(e)}")
                continue
        
        # Combine parsing results with analysis
        result = {
            'tickets': parsed_tickets,
            'analysis': resolution_analysis,
            'priority_analysis': priority_analysis
        }
        
        # Save results
        output_path = Path(excel_path).parent / 'parsed_tickets_output.json'
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Processing complete. Results saved to {output_path}")
        
        # Log sample of analysis
        logger.info("\nPriority Analysis Summary:")
        logger.info(f"Current Distribution: {priority_analysis['priority_distribution']}")
        logger.info(f"Suggested Changes: {len(priority_analysis['suggested_priority_changes'])} tickets")
        return result
        
    except Exception as e:
        logger.error(f"Error loading or processing tickets: {str(e)}")
        raise

def process_tickets(input_file: str, output_file: str):
    """Process IT support tickets from Excel and save parsed results"""
    # Read tickets from Excel file
    tickets_df = pd.read_excel(input_file, nrows=500)
    logger.info(f"Loaded {len(tickets_df)} tickets from Excel file")
    
    # Initialize parser
    parser = TicketParser("src/parsing/T_template.yaml")
    
    # First pass - parse all tickets to get basic data for relationship detection
    all_tickets = []
    for idx, row in tickets_df.iterrows():
        # Format ticket data
        ticket_text = f"""
        Ticket ID: {row['Ticket ID']}
        Title: {row['Title']}
        Description: {row['Description']}
        Priority: {row['Priority']}
        Status: {row['Status']}
        Created Date: {row['Created Date']}
        Closed Date: {row['Closed Date']}
        Assigned To: {row['Assigned To']}
        Category: {row['Category']}
        Resolution Notes: {row['Resolution Notes']}
        """
        
        # Parse ticket to get basic data
        parsed_data = parser._rule_based_parse(ticket_text)
        all_tickets.append(parsed_data)
    
    # Second pass - process tickets with relationship detection
    processed_tickets = []
    for idx, row in tickets_df.iterrows():
        ticket_id = row['Ticket ID']
        category = row['Category']
        title = row['Title']
        desc = row['Description']
        
        # Find similar tickets
        related = []
        for other in tickets_df.iterrows():
            other_data = other[1]
            other_id = other_data['Ticket ID']
            if other_id != ticket_id:
                other_cat = other_data['Category']
                other_title = other_data['Title']
                other_desc = other_data['Description']
                
                if category == other_cat and (
                    title == other_title or 
                    desc == other_desc or
                    title in other_title or other_title in title or
                    desc in other_desc or other_desc in desc
                ):
                    related.append(other_id)
        
        # Format ticket data
        ticket_text = f"""
        Ticket ID: {row['Ticket ID']}
        Title: {row['Title']}
        Description: {row['Description']}
        Priority: {row['Priority']}
        Status: {row['Status']}
        Created Date: {row['Created Date']}
        Closed Date: {row['Closed Date']}
        Assigned To: {row['Assigned To']}
        Category: {row['Category']}
        Resolution Notes: {row['Resolution Notes']}
        """
        
        # Parse ticket metadata
        parsed_data = parser._rule_based_parse(ticket_text)
        
        # Build structured ticket with proper JSON format
        structured_data = {
            "type": "ticket",
            "metadata": {
                "ticket_id": parsed_data.get('ticket_id'),
                "title": parsed_data.get('title'),
                "priority": parsed_data.get('priority'),
                "status": parsed_data.get('status'),
                "created_date": parsed_data.get('created_date'),
                "closed_date": parsed_data.get('closed_date'),
                "assigned_to": parsed_data.get('assigned_to'),
                "category": parsed_data.get('category')
            },
            "content": {
                "description": {
                    "description": parsed_data.get('description', []),
                    "problem_statement": parsed_data.get('problem_statement', [])
                },
                "technical_details": parsed_data.get('technical_details', {}),
                "steps": {
                    "steps_to_reproduce": parsed_data.get('steps_to_reproduce', [])
                },
                "environment": parsed_data.get('environment', [])
            },
            "relations": {
                "related_tickets": related
            }
        }
        
        # Store results
        processed_tickets.append({
            "ticket_number": idx + 1,
            "original_data": row.to_dict(),
            "parsed_data": structured_data
        })
        
        logger.info(f"Successfully parsed ticket {idx + 1}")
        
    # Convert datetimes to strings for JSON serialization
    for ticket in processed_tickets:
        for key, value in ticket["original_data"].items():
            if pd.isna(value):
                ticket["original_data"][key] = None
            elif isinstance(value, pd.Timestamp):
                ticket["original_data"][key] = value.strftime("%Y-%m-%d %H:%M:%S")
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(processed_tickets, f, indent=2)

def main():
    """Main entry point"""
    try:
        # Setup paths
        base_path = Path(__file__).parent.parent
        excel_path = base_path / 'Data' / 'it_support_ticket_data.xlsx'
        template_path = base_path / 'src' / 'parsing' / 'T_template.yaml'
        output_path = base_path / 'Data' / 'parsed_tickets_output.json'
        
        # Process tickets
        results = process_sample_tickets(str(excel_path), str(template_path), num_samples=20)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Processing complete. Results saved to {output_path}")
        logger.info("Sample of processed tickets:")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
