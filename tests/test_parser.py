import unittest
from pathlib import Path
from src.parsing.parser import TicketParser

class TestTicketParser(unittest.TestCase):
    def setUp(self):
        template_path = Path(__file__).parent.parent / 'src' / 'parsing' / 'T_template.yaml'
        self.parser = TicketParser(str(template_path))

    def test_rule_based_parsing(self):
        test_ticket = """
        Ticket #12345
        
        Error: Connection timeout when accessing database
        
        Code sample:
        ```python
        def connect():
            try:
                db.connect()
            except Exception as e:
                print(f"Failed to connect: {e}")
        ```
        
        Stack trace:
        Traceback (most recent call last):
          File "app.py", line 45, in connect
            db.connect()
          File "db.py", line 23, in connect
            raise TimeoutError("Database connection timeout")
        """
        
        parsed = self.parser._rule_based_parse(test_ticket)
        self.assertIn('ticket_id', parsed)
        self.assertIn('code_snippets', parsed)
        self.assertIn('error_messages', parsed)
        self.assertIn('stack_traces', parsed)

    def test_llm_based_parsing(self):
        test_ticket = """
        The application crashes when uploading files larger than 100MB.
        Expected: Files of any size should be uploaded successfully.
        Actual: Application throws out of memory error for large files.
        """
        
        problem = self.parser._llm_based_parse(test_ticket, 'problem_statement')
        self.assertIsNotNone(problem)
        self.assertNotEqual(problem, 'None')

    def test_full_parsing(self):
        test_ticket = """
        Ticket #12345
        Priority: High
        
        Problem:
        The application crashes when uploading files larger than 100MB.
        
        Expected behavior:
        Files of any size should be uploaded successfully.
        
        Actual behavior:
        Application throws out of memory error for large files.
        
        Code:
        ```python
        def upload_file(file):
            with open(file, 'rb') as f:
                data = f.read()  # This loads entire file into memory
        ```
        
        Error: OutOfMemoryError: Java heap space
        
        Related to: Ticket #12300
        """
        
        tree = self.parser.parse(test_ticket)
        
        # Validate tree structure
        self.assertEqual(tree.section, "ticket_root")
        self.assertTrue(any(child.section == "metadata" for child in tree.children))
        self.assertTrue(any(child.section == "content" for child in tree.children))
        
        # Print tree visualization for manual inspection
        print("\nParsed Ticket Tree:")
        print(self.parser.visualize_tree(tree))

if __name__ == '__main__':
    unittest.main()
