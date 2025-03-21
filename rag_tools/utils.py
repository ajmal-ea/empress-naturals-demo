"""
Utility functions for RAG tools.
"""
import re
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from dateutil import parser as date_parser
import os
import sys
from rich.console import Console
from rich.prompt import Confirm

# Configure logging
def setup_logging(log_file: str = "vector_storage.log", level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging with console and file handlers.
    
    Args:
        log_file: Path to the log file
        level: Logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger("rag_tools")
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def abbreviate_company_name(name: str) -> str:
    """
    Create an abbreviation from a company name by taking the first letter of each word.
    
    Args:
        name: Company name
        
    Returns:
        Abbreviation of the company name
    """
    # Remove common legal suffixes
    name = re.sub(r'\b(Inc|LLC|Ltd|Limited|Corp|Corporation|Private|Public|Co)\b', '', name, flags=re.IGNORECASE)
    
    # Split by spaces, hyphens, underscores, and handle camelCase
    words = re.findall(r'[A-Z][a-z]*|\b[a-z]+\b', name)
    
    # Take first letter of each word and convert to uppercase
    abbreviation = ''.join(word[0].upper() for word in words if word)
    
    # If abbreviation is too short (less than 2 chars), use more characters
    if len(abbreviation) < 2 and name:
        abbreviation = name[:3].upper()
        
    return abbreviation

def parse_date(date_string: Optional[str]) -> Optional[datetime]:
    """
    Parse a date string into a datetime object.
    
    Args:
        date_string: Date string to parse
        
    Returns:
        Datetime object or None if parsing fails
    """
    if not date_string:
        return None
        
    try:
        return date_parser.parse(date_string)
    except Exception:
        return None

def check_environment_variables(required_vars: list) -> bool:
    """
    Check if required environment variables are set.
    
    Args:
        required_vars: List of required environment variable names
        
    Returns:
        True if all variables are set, False otherwise
    """
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment.")
        return False
    
    return True

def get_sql_file_path(company_abbr: str) -> str:
    """
    Get the path to the SQL file for database setup.
    
    Args:
        company_abbr: Company abbreviation
        
    Returns:
        Path to the SQL file
    """
    return f"{company_abbr.lower()}_db_setup.sql"

def prompt_for_sql_execution(sql_file_path: str) -> bool:
    """
    Prompt the user to execute SQL and wait for confirmation.
    
    Args:
        sql_file_path: Path to the SQL file
        
    Returns:
        True if user confirms execution, False otherwise
    """
    console = Console()
    
    console.print("\n[bold yellow]IMPORTANT DATABASE SETUP REQUIRED[/bold yellow]")
    console.print("[bold yellow]--------------------------------[/bold yellow]")
    console.print(f"Supabase does not allow table creation via the API. You need to manually run the SQL script.")
    console.print("\n[bold]Steps to follow:[/bold]")
    console.print("1. Open your [link=https://app.supabase.com]Supabase dashboard[/link]")
    console.print("2. Navigate to the SQL Editor section")
    console.print(f"3. Copy the SQL from the file: [bold cyan]{os.path.abspath(sql_file_path)}[/bold cyan]")
    console.print("4. Paste and execute the SQL in your Supabase SQL Editor")
    console.print("5. Return to this terminal and confirm completion")
    
    with open(sql_file_path, 'r') as f:
        sql_content = f.read()
    
    console.print("\n[bold]SQL to execute:[/bold]")
    console.print(f"[dim]{sql_content}[/dim]")
    
    console.print("\n[bold yellow]After executing the SQL in Supabase:[/bold yellow]")
    
    try:
        confirmation = Confirm.ask("Have you executed the SQL in your Supabase SQL Editor?")
        if confirmation:
            console.print("\n[bold green]Thank you! Continuing with the process...[/bold green]\n")
            return True
        else:
            console.print("[bold red]Please execute the SQL before continuing.[/bold red]")
            console.print(f"The SQL script is located at: [cyan]{os.path.abspath(sql_file_path)}[/cyan]")
            return False
    except KeyboardInterrupt:
        console.print("\n[bold red]Operation cancelled by user.[/bold red]")
        return False 