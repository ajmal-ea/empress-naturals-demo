"""
Progress tracking and UI for RAG tools.
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TaskID
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger("rag_tools")

class ProgressTracker:
    """
    Track progress of website scraping and display UI.
    """
    
    def __init__(self, refresh_rate: float = 0.5):
        """
        Initialize progress tracker.
        
        Args:
            refresh_rate: How often to refresh the display (in seconds)
        """
        self.console = Console()
        self.progress = None
        self.task_id = None
        self.refresh_rate = refresh_rate
        
        # Statistics
        self.total_urls = 0
        self.processed_urls = 0
        self.successful_urls = 0
        self.failed_urls = 0
        self.skipped_urls = 0
        self.http_errors = 0
        self.api_errors = 0
        
        # HTTP status tracking
        self.status_counts = {}  # Status code -> count
        
        # Timing
        self.start_time = None
        self.last_update_time = 0
        self.processing_times = []  # List of URL processing times in seconds
        
        # URL queue
        self.processing_queue = []  # URLs currently being processed
        self.last_completed = []  # Recently completed URLs with status
    
    def initialize(self, total_urls: int):
        """
        Initialize the progress bar with the total number of URLs to process.
        
        Args:
            total_urls: Total number of URLs to process
        """
        self.total_urls = total_urls
        self.processed_urls = 0
        self.successful_urls = 0
        self.failed_urls = 0
        self.skipped_urls = 0
        self.http_errors = 0
        self.api_errors = 0
        self.status_counts = {}
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.processing_times = []
        self.processing_queue = []
        self.last_completed = []
        
        # Create progress display
        self.progress = Progress(
            TextColumn("Processing websites"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            "•",
            TextColumn("{task.completed}/{task.total} processed"),
            "•", 
            TextColumn("{task.fields[success]} success"),
            "•",
            TextColumn("{task.fields[failed]} failed"),
            "•",
            TextColumn("{task.fields[skipped]} skipped"),
            TimeRemainingColumn(),
            refresh_per_second=1/self.refresh_rate  # Convert to refresh per second
        )
        
        self.progress.start()
        self.task_id = self.progress.add_task(
            "Processing", 
            total=total_urls,
            processed=self.processed_urls,
            success=self.successful_urls,
            failed=self.failed_urls,
            skipped=self.skipped_urls
        )
    
    def url_started(self, url: str):
        """
        Mark a URL as started processing.
        
        Args:
            url: URL being processed
        """
        self.processing_queue.append({
            "url": url,
            "start_time": time.time()
        })
        
        self._update_display()
    
    def url_completed(self, url: str, status: str, http_code: int = None, error_type: str = None):
        """
        Update progress when a URL is processed.
        
        Args:
            url: URL that was processed
            status: Status of the processing ('success', 'failed', or 'skipped')
            http_code: HTTP status code if applicable
            error_type: Type of error if applicable ('http', 'api', 'other')
        """
        current_time = time.time()
        
        # Find the URL in the processing queue
        processing_time = None
        for i, item in enumerate(self.processing_queue):
            if item["url"] == url:
                processing_time = current_time - item["start_time"]
                self.processing_times.append(processing_time)
                self.processing_queue.pop(i)
                break
        
        # Add to completed list
        self.last_completed.append({
            "url": url,
            "status": status,
            "http_code": http_code,
            "error_type": error_type,
            "processing_time": processing_time,
            "completed_at": current_time
        })
        
        # Keep only the last 5 completed
        if len(self.last_completed) > 5:
            self.last_completed.pop(0)
        
        # Update statistics
        self.processed_urls += 1
        
        if status == "success":
            self.successful_urls += 1
        elif status == "failed":
            self.failed_urls += 1
            if error_type == "http":
                self.http_errors += 1
            elif error_type == "api":
                self.api_errors += 1
        elif status == "skipped":
            self.skipped_urls += 1
        
        # Track HTTP status codes
        if http_code is not None:
            self.status_counts[http_code] = self.status_counts.get(http_code, 0) + 1
        
        # Update the progress display
        self._update_display()
    
    def _update_display(self):
        """Update the progress display with current statistics"""
        current_time = time.time()
        
        # Throttle updates to avoid excessive refreshing
        if current_time - self.last_update_time < self.refresh_rate:
            return
        
        try:
            self.progress.update(
                self.task_id, 
                advance=0,  # Don't advance automatically
                completed=self.processed_urls,  # Set absolute completion
                processed=self.processed_urls,
                success=self.successful_urls,
                failed=self.failed_urls,
                skipped=self.skipped_urls,
                refresh=True
            )
        except Exception as e:
            logger.error(f"Error updating progress display: {e}")
        
        self.last_update_time = current_time
    
    def finish(self) -> Dict[str, Any]:
        """
        Complete the progress tracking and show summary statistics.
        
        Returns:
            Dictionary of statistics
        """
        if self.progress:
            try:
                self.progress.stop()
            except Exception as e:
                logger.error(f"Error stopping progress: {e}")
        
        # Calculate statistics
        end_time = time.time()
        duration = max(0.001, end_time - self.start_time)  # Avoid division by zero
        
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        else:
            avg_processing_time = 0
        
        # URLs per second (avoid division by zero)
        urls_per_second = self.processed_urls / duration
        
        # Avoid division by zero for estimated completion time
        if urls_per_second > 0 and self.processed_urls < self.total_urls:
            estimated_completion_time = (self.total_urls - self.processed_urls) / urls_per_second
        else:
            estimated_completion_time = 0
            
        # Create summary table
        self.console.print("\n")
        
        summary_table = Table(title="Website Processing Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total URLs", str(self.total_urls))
        summary_table.add_row("Processed", str(self.processed_urls))
        summary_table.add_row("Successful", str(self.successful_urls))
        summary_table.add_row("Failed", str(self.failed_urls))
        summary_table.add_row("Skipped", str(self.skipped_urls))
        summary_table.add_row("HTTP Errors", str(self.http_errors))
        summary_table.add_row("API Errors", str(self.api_errors))
        summary_table.add_row("Duration", f"{duration:.2f} seconds")
        summary_table.add_row("Avg Time Per URL", f"{avg_processing_time:.2f} seconds")
        summary_table.add_row("Processing Rate", f"{urls_per_second:.2f} URLs/second")
        
        if estimated_completion_time > 0:
            etc = timedelta(seconds=estimated_completion_time)
            summary_table.add_row("Estimated Time to Completion", str(etc))
            
        self.console.print(summary_table)
        
        # HTTP status code breakdown if we have any
        if self.status_counts:
            status_table = Table(title="HTTP Status Codes")
            status_table.add_column("Status Code", style="cyan")
            status_table.add_column("Count", style="green")
            status_table.add_column("Description", style="yellow")
            
            status_descriptions = {
                200: "OK",
                201: "Created",
                301: "Moved Permanently",
                302: "Found",
                304: "Not Modified",
                400: "Bad Request",
                401: "Unauthorized",
                403: "Forbidden",
                404: "Not Found",
                408: "Request Timeout",
                429: "Too Many Requests",
                500: "Internal Server Error",
                502: "Bad Gateway",
                503: "Service Unavailable",
                504: "Gateway Timeout"
            }
            
            for status, count in sorted(self.status_counts.items()):
                description = status_descriptions.get(status, "Unknown")
                status_table.add_row(str(status), str(count), description)
            
            self.console.print("\n")
            self.console.print(status_table)
        
        # Return statistics
        return {
            "total_urls": self.total_urls,
            "processed_urls": self.processed_urls,
            "successful_urls": self.successful_urls,
            "failed_urls": self.failed_urls,
            "skipped_urls": self.skipped_urls,
            "http_errors": self.http_errors,
            "api_errors": self.api_errors,
            "duration": duration,
            "avg_processing_time": avg_processing_time,
            "urls_per_second": urls_per_second,
            "status_counts": self.status_counts
        } 