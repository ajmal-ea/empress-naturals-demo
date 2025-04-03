from arize.phoenix.evals import (
    generate_eval_config,
    run_evals,
    trace_to_span_dataframe,
    render_evals,
)
from arize.phoenix.evals.eval_functions import (
    response_factuality,
    response_completeness,
    response_relevance,
    response_toxicity,
    response_political_bias,
    response_helpfulness,
)
import pandas as pd
import json
import os
import logging

logger = logging.getLogger(__name__)

class PhoenixEvaluator:
    """
    Handles evaluation of Phoenix traces to assess LLM performance
    """
    
    def __init__(self, trace_file=None):
        """Initialize the evaluator with the path to Phoenix traces"""
        self.trace_file = trace_file or os.getenv("PHOENIX_TRACE_FILE", "ea_chatbot_traces.jsonl")
        logger.info(f"Phoenix evaluator initialized with trace file: {self.trace_file}")
    
    def load_traces(self):
        """Load traces from the trace file"""
        try:
            # Convert traces to a DataFrame
            df = trace_to_span_dataframe(self.trace_file)
            
            # Filter for LLM call spans
            llm_calls = df[df["span_name"] == "llm_call"]
            
            return llm_calls
        except Exception as e:
            logger.error(f"Error loading traces for evaluation: {str(e)}")
            return pd.DataFrame()
    
    def run_standard_evals(self):
        """Run a standard set of evaluations on the traces"""
        try:
            # Create the evaluation configuration
            eval_config = generate_eval_config([
                response_factuality,
                response_relevance,
                response_completeness,
                response_toxicity,
                response_political_bias,
                response_helpfulness,
            ])
            
            # Run the evaluations
            results = run_evals(
                trace_path=self.trace_file, 
                eval_config=eval_config,
                output_path="evaluation_results.jsonl"
            )
            
            return results
        except Exception as e:
            logger.error(f"Error running evaluations: {str(e)}")
            return None
    
    def run_custom_eval(self, custom_eval_function, config=None):
        """Run a custom evaluation function on the traces"""
        try:
            # Create custom evaluation configuration
            eval_config = generate_eval_config([custom_eval_function], config)
            
            # Run the evaluation
            results = run_evals(
                trace_path=self.trace_file, 
                eval_config=eval_config,
                output_path="custom_evaluation_results.jsonl"
            )
            
            return results
        except Exception as e:
            logger.error(f"Error running custom evaluation: {str(e)}")
            return None
    
    def generate_eval_report(self, results_path="evaluation_results.jsonl"):
        """Generate a report from evaluation results"""
        try:
            # Load results from the file
            with open(results_path, 'r') as f:
                results = [json.loads(line) for line in f]
            
            # Create a summary report
            summary = {}
            for result in results:
                eval_name = result.get("eval_name", "unknown")
                score = result.get("score")
                
                if eval_name not in summary:
                    summary[eval_name] = {"total": 0, "count": 0}
                
                if score is not None:
                    summary[eval_name]["total"] += score
                    summary[eval_name]["count"] += 1
            
            # Calculate averages
            for eval_name, data in summary.items():
                if data["count"] > 0:
                    data["average"] = data["total"] / data["count"]
                else:
                    data["average"] = None
            
            return summary
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            return {}

# Create a singleton instance
phoenix_evaluator = PhoenixEvaluator() 