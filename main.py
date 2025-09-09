from scratchpad import scratch_pad_generation
from find_tool_call import final_tool_call
from find_tasks import final_task
from openai import OpenAI
import os, json, time
from dotenv import load_dotenv

# Import existing tau-bench logging infrastructure
import sys
sys.path.append('tau-bench')
from tau_bench.model_utils.api.logging import log_call, prep_for_json_serialization, log_files
from multiprocessing import Lock

load_dotenv()

base_url = "https://api.groq.com/openai/v1"
api_key = os.environ.get("GROQ_API_KEY")

client = OpenAI(base_url=base_url,api_key=api_key) 

# Setup logging using existing tau-bench infrastructure
def setup_srm_logging(log_file_path=None):
    """Setup SRM pipeline logging using existing tau-bench framework"""
    if log_file_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file_path = f"logs/srm_pipeline_{timestamp}.jsonl"
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Initialize tau-bench logging
    if log_file_path not in log_files:
        log_files[log_file_path] = Lock()
    
    return log_file_path

# Create a logging class that uses tau-bench's framework
class SRMPipeline:
    def __init__(self, client, log_file_path=None):
        self.client = client
        self._log_file = log_file_path or setup_srm_logging()
    
    @log_call  # Using existing tau-bench decorator
    def run_pipeline(self, messages, model_name, tools, request_id=None):
        if request_id:
            print(f"[SRM-PIPELINE] Starting pipeline for request: {request_id}")
        return self._run_srm_pipeline_internal(messages, model_name, tools)
    
    def _run_srm_pipeline_internal(self, messages, model_name, tools):
        t1 = time.time()
        print("Running SRM pipeline...")
        print("Step 1: Scratchpad Generation")
        scratchpad = scratch_pad_generation(self.client, messages, model_name, tools)
        print(f"Scratchpad generated in {time.time() - t1:.2f} seconds")

        print("Step 2: Unit Task Generation")
        t2 = time.time()
        verdict = ""
        for i in range(3): 
            unit_task,verdict = final_task(scratchpad, messages, self.client, model_name)
            if "done" in unit_task.lower() or "final unit task" in unit_task.lower():
                    return scratchpad,"User Request is Achieved. Reply as done", "No tool call needed"

            print(verdict)
            if verdict.strip().lower()== "yes":
                print(f"Prerequisites missing", unit_task)
                return scratchpad, unit_task, "No tool call needed"
            elif verdict.strip().lower()== "no" or i==2:
                print(f"Unit generated in {time.time() - t2:.2f} seconds")
                print(f"Final Unit Task: {unit_task}")

                print("Step 3: Tool Call Generation")
                t3 = time.time()
                tool_call,unit_task = final_tool_call(unit_task,messages,scratchpad, self.client, model_name,tools)
                print(f"Unit Task: {unit_task}")
                print(f"Tool Call: {tool_call}")
                print(f"Tool Call Generated in {time.time() - t3:.2f} seconds")
                print("SRM pipeline completed.")
                return scratchpad, unit_task, tool_call

# Global SRM pipeline instance with logging
srm_pipeline = SRMPipeline(client)

def run_srm_pipeline(client,messages,model_name,tools,request_id=None):
    """Main SRM pipeline function with automatic logging using existing tau-bench framework"""
    return srm_pipeline.run_pipeline(messages, model_name, tools, request_id)



