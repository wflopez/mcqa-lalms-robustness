import os
import json
import argparse
import time
from google import genai
from tqdm import tqdm


def rephrase_questions(args):
    """
    Reads questions from an input file, rephrases them using Gemini,
    and generates three output files with different rephrased versions.
    """
    
    def make_request_with_retry(client, request, max_retries=3):
        """Make a single request with retry logic for rate limiting."""
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=request["model"],
                    contents=request["contents"],
                    config=request["config"]
                )
                return response
                    
            except Exception as e:
                error_str = str(e)
                if "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str:
                    if attempt < max_retries - 1:
                        sleep_time = 60  # Wait 60 seconds for server overload
                        print(f"Server overloaded (503). Sleeping for {sleep_time} seconds before retry {attempt + 2}/{max_retries}")
                        time.sleep(sleep_time)
                    else:
                        print(f"Max retries reached for server overload. Request failed.")
                        raise e
                elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        sleep_time = 30  # Wait 30 seconds for rate limit
                        print(f"Rate limit hit. Sleeping for {sleep_time} seconds before retry {attempt + 2}/{max_retries}")
                        time.sleep(sleep_time)
                    else:
                        print(f"Max retries reached for rate limit. Request failed.")
                        raise e
                else:
                    raise e
        return None
    
    # Initialize Gemini client
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    if not client:
        raise ValueError(f"Failed to initialize the model {args.model_name}. Please check the model name and API key.")
    
    # Set rate limits based on model (updated limits)
    rate_limits = {
        "gemini-2.5-pro": {"rpm": 150, "delay": 0.4},        # 150 RPM = wait 0.4 seconds
        "gemini-2.0-flash-exp": {"rpm": 1000, "delay": 0.06}, # 1000 RPM = wait 0.06 seconds
        "gemini-2.5-flash": {"rpm": 1000, "delay": 0.06},     # 1000 RPM = wait 0.06 seconds  
    }
    
    # Get rate limit for current model
    model_key = args.model_name.lower()
    if model_key in rate_limits:
        request_delay = rate_limits[model_key]["delay"]
        rpm_limit = rate_limits[model_key]["rpm"]
    else:
        # Default to most conservative limit
        request_delay = 0.4
        rpm_limit = 150
        print(f"Warning: Unknown model {args.model_name}, using conservative rate limit of {rpm_limit} RPM")
    
    print(f"Using model: {args.model_name}")
    print(f"Rate limit: {rpm_limit} requests/minute (waiting {request_delay}s between requests)")
    
    # Base prompt for rephrasing individual questions
    base_prompt = """
    You are helping to rephrase a question for a question-answering system.
    I will provide a single question. Provide three rephrased versions that maintain the original meaning but use different wording.
    The rephrasing should be clear and concise, suitable for a user to understand without losing the context of the original question.
    Avoid including any additional information or context that is not present in the original question.
    """

    # Load the dataset
    dataset_file = os.path.join(args.dataset_folder, args.dataset_filename)
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file {dataset_file} does not exist.")
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    
    if not dataset:
        raise ValueError("Dataset is empty. Please provide a valid dataset.")

    model_short_name = args.model_name.split("/")[-1]
    
    # Prepare prompt based on options
    if args.include_distractors:
        prompt = base_prompt + "\nConsider the distractors when rephrasing."
    elif args.include_gt:
        prompt = base_prompt + "\nConsider the ground truth answer when rephrasing."
    else:
        prompt = base_prompt

    # Process questions one by one
    all_rephrased = []
    
    print(f"Processing {len(dataset)} questions with {request_delay:.2f}s delay between requests...")
    
    for i, item in enumerate(tqdm(dataset, desc="Rephrasing questions")):
        # Format prompt based on options
        if args.include_distractors:
            prompt += f"\nConsider the distractors: {item['choices']}"
        elif args.include_gt and 'answer' in item:
            prompt += f"\nConsider the ground truth answer: {item['answer']}"
        
        prompt += f"\nOriginal question: {item['question']}"
        
        # Create request
        request = {
            "model": args.model_name,
            "contents": prompt,
            "config": genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "rephrased": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3
                        }
                    },
                    "required": ["rephrased"]
                }
            )
        }
        
        response = None
        try:
            response = make_request_with_retry(client, request)
            
            if not response or not response.text:
                raise ValueError("Empty response received")
                
            rephrased_data = json.loads(response.text.strip())
            rephrased_versions = rephrased_data["rephrased"]
            
            if len(rephrased_versions) != 3:
                raise ValueError(f"Expected 3 rephrased questions, got {len(rephrased_versions)}")
            
            all_rephrased.append({
                'id': item['id'],
                'original_question': item['question'],
                'rephrased_versions': rephrased_versions
            })
            
            print(f"✓ Successfully processed question {item['id']}")
            
        except Exception as e:
            print(f"Error processing question {item['id']}: {e}")
            
            # Check if we have a valid response despite the error
            success_from_response = False
            if response and hasattr(response, 'text') and response.text:
                print(f"Response text: {response.text}")
                try:
                    # Try to parse the response even if there was an error
                    rephrased_data = json.loads(response.text.strip())
                    rephrased_versions = rephrased_data["rephrased"]
                    
                    if len(rephrased_versions) == 3:
                        all_rephrased.append({
                            'id': item['id'],
                            'original_question': item['question'],
                            'rephrased_versions': rephrased_versions
                        })
                        print(f"✓ Successfully extracted response despite error for question {item['id']}")
                        success_from_response = True
                except:
                    pass
            
            # If we couldn't extract a valid response, wait longer and try fallback
            if not success_from_response:
                print(f"Waiting 60 seconds before using fallback for question {item['id']}...")
                time.sleep(60)
                
                # Use original question as fallback
                original_q = item['question']
                all_rephrased.append({
                    'id': item['id'],
                    'original_question': original_q,
                    'rephrased_versions': [original_q, original_q, original_q]
                })
                print(f"✗ Used fallback for question {item['id']}")
        
        # Rate limiting delay (except for the last request)
        if i < len(dataset) - 1:
            time.sleep(request_delay)
            

    # Generate three output files
    dataset_name = args.dataset_filename.replace('.json', '')
    
    for version_num in range(1, 4):
        base_filename = f"rephrased-q-with-da_{model_short_name}-{version_num}_{dataset_name}"
        output_file = os.path.join(args.dataset_folder, f"{base_filename}.json")
        
        # Create new dataset with the specific rephrased version
        new_dataset = []
        for original_item, rephrased_item in zip(dataset, all_rephrased):
            new_item = original_item.copy()
            new_item['question'] = rephrased_item['rephrased_versions'][version_num - 1]
            new_dataset.append(new_item)
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(new_dataset, f, indent=4)
        
        print(f"Generated: {output_file}")

    print(f"Successfully generated 3 rephrased dataset files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rephrase questions using Gemini Flash 2.5")
    parser.add_argument(
        '--dataset_folder',
        type=str,
        required=True,
        help='Path to the dataset folder'
    )
    parser.add_argument(
        '--dataset_filename',
        type=str,
        required=True,
        help='Name of the dataset file'
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.0-flash-exp",
        help="Name of the Gemini model to use for rephrasing"
    )
    
    # Ground truth and distractors options
    parser.add_argument(
        '--include_gt',
        action='store_true',
        help='Include ground truth in the rephrasing prompt'
    )
    parser.add_argument(
        '--include_distractors',
        action='store_true',
        help='Include distractors in the rephrasing prompt'
    )

    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        raise ValueError("Please set the GEMINI_API_KEY environment variable")
    
    rephrase_questions(args)
