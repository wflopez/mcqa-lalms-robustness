import os
import json
import argparse
import time
from google import genai
from tqdm import tqdm
import copy


def rephrase_distractors(args):
    """
    Reads questions from an input file, rephrases distractors using Gemini,
    and writes the rephrased distractors to an output file.
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
    
    # Set rate limits based on model
    rate_limits = {
        "gemini-2.5-pro": {"rpm": 150, "delay": 0.4},
        "gemini-2.0-flash-exp": {"rpm": 1000, "delay": 0.06},
        "gemini-2.5-flash": {"rpm": 1000, "delay": 0.06},
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
    
    # Base prompt for rephrasing distractors
    base_prompt = """
    You are helping to rephrase distractors (incorrect answer choices) for a question-answering system.
    For each distractor, provide three rephrased versions that maintain the original meaning but use different wording.
    The rephrased distractors should remain plausible but incorrect, and should be clearly differentiated from the correct answer.
    Avoid including any additional information or context that is not present in the original distractor.
    """

    # Load the dataset
    dataset_file = os.path.join(args.dataset_folder, args.dataset_filename)
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file {dataset_file} does not exist.")
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    # Sample for testing
    #dataset = dataset[:8]
    
    if not dataset:
        raise ValueError("Dataset is empty. Please provide a valid dataset.")

    model_short_name = args.model_name.split("/")[-1]
    
    # Check if output files already exist for all three versions
    dataset_name = args.dataset_filename.replace('.json', '')
    
    for version_num in range(1, 4):
        if args.include_questions and args.include_answer:
            output_file = os.path.join(args.dataset_folder, f"rephrased-d-with-qa_{model_short_name}-{version_num}_{dataset_name}.json")
        elif args.include_answer and not args.include_questions:
            output_file = os.path.join(args.dataset_folder, f"rephrased-d-with-a_{model_short_name}-{version_num}_{dataset_name}.json")
        elif not args.include_answer and args.include_questions:
            output_file = os.path.join(args.dataset_folder, f"rephrased-d-with-q_{model_short_name}-{version_num}_{dataset_name}.json")
        else:
            output_file = os.path.join(args.dataset_folder, f"rephrased-d_{model_short_name}-{version_num}_{dataset_name}.json")

        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Please remove it or choose a different model/dataset.")
            return
    
    print(f"Rephrasing distractors in {dataset_file} using model {args.model_name} and will generate 3 versions")

    # Flatten the data: create one entry per distractor
    flattened_data = []
    for item in dataset:
        question = item['question']
        answer = item['answer']
        choices = item.get('choices', [])
        choices = [str(choice) for choice in choices if choice is not None]

        # Handle edge cases for specific answers
        if "middle aged adult" in answer:
            answer = answer.replace("middle aged adult", "Middle-aged adult")
        elif "elderly adult" in answer:
            answer = answer.replace("elderly adult", "Elderly adult")
        
        # Find distractors (choices that are not the correct answer)
        distractors = [choice for choice in choices if choice != answer]
        
        # Create one entry per distractor
        for distractor_idx, distractor in enumerate(distractors):
            flattened_data.append({
                'original_id': item['id'],
                'question': question,
                'answer': answer,
                'distractor': distractor,
                'distractor_idx': distractor_idx,
                'all_choices': choices
            })

    # Process distractors one by one
    all_rephrased = []
    
    print(f"Processing {len(flattened_data)} distractors with {request_delay:.2f}s delay between requests...")
    
    for i, item in enumerate(tqdm(flattened_data, desc="Rephrasing distractors")):
        # Build prompt based on options
        prompt = base_prompt
        
        if args.include_questions:
            prompt += f"\nQuestion: {item['question']}"
        if args.include_answer:
            prompt += f"\nCorrect Answer: {item['answer']}"
        
        prompt += f"\nAll choices: {item['all_choices']}"
        prompt += f"\nDistractor to rephrase: {item['distractor']}"

        # For the first distractor, print the prompt to check
        if i == 0:
            print(f"Prompt for distractor {item['original_id']}_distractor_{item['distractor_idx']}:\n{prompt}")

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
        sample_id = f"{item['original_id']}_distractor_{item['distractor_idx']}"
        
        try:
            response = make_request_with_retry(client, request)
            
            if not response or not response.text:
                raise ValueError("Empty response received")
                
            rephrased_data = json.loads(response.text.strip())
            rephrased_versions = rephrased_data["rephrased"]
            
            if len(rephrased_versions) != 3:
                raise ValueError(f"Expected 3 rephrased distractors, got {len(rephrased_versions)}")

            all_rephrased.append({
                'id': sample_id,
                'original': item['distractor'],
                'rephrased_versions': rephrased_versions
            })
            
            print(f"✓ Successfully processed distractor {sample_id}")
            
        except Exception as e:
            print(f"Error processing distractor {sample_id}: {e}")
            
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
                            'id': sample_id,
                            'original': item['distractor'],
                            'rephrased_versions': rephrased_versions
                        })
                        print(f"✓ Successfully extracted response despite error for distractor {sample_id}")
                        success_from_response = True
                except:
                    pass
            
            # If we couldn't extract a valid response, wait longer and use fallback
            if not success_from_response:
                print(f"Waiting 60 seconds before using fallback for distractor {sample_id}...")
                time.sleep(60)

                # Use original distractor as fallback
                original_d = item['distractor']
                all_rephrased.append({
                    'id': sample_id,
                    'original': original_d,
                    'rephrased_versions': [original_d, original_d, original_d]
                })
                print(f"✗ Used fallback for distractor {sample_id}")
        
        # Rate limiting delay (except for the last request)
        if i < len(flattened_data) - 1:
            time.sleep(request_delay)

    # Create a mapping from rephrased distractors for each version
    rephrased_maps = [{}, {}, {}]  # Three versions
    for item in all_rephrased:
        parts = item['id'].split('_distractor_')
        original_id = parts[0]
        distractor_idx = int(parts[1])
        
        for version_idx in range(3):
            if original_id not in rephrased_maps[version_idx]:
                rephrased_maps[version_idx][original_id] = {}
            rephrased_maps[version_idx][original_id][distractor_idx] = item['rephrased_versions'][version_idx]

    # Generate three output files
    for version_num in range(1, 4):
        # Determine output filename based on options
        if args.include_questions and args.include_answer:
            output_file = os.path.join(args.dataset_folder, f"rephrased-d-with-qa_{model_short_name}-{version_num}_{dataset_name}.json")
        elif args.include_answer and not args.include_questions:
            output_file = os.path.join(args.dataset_folder, f"rephrased-d-with-a_{model_short_name}-{version_num}_{dataset_name}.json")
        elif not args.include_answer and args.include_questions:
            output_file = os.path.join(args.dataset_folder, f"rephrased-d-with-q_{model_short_name}-{version_num}_{dataset_name}.json")
        else:
            output_file = os.path.join(args.dataset_folder, f"rephrased-d_{model_short_name}-{version_num}_{dataset_name}.json")

        # Generate new dataset with rephrased distractors for this version
        new_dataset = []
        rephrased_map = rephrased_maps[version_num - 1]
        
        for sample in dataset:
            sample_id = sample['id']
            answer = sample['answer']
            choices = sample['choices']

            # Create new choices list with rephrased distractors
            new_choices = copy.deepcopy(choices)
            distractor_counter = 0
            
            for choice_idx, choice in enumerate(choices):
                if choice != answer:  # This is a distractor
                    if sample_id in rephrased_map and distractor_counter in rephrased_map[sample_id]:
                        new_choices[choice_idx] = rephrased_map[sample_id][distractor_counter]
                    distractor_counter += 1
            
            sample_copy = copy.deepcopy(sample)
            sample_copy['choices'] = new_choices
            new_dataset.append(sample_copy)

        # Write the rephrased distractors to the output file
        with open(output_file, 'w') as f:
            json.dump(new_dataset, f, indent=4)
        
        print(f"Generated: {output_file}")

    print(f"Successfully generated 3 rephrased dataset files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rephrase distractors using Gemini")
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
    
    # Add answer within the prompt
    parser.add_argument(
        '--not_include_answer',
        dest='include_answer',
        action='store_false',
        default=True,
        help='Whether to include answer in the prompt'
    )

    # Add questions within the prompt
    parser.add_argument(
        '--not_include_questions',
        dest='include_questions',
        action='store_false',
        default=True,
        help='Whether to include questions in the prompt'
    )

    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        raise ValueError("Please set the GEMINI_API_KEY environment variable")

    print("Running with arguments:", vars(args))
    
    rephrase_distractors(args)
