import sys
sys.path.insert(1, sys.path[0].replace(sys.path[0].split('/')[-1], ''))

import os
import json
import torch
import random
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from loader_mmau_text import get_mmau_text_dataloader


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(description="Generate Qwen outputs.")
    parser.add_argument('--dataset_folder', type=str, required=True, help='Path to the dataset file root')
    parser.add_argument('--dataset_filename', type=str, required=True, help='Name of the dataset')

    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--cache_dir', type=str, default='models/', help='Directory to cache the model')

    parser.add_argument('--output_filename', type=str, required=True, help='Output file name')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder to save the results')

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for dataloader')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--max_tokens', type=int, default=512, help='Max tokens for dataloader')
    
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set the seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set Hugging Face cache directory
    model_name = args.model_name
    cache_dir = args.cache_dir

    # Quantization configuration for 8-bit model loading
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side="left")
    #model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map=device)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map=device, quantization_config=quantization_config, torch_dtype=torch.bfloat16)

    # Ensure model is in evaluation mode
    model.eval()

    # Add special tokens for audio and padding
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<audio>", "<|endofchunk|>", "<|PAD_TOKEN|>"]}
    )

    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "<SEP>"})

    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Configuration to simulate Audio Flamingo 2
    # FIXME: Use gready/beam search for better results
    inference_kwargs = {
        #"do_sample": True,
        "do_sample": False,  # Set to True for sampling, False for greedy/beam search
        #"top_k": 30,
        #"top_p": 0.95,
        "num_beams": 1,
        "num_return_sequences": 1,
    }

    tokenizer.pad_token = None
    tokenizer.pad_token_id = None
    tokenizer.pad_token = "<|PAD_TOKEN|>"
    tokenizer.pad_token_id = tokenizer.encode("<|PAD_TOKEN|>")[-1]
    
    print("Tokenizer special tokens: ", tokenizer.special_tokens_map)
    print("Tokenizer pad token: ", tokenizer.pad_token)
    print("Tokenizer eos token: ", tokenizer.eos_token)
    print("Tokenizer sep token: ", tokenizer.sep_token)
    print("Tokenizer unk token: ", tokenizer.unk_token)
    

    # Base prompt
    base_instruction = """
        Answer the user's question without providing extra information, choice one of the options.
        Choices are in the format: (a) xxx. (b) yyy. (c) zzz. (d) uuu.
        Return only one of the options in the format: (a) xxx.
        Do not add any other text.
        """

    # Dataloader for the MMAU dataset
    mmau_dataloader = get_mmau_text_dataloader(
        dataset_file_root = args.dataset_folder,
        dataset_name=args.dataset_filename,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        base_instruction=base_instruction
    )
    
    # Store the decoded outputs in a list of dicts
    decoded_results = {}

    for batch in tqdm(mmau_dataloader):    
        sample_ids = batch["sample_ids"]
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Generate output (no gradients needed for inference)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                #pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=50,
                **inference_kwargs
            )
        
        # Decode generated tokens to text
        decoded_outputs = tokenizer.batch_decode(outputs)

        # Decode inputs also
        decoded_inputs = tokenizer.batch_decode(input_ids)
        
        # Print or process the outputs as needed: sample_ids and generated text
        for sample_id, decoded_output, decoded_input in zip(sample_ids, decoded_outputs, decoded_inputs):

            #question = decoded_output.split(tokenizer.sep_token)[0].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '')
            #qwen_output =  decoded_output.split('Answer:')[-1].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '')
            #qwen_output = qwen_output.split("\n")[0]

            """
            question = decoded_output.split(tokenizer.sep_token)[0].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '')
            qwen_output = decoded_output.split(tokenizer.sep_token)[-1].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '').replace('<|endoftext|>', '')
            qwen_output = qwen_output.split("\n")[0].replace("The answer is: ", "").replace("The answer is", "").replace("The answer:", "").replace("Answer:", "").replace("Answer is:", "").replace("Answer is", "").replace("Answer", "")
            """
            
            llm_output = decoded_output.replace(decoded_input, "").replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '').replace('<|endoftext|>', '').replace('<end_of_turn>', '').replace("\n", "")

            """
            print("\n" + "*"*100)
            print("Separator token: ", tokenizer.sep_token)
            print("Separator token ID: ", tokenizer.sep_token_id)
            print("EOS token: ", tokenizer.eos_token)

            print("."*100)
            print(f"Decoded Output: {decoded_output}")
            print("."*100)
            print(f"Decoded Split: {decoded_output.split(tokenizer.sep_token)}")

            print("\n" + "*"*100)
            print(f"Sample ID: {sample_id}")
            print(f"Question: {question}")
            print(f"Generated Output: {qwen_output}")
            """
            
            # Store the decoded output in a list of dicts
            #decoded_results.append({
            #    sample_id: llm_output
            #})

            decoded_results[sample_id] = {
                "prompt": decoded_input,
                "model_output": llm_output
            }

    # Save the decoded outputs to a file
    output_file = args.output_filename
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # Generate the output results
    llm_results = []

    for item in mmau_dataloader.dataset.data:
        sample_id = item["id"] 
        item["model_output"] = decoded_results[sample_id]["model_output"]
        item["prompt"] = decoded_results[sample_id]["prompt"]
        llm_results.append(item)

    # Save the updated dataset with model outputs
    with open(os.path.join(output_folder, output_file), 'w') as f:
        json.dump(llm_results, f, indent=4)
    
    print(f"Results saved to {os.path.join(output_folder, output_file)}")
    
