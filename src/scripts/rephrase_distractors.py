import os
import json
import copy
import argparse
from tqdm import tqdm

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class TextCollator:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        sample_ids, input_ids, attention_masks = zip(*batch)

        max_length = max([ids.shape[1] for ids in input_ids])

        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(input_ids, attention_masks):
            if ids.shape[1] < max_length:
                padded_input_ids.append(
                    torch.cat([ids, torch.LongTensor([self.tokenizer.pad_token_id] * (max_length - ids.shape[1])).unsqueeze(0)], dim=1)
                )
                padded_attention_masks.append(
                    torch.cat([mask, torch.LongTensor([0] * (max_length - mask.shape[1])).unsqueeze(0)], dim=1)
                )
            else:
                padded_input_ids.append(ids)
                padded_attention_masks.append(mask)
        
        padded_input_ids = torch.cat(padded_input_ids, dim=0)
        padded_attention_masks = torch.cat(padded_attention_masks, dim=0).bool()
        
        out_dict = dict(
            sample_ids=sample_ids,
            input_ids=padded_input_ids,
            attention_mask=padded_attention_masks
        )
        return out_dict


class TextLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_file_root: str,
        max_tokens: int,
        dataset_name: str,
        seed: int,
        base_instruction: str,
        include_answer: bool = True,
        include_questions: bool = True
    ):
        dataset_file = os.path.join(dataset_file_root, dataset_name)
        contents = self._read_dataset_file(dataset_file)

        # Sample 8 elements for testing
        #contents = contents[:8]

        # Flatten the data: create one entry per distractor
        self.flattened_data = []
        for item in contents:
            question = item['question']
            answer = item['answer']
            choices = item.get('choices', [])
            choices = [str(choice) for choice in choices if choice is not None]
            
            # Find distractors (choices that are not the correct answer)
            distractors = [choice for choice in choices if choice != answer]
            
            # Create one entry per distractor
            for distractor_idx, distractor in enumerate(distractors):
                self.flattened_data.append({
                    'original_id': item['id'],
                    'question': question,
                    'answer': answer,
                    'distractor': distractor,
                    'distractor_idx': distractor_idx,
                    'all_choices': choices
                })

        self.base_instruction = base_instruction
        self.include_answer = include_answer
        self.include_questions = include_questions

        # Tokenizer
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.max_tokens = max_tokens

    def _read_dataset_file(self, dataset_file):
        with open(dataset_file) as f:
            contents = f.read()
        contents = json.loads(contents)
        return contents        

    def preprocess_string_for_eval(self, x):
        x = x.rstrip().lstrip()
        x = x.lower()
        return x

    def __getitem__(self, i):
        item = self.flattened_data[i]
        
        sample_id = f"{item['original_id']}_distractor_{item['distractor_idx']}"
        question = item['question']
        answer = item['answer']
        distractor = item['distractor']
        choices = item['all_choices']

        sample = f"{self.base_instruction.lower().strip()}"

        if self.include_questions:
            sample += f"\nQuestion: {question}"
        if self.include_answer:
            sample += f"\nCorrect Answer: {answer}"

        sample += f"\nAll choices: {choices}"
        sample += f"\nDistractor to rephrase: {distractor}"
        sample += f"\nRephrased Distractor:"

        text = self.tokenizer(
                sample,
                max_length=self.max_tokens,
                padding="longest",
                truncation="only_first",
                return_tensors="pt"
        )
  
        return (sample_id, text["input_ids"], text["attention_mask"])

    def __len__(self):
        return len(self.flattened_data)


def rephrase_questions(args):
    """
    Reads questions from an input file, rephrases them using a prompt, and writes the rephrased questions to an output file.
    """

    base_prompt = """
    You are helping to rephrase distractors (incorrect answer choices) for a question-answering system.
    For each distractor, provide a rephrased version that maintains the original meaning but uses different wording.
    The rephrased distractor should remain plausible but incorrect, and should be clearly differentiated from the correct answer.
    Avoid including any additional information or context that is not present in the original distractor.
    """
    
    model_name = args.model_name
    model_short_name = model_name.split("/")[-1]
    input_file = os.path.join(args.dataset_folder, args.dataset_filename)
    cache_dir = args.cache_dir

    if args.include_questions and args.include_answer:
        output_file = os.path.join(args.dataset_folder, f"rephrased-d-with-qa_{model_short_name}_{args.dataset_filename}")
    elif args.include_answer and not args.include_questions:
        output_file = os.path.join(args.dataset_folder, f"rephrased-d-with-a_{model_short_name}_{args.dataset_filename}")
    elif not args.include_answer and args.include_questions:
        output_file = os.path.join(args.dataset_folder, f"rephrased-d-with-q_{model_short_name}_{args.dataset_filename}")
    else:
        # not recommended: no context
        output_file = os.path.join(args.dataset_folder, f"rephrased-d_{model_short_name}_{args.dataset_filename}")

    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Please remove it or choose a different model/dataset.")
        return
    
    print(f"Rephrasing distractors in {input_file} using model {model_name} and saving to {output_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )

    # Download and cache the tokenizer and model locally
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        padding_side="left",
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map=device,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16
        )
    model.eval()

    # Load the dataset
    dataset = TextLoader(
        tokenizer=tokenizer,
        dataset_file_root=args.dataset_folder,
        max_tokens=args.max_tokens,
        dataset_name=args.dataset_filename,
        base_instruction=base_prompt,
        seed=args.seed,
        include_questions=args.include_questions,
        include_answer=args.include_answer
    )

    # Prepare the DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=TextCollator(tokenizer),
        num_workers=args.num_workers,
    )

    rephrased_distractors = []

    # Iterate through the DataLoader and rephrase questions with tqdm
    for batch in tqdm(dataloader, desc="Rephrasing distractors"):
        sample_ids = batch['sample_ids']
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Generate rephrased questions
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.max_tokens+100,  # Allow some extra tokens for the output
                max_new_tokens=100,
                early_stopping=True,
                do_sample=False,
                num_beams=4,
                num_return_sequences=1,
            )

        # Remove input prompts from the rephrased texts
        outputs = outputs[:, input_ids.shape[1]:]

        # Decode the generated outputs
        rephrased_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Process the rephrased texts and store them
        for rephrased_text, sample_id in zip(rephrased_texts, sample_ids):
            rephrased_distractors.append({
                "id": sample_id,
                "rephrased": rephrased_text.strip().lstrip(": ").split("\n")[0].strip()
            })

    # Generate a new JSON version replacing original distractors with rephrased ones
    # Read original dataset file
    with open(input_file, 'r') as f:
        original_questions = json.load(f)

    # Create a mapping from rephrased distractors
    rephrased_map = {}
    for item in rephrased_distractors:
        parts = item['id'].split('_distractor_')
        original_id = parts[0]
        distractor_idx = int(parts[1])
        
        if original_id not in rephrased_map:
            rephrased_map[original_id] = {}
        rephrased_map[original_id][distractor_idx] = item['rephrased']

    new_dataset = []
    
    for sample in original_questions:
        sample_id = sample['id']
        answer = sample['answer']
        choices = sample['choices']

        # Handle edge cases for specific answers
        if "middle aged adult" in answer:
            answer = answer.replace("middle aged adult", "Middle-aged adult")
        elif "elderly adult" in answer:
            answer = answer.replace("elderly adult", "Elderly adult")

        # Find distractors and their indices
        distractors = [(i, choice) for i, choice in enumerate(choices) if choice != answer]
        
        # Create new choices list with rephrased distractors
        new_choices = copy.deepcopy(choices)
        distractor_counter = 0
        
        for choice_idx, choice in enumerate(choices):
            if choice != answer:  # This is a distractor
                if sample_id in rephrased_map and distractor_counter in rephrased_map[sample_id]:
                    new_choices[choice_idx] = rephrased_map[sample_id][distractor_counter]
                distractor_counter += 1
        
        sample['choices'] = new_choices
        new_dataset.append(sample)

    # Write the rephrased distractors to the output file
    with open(output_file, 'w') as f:
        json.dump(new_dataset, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Rephrase distractors in a file.")
    parser.add_argument(
        '--dataset_folder',
        type=str,
        required=True,
        help='Path to the dataset file root'
        )
    parser.add_argument(
        '--dataset_filename',
        type=str,
        required=True,
        help='Name of the dataset'
        )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the pre-trained model to use for rephrasing.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='models/',
        help="Directory to cache the model",
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for dataloader'
        )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,
        help='Number of workers for dataloader'
        )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=512,
        help='Max tokens for dataloader'
        )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducibility'
        )

    # Add distractors within the prompt
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

    rephrase_questions(args)