import os
import json
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
        include_gt: bool = False,
        include_distractors: bool = False
    ):
        dataset_file = os.path.join(dataset_file_root, dataset_name)
        contents = self._read_dataset_file(dataset_file)

        self.data = contents
        self.base_instruction = base_instruction

        # Tokenizer
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.max_tokens = max_tokens
        self.include_gt = include_gt
        self.include_distractors = include_distractors

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

        item = self.data[i]
        sample_id = item['id']
        question = item['question']
        answer = item['answer']

        if self.include_distractors:
            choices = item.get('choices', [])
            choices = [str(item) for item in choices if item is not None] # Remove none types and cast to string
            sample = f"{self.base_instruction.lower().strip()}\nQuestion: {question}\nChoices: {', '.join(choices)}\nRephrased Question:"
        elif self.include_gt:
            sample = f"{self.base_instruction.lower().strip()}\nOriginal Question: {question}\nAnswer: {answer}\nRephrased Question:"
        else:
            sample = f"{self.base_instruction.lower().strip()}\nOriginal Question: {question}\nRephrased Question:"

        text = self.tokenizer(
                sample,
                max_length=self.max_tokens,
                padding="longest",
                truncation="only_first",
                return_tensors="pt"
        )
  
        return (sample_id, text["input_ids"], text["attention_mask"])

    def __len__(self):
        return len(self.data)


def rephrase_questions(args):
    """
    Reads questions from an input file, rephrases them using a prompt, and writes the rephrased questions to an output file.
    """

    base_prompt = """
    You are helping to rephrase questions for a question-answering system.
    For each question, provide a rephrased version that maintains the original meaning but uses different wording.
    The rephrased question should be clear and concise, suitable for a user to understand without losing the context of the original question.
    Avoid including any additional information or context that is not present in the original question.
    """
    
    model_name = args.model_name
    model_short_name = model_name.split("/")[-1]
    input_file = os.path.join(args.dataset_folder, args.dataset_filename)
    cache_dir = args.cache_dir

    if args.include_distractors:
        output_file = os.path.join(args.dataset_folder, f"rephrased-q-with-d_{model_short_name}_{args.dataset_filename}")
    elif args.include_gt:
        output_file = os.path.join(args.dataset_folder, f"rephrased-q-with-a_{model_short_name}_{args.dataset_filename}")
    else:
        output_file = os.path.join(args.dataset_folder, f"rephrased-q_{model_short_name}_{args.dataset_filename}")

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
        include_gt=args.include_gt,
        include_distractors=args.include_distractors
    )

    # Prepare the DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=TextCollator(tokenizer),
        num_workers=args.num_workers,
    )

    rephrased_questions = []

    # Iterate through the DataLoader and rephrase questions with tqdm
    for batch in tqdm(dataloader, desc="Rephrasing questions"):
        sample_ids = batch['sample_ids']
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Generate rephrased questions
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.max_tokens,  # Allow some extra tokens for the output
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
            rephrased_questions.append({
                "id": sample_id,
                "rephrased": rephrased_text.strip().split("\n")[0].strip()  # Take the first line as the rephrased question
            })

    # Generate a new JSON version replacing original questions with rephrased ones
    # Read original dataset file
    with open(input_file, 'r') as f:
        original_questions = json.load(f)

    new_dataset = []
    
    for sample in original_questions:
        sample_id = sample['id']
        # Find the rephrased question for this sample_id
        rephrased_sample = next((q for q in rephrased_questions if q['id'] == sample_id), None)
        if rephrased_sample:
            sample['question'] = rephrased_sample['rephrased']
        new_dataset.append(sample)
    
    # Write the rephrased questions to the output file
    with open(output_file, 'w') as f:
        json.dump(new_dataset, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Rephrase questions in a file.")
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

    # Add ground truth within the prompt
    parser.add_argument(
        '--not_include_gt',
        dest='include_gt',
        action='store_false',
        default=True,
        help='Whether to include ground truth in the prompt'
        )

    # Add distractors within the prompt
    parser.add_argument(
        '--not_include_distractors',
        dest='include_distractors',
        action='store_false',
        default=True,
        help='Whether to include distractors in the prompt'
        )

    args = parser.parse_args()

    rephrase_questions(args)
