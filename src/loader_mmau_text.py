import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable the tokenizer parallelism warning

import json
import random
import itertools

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer


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

def get_mmau_text_dataloader(
    dataset_file_root,
    dataset_name,
    tokenizer,
    max_tokens,
    seed,
    batch_size,
    num_workers,
    base_instruction,
    perm_idx=None,
    open_ended=False,
    ):
    
    test_set = MMAUText(
        dataset_file_root=dataset_file_root,
        tokenizer=tokenizer, 
        max_tokens=max_tokens,
        dataset_name=dataset_name,
        seed=seed,
        base_instruction=base_instruction,
        perm_idx=perm_idx,
        open_ended=open_ended
    )
    
    # single GPU
    dataloader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=TextCollator(tokenizer), 
        num_workers=num_workers
    )

    return dataloader


class MMAUText(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_file_root: str,
        max_tokens: int,
        dataset_name: str = 'mmau-test-mini.json',
        seed: int = 0,
        base_instruction: str = "Answer the user's question without providing extra information, choice one of the options.",
        perm_idx: int = None,
        open_ended = False
    ):
        dataset_file = os.path.join(dataset_file_root, dataset_name)
        contents = self._read_dataset_file(dataset_file)

        # Set random seed for reproducibility
        random.seed(seed)

        self.data = contents
        self.base_instruction = base_instruction
        self.open_ended = open_ended

        # Tokenizer
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        # self.tokenizer.padding_side = "right" # AF2 implementation has this value
        self.max_tokens = max_tokens

        # Permutation index for choices
        self.perm_idx = perm_idx

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
        # Get the id, question, choices, and answer
        sample_id = item['id']
        question = item['question']
        choices = item['choices'].copy()  # Make a copy to avoid modifying the original list

        # Put questions and choices in the same format as the training data
        # Question? (A) xxx. (B) yyy. (C) zzz. (D) uuu.

        if self.perm_idx is None:

            # Shuffle the choices
            random.shuffle(choices)

            if self.open_ended or len(choices) == 0:
                text_prompt = question
            
            else:
                if len(choices) == 1:
                    text_prompt = f"{question} (A) {choices[0]}."
                elif len(choices) == 2:
                    text_prompt = f"{question} (A) {choices[0]}. (B) {choices[1]}."
                elif len(choices) == 3:
                    text_prompt = f"{question} (A) {choices[0]}. (B) {choices[1]}. (C) {choices[2]}."
                elif len(choices) == 4:
                    text_prompt = f"{question} (A) {choices[0]}. (B) {choices[1]}. (C) {choices[2]}. (D) {choices[3]}."
                elif len(choices) == 5:
                    text_prompt = f"{question} (A) {choices[0]}. (B) {choices[1]}. (C) {choices[2]}. (D) {choices[3]}. (E) {choices[4]}."
                elif len(choices) == 6:
                    text_prompt = f"{question} (A) {choices[0]}. (B) {choices[1]}. (C) {choices[2]}. (D) {choices[3]}. (E) {choices[4]}. (F) {choices[5]}."
                elif len(choices) == 7:
                    text_prompt = f"{question} (A) {choices[0]}. (B) {choices[1]}. (C) {choices[2]}. (D) {choices[3]}. (E) {choices[4]}. (F) {choices[5]}. (G) {choices[6]}."
                elif len(choices) == 8:
                    text_prompt = f"{question} (A) {choices[0]}. (B) {choices[1]}. (C) {choices[2]}. (D) {choices[3]}. (E) {choices[4]}. (F) {choices[5]}. (G) {choices[6]}. (H) {choices[7]}."
                elif len(choices) == 9:
                    text_prompt = f"{question} (A) {choices[0]}. (B) {choices[1]}. (C) {choices[2]}. (D) {choices[3]}. (E) {choices[4]}. (F) {choices[5]}. (G) {choices[6]}. (H) {choices[7]}. (I) {choices[8]}."
                elif len(choices) == 10:
                    text_prompt = f"{question} (A) {choices[0]}. (B) {choices[1]}. (C) {choices[2]}. (D) {choices[3]}. (E) {choices[4]}. (F) {choices[5]}. (G) {choices[6]}. (H) {choices[7]}. (I) {choices[8]}. (J) {choices[9]}."
                elif len(choices) == 11:
                    text_prompt = f"{question} (A) {choices[0]}. (B) {choices[1]}. (C) {choices[2]}. (D) {choices[3]}. (E) {choices[4]}. (F) {choices[5]}. (G) {choices[6]}. (H) {choices[7]}. (I) {choices[8]}. (J) {choices[9]}. (K) {choices[10]}."
                else:
                    raise ValueError(f"Unexpected number of choices: {len(choices)}")
        else:

            # When there are more that 4 choices: truncate the choices to 4 including the correct answer
            if len(choices) > 4:
                # Get the correct answer
                answer = item['answer']
                # Remove correct answer from the choices list
                choices.remove(answer)
                # Randomly sample 3 choices from the remaining choices and add the correct answer
                choices = [answer] + random.sample(choices, 3)
            
            # Calulate all possible permutations of the choices
            permutations = list(itertools.permutations(choices))
            
            # Select the permutation based on the perm_idx argument
            if len(permutations) == 2:
                # Some question only have 2 choices -> 2 possible permutations
                # For index bigger than 1, we will take the second permutation
                idx_to_use = self.perm_idx
                if (self.perm_idx > 1):
                    idx_to_use = -1
                choices = list(permutations[idx_to_use])
                text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}"
            elif len(permutations) == 24:
                choices = list(permutations[self.perm_idx])
                text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}"
            else:
                raise ValueError(f"Unexpected number of permutations: {len(permutations)}. Expected 24 permutations for 4 choices.")
        
        #text_output = self.preprocess_string_for_eval(str(item['output']).lower())


        sample = f"{self.base_instruction.lower().strip()} question: {text_prompt.lower().strip()}{self.tokenizer.sep_token}. Answer:"
           
        #print("Input text to the model: ", sample)

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
