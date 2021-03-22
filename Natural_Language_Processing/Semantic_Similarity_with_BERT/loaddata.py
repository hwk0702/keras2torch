import torch

class BertSemanticDataset(torch.utils.data.Dataset):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        targets: Array of labels.
        max_length: maximun length of sentence
        include_targets: boolean, whether to incude the labels.

    Returns:
        Dictionary keys : ['input_ids','attention_mask','token_type_ids','target']
        (or just [input_ids, attention_mask, token_type_ids] if include_targets=False)
    """
    
    def __init__(
        self,
        sentence_pairs,
        targets,
        tokenizer,
        max_length,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.targets = targets
        self.include_targets = include_targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        # Denotes the number of sentence pairs
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode_plus(
            self.sentence_pairs[idx][0],
            text_pair=self.sentence_pairs[idx][1],
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            truncation=True,
            return_tensors="pt",
        )
        
        if self.include_targets:
            return {
                'input_ids':encoded['input_ids'][0],
                'attention_mask':encoded['attention_mask'][0],
                'token_type_ids':encoded['token_type_ids'][0],
                'target': self.targets[idx]
            }
        else:
            return {
                'input_ids':encoded['input_ids'][0],
                'attention_mask':encoded['attention_mask'][0],
                'token_type_ids':encoded['token_type_ids'][0]
            }