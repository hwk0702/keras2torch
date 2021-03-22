import torch 
from transformers import BertModel
import string
import re

class QABert(torch.nn.Module):
    def __init__(self):
        super(QABert, self).__init__()
        # Bert encoder
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # start token layer
        self.linear_start = torch.nn.Linear(in_features=self.bert.config.hidden_size, 
                                            out_features=1,
                                            bias=False)
        # end token layer
        self.linear_end = torch.nn.Linear(in_features=self.bert.config.hidden_size, 
                                          out_features=1,
                                          bias=False)
        
    def forward(self, 
                input_ids,
                token_type_ids,
                attention_mask):
        embedding = self.bert(input_ids,
                              token_type_ids=token_type_ids,
                              attention_mask=attention_mask)
        
        start_output = self.linear_start(embedding[0]).squeeze()
        end_output = self.linear_end(embedding[0]).squeeze()
        
        return start_output, end_output


def normalize_text(text):
    text = text.lower()

    # Remove punctuations
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    # Remove articles
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)

    # Remove extra white space
    text = " ".join(text.split())
    return text

class ExactMatch(object):
    def __init__(self, squad_examples=None):
        self.squad_examples = [_ for _ in squad_examples if _.skip == False]
        
    def evaluate(self, start_preds, end_preds):
        # ExactMatch
        count = 0

        for idx, (start, end) in enumerate(zip(start_preds, end_preds)):
            squad_eg = self.squad_examples[idx]
            pred_ans, true_ans = self._inference(start=start,
                                                end=end,
                                                squad_example=squad_eg)
            if (pred_ans is None) or (true_ans is None):
                continue
                
            if pred_ans in true_ans:
                count += 1
            
        acc = count / len(start_preds)
        print(f"Exact Match Score={acc:.2%}")
        
        
    def _inference(self, start, end, squad_example):
        return self.inference(start, end, squad_example)
    
    
    @staticmethod
    def inference(start, end, squad_example):
        offsets = squad_example.context_token_to_char
        
        # if answer start token index larger than offset length, then return None
        if start >= len(offsets):
            return None, None
        
        pred_char_start = offsets[start][0]

        if end < len(offsets):
            pred_char_end = offsets[end][1]
            pred_ans = squad_example.context[pred_char_start:pred_char_end]
        else:
            pred_ans = squad_example.context[pred_char_start:]

        normalized_pred_ans = normalize_text(pred_ans)
        normalized_true_ans = [normalize_text(_) for _ in squad_example.all_answers]
        
        return normalized_pred_ans, normalized_true_ans