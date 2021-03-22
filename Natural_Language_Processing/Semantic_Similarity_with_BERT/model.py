import torch 
import transformers

class BertSemanticModel(torch.nn.Module):
    def __init__(self):
        super(BertSemanticModel, self).__init__() 
        
        self.no_grad = False
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
            
        self.bi_lstm = torch.nn.LSTM(input_size=self.bert.config.hidden_size, 
                                     hidden_size=64,
                                     bidirectional=True)
        
        self.linear = torch.nn.Linear(in_features=64*2*2, out_features=3) 
        self.dropout = torch.nn.Dropout(p=0.3)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        
        if self.no_grad:
            with torch.no_grad():
                embedding = self.bert(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
        else:
            embedding = self.bert(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
        
        # sequence_output (batch size x #token x hidden size) : (batch size x 128 x 768)        
        # pooled_output (batch size x  hidden size)           : (batch size x  768) CLS token에 linear mapping 후 tanh 결과
        sequence_output, pooled_output = embedding[0], embedding[1]
        
        # lstm_out (batch size x #token x hidden size)        : (batch size x 128 x 128)
        lstm_out, _ = self.bi_lstm(sequence_output)

        # gap_out (batch size x hidden size)                  : (batch size x 128)
        gap_out = lstm_out.mean(dim=1) # GAP
        
        # gmp_out (batch size x hidden size)                  : (batch size x 128)
        gmp_out, _ = lstm_out.max(dim=1) # GMP
           
        # out (batch size x hidden size)                      : (batch size x 256)
        out = torch.cat([gap_out, gmp_out], dim=1)
        out = self.dropout(out)
        
        # out (batch size x #class)                           : (batch size x 3)
        out = self.linear(out)
        
        return out