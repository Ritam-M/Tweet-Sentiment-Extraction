class TweetModel(transformers.BertPreTrainedModel):
    """
    Model class that combines a pretrained bert model with a linear later
    """
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        # Load the pretrained BERT model
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, config=conf)
        # Set 10% dropout to be applied to the BERT backbone's output
        self.drop_out = nn.Dropout(0.1)
        # 768 is the dimensionality of bert-base-uncased's hidden representations
        # Multiplied by 2 since the forward pass concatenates the last two hidden representation layers
        # The output will have two dimensions ("start_logits", and "end_logits")
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        # Return the hidden states from the BERT backbone
        _, _, out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        ) # bert_layers x bs x SL x (768)

        # Concatenate the last two hidden states
        # This is done since experiments have shown that just getting the last layer
        # gives out vectors that may be too taylored to the original BERT training objectives (MLM + NSP)
        # Sample explanation: https://bert-as-service.readthedocs.io/en/latest/section/faq.html#why-not-the-last-hidden-layer-why-second-to-last
        out = torch.cat((out[-1], out[-2]), dim=-1) # bs x SL x (768 * 2)
        # Apply 10% dropout to the last 2 hidden states
        out = self.drop_out(out) # bs x SL x (768 * 2)
        # The "dropped out" hidden vectors are now fed into the linear layer to output two scores
        logits = self.l0(out) # bs x SL x 2

        # Splits the tensor into start_logits and end_logits
        # (bs x SL x 2) -> (bs x SL x 1), (bs x SL x 1)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1) # (bs x SL)
        end_logits = end_logits.squeeze(-1) # (bs x SL)

        return start_logits, end_logits
