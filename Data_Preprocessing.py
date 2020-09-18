def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    """
    Processes the tweet and outputs the features necessary for model training and inference
    """
    len_st = len(selected_text)
    idx0 = None
    idx1 = None
    # Finds the start and end indices of the span
    for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
        if tweet[ind: ind+len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    # Assign a positive label for the characters within the selected span (based on the start and end indices)
    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1
    
    # Encode the tweet using the set tokenizer (converted to ids corresponding to word pieces)
    tok_tweet = tokenizer.encode(tweet)
    # Save the ids and offsets (character level indices)
    # Ommit the first and last tokens, which should be the [CLS] and [SEP] tokens
    input_ids_orig = tok_tweet.ids[1:-1]
    tweet_offsets = tok_tweet.offsets[1:-1]
    
    # A token is considered "positive" when it has at least one character with a positive label
    # The indices of the "positive" tokens are stored in `target_idx`.
    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)
    
    # Store the indices of the first and last "positive" tokens
    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    sentiment_id = {
        'positive': 3893,
        'negative': 4997,
        'neutral': 8699
    }
    
    # Prepend the beginning of the tweet with the [CLS] token (101), and tweet sentiment, represented by the `sentiment_id`
    # The [SEP] token (102) is both prepended and appended to the tweet
    # You can find the indices in the vocab file: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
    input_ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids_orig + [102]
    token_type_ids = [0, 0, 0] + [1] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 3 + tweet_offsets + [(0, 0)]
    targets_start += 3
    targets_end += 3

    # Pad sequence if its length < `max_len`
    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
    
    # Return processed tweet as a dictionary
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets
    }
