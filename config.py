class config:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 16
    EPOCHS = 5
    BERT_PATH = "../input/bert-base-uncased/"
    MODEL_PATH = "model.bin"
    TRAINING_FILE = "../input/tweet-train-folds/train_folds.csv"
    TOKENIZER = tokenizers.BertWordPieceTokenizer(
        f"{BERT_PATH}/vocab.txt", 
        lowercase=True
    )
