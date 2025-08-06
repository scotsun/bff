from utils.tokenizer_utils import BERT_SPECIAL_TOKENS


def pad_masking(tokens):
    padded_positions = tokens == BERT_SPECIAL_TOKENS["<PAD>"]
    return padded_positions
