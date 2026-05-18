from app.backends.jetson.paraformer_trt import decode_ids


def test_decode_ids_preserves_adjacent_repeated_tokens():
    tokens = ["<blank>", "<s>", "</s>", "今", "天", "气"]

    assert decode_ids([3, 4, 4, 5], tokens) == "今天天气"


def test_decode_ids_filters_special_tokens():
    tokens = ["<blank>", "<s>", "</s>", "你", "好"]

    assert decode_ids([1, 3, 0, 4, 2], tokens) == "你好"
