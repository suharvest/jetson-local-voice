from app.core.language import detect_zh_en, normalize_auto_language


def test_detect_zh_en_defaults_to_english_for_latin_text():
    assert detect_zh_en("Hello, good morning.") == "en"


def test_detect_zh_en_anchors_mixed_text_to_chinese():
    assert detect_zh_en("你好，OpenVoiceStream") == "zh"


def test_detect_zh_en_normalizes_explicit_language():
    assert detect_zh_en("hello", "Chinese") == "zh"
    assert detect_zh_en("你好", "en-US") == "en"


def test_normalize_auto_language():
    assert normalize_auto_language(None) is None
    assert normalize_auto_language("auto") is None
    assert normalize_auto_language("zh") == "zh"
