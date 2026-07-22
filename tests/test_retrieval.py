from retrieval import tokenize


def test_tokenize_is_accent_insensitive():
    assert tokenize("numéro") == tokenize("numero")


def test_tokenize_removes_french_stopwords():
    tokens = tokenize("le numéro de la création")
    for stopword in ("le", "de", "la"):
        assert stopword not in tokens


def test_tokenize_stems_words():
    # Snowball reduces inflected forms to a shared root
    assert tokenize("créations") == tokenize("création")


def test_tokenize_lowercases():
    assert tokenize("SIRET") == tokenize("siret")
