from backend.voice_text_cleanup import clean_text_for_speech, split_completed_sentences, split_speech_text


def test_clean_text_removes_citations_and_markdown_but_keeps_visible_words():
    answer = "## Key point\nThe **policy** applies. [Source 1: Page 2]\n- Read the [full text](https://example.com)."

    assert clean_text_for_speech(answer) == "Key point The policy applies. Read the full text."


def test_sentence_splitter_discards_citation_between_complete_sentences():
    sentences, remainder = split_completed_sentences(
        "First sentence. [Source 1: Page 2] Second sentence is still streaming"
    )

    assert sentences == ["First sentence."]
    assert remainder == "Second sentence is still streaming"


def test_unrecognized_brackets_are_not_stripped_as_citations():
    assert clean_text_for_speech("Use [draft] as the label.") == "Use [draft] as the label."


def test_long_speech_text_is_split_at_word_boundaries_within_provider_limit():
    segments = split_speech_text("word " * 50)

    assert len(segments) > 1
    assert all(len(segment) <= 200 for segment in segments)
    assert " ".join(segments) == "word " * 49 + "word"