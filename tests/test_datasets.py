from effirag.datasets import _parse_hotpot_record


def test_parse_hotpot_local_list_schema():
    rec = {
        "_id": "l1",
        "question": "What city is the capital of France?",
        "answer": "Paris",
        "context": [
            ["Paris", ["Paris is the capital of France."]],
            ["France", ["France is in Europe."]],
        ],
        "supporting_facts": [["Paris", 0]],
        "type": "bridge",
    }

    sample = _parse_hotpot_record(rec)

    assert sample.qid == "l1"
    assert len(sample.contexts) == 2
    assert sample.contexts[0].title == "Paris"
    assert sample.supporting_facts == [("Paris", 0)]


def test_parse_hotpot_hf_dict_schema():
    rec = {
        "_id": "h1",
        "question": "Who wrote Hamlet?",
        "answer": "William Shakespeare",
        "context": {
            "title": ["Hamlet", "Shakespeare"],
            "sentences": [
                ["Hamlet is a tragedy.", "It was written by William Shakespeare."],
                ["William Shakespeare was an English playwright."],
            ],
        },
        "supporting_facts": {
            "title": ["Hamlet"],
            "sent_id": [1],
        },
        "type": "bridge",
    }

    sample = _parse_hotpot_record(rec)

    assert sample.qid == "h1"
    assert len(sample.contexts) == 2
    assert sample.contexts[0].title == "Hamlet"
    assert sample.contexts[0].sentences[1] == "It was written by William Shakespeare."
    assert sample.supporting_facts == [("Hamlet", 1)]
