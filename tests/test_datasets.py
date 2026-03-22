from effirag.datasets import (
    _parse_2wikimultihopqa_record,
    _parse_hotpot_record,
    _parse_musique_record,
    _parse_popqa_record,
)


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


def test_parse_musique_paragraph_schema():
    rec = {
        "id": "m1",
        "question": "When was Messi signed by Barcelona?",
        "answer": "February 2002",
        "paragraphs": [
            {
                "title": "Lionel Messi",
                "paragraph_text": "Messi was enrolled in the RFEF in February 2002.",
                "is_supporting": True,
            },
            {
                "title": "Barcelona",
                "paragraph_text": "Barcelona is a Spanish club.",
                "is_supporting": False,
            },
        ],
    }

    sample = _parse_musique_record(rec)

    assert sample.qid == "m1"
    assert sample.answer == "February 2002"
    assert len(sample.contexts) == 2
    assert sample.contexts[0].title == "Lionel Messi"
    assert sample.supporting_facts == [("Lionel Messi", 0)]


def test_parse_2wiki_hotpot_like_schema():
    rec = {
        "_id": "w1",
        "question": "When did Lothair II's mother die?",
        "answer": "20 March 851",
        "supporting_facts": [["Lothair II", 1], ["Ermengarde of Tours", 0]],
        "context": [
            ["Lothair II", ["Lothair II had parents.", "His mother was Ermengarde of Tours."]],
            ["Ermengarde of Tours", ["Ermengarde of Tours died on 20 March 851."]],
        ],
        "type": "bridge",
    }

    sample = _parse_2wikimultihopqa_record(rec)

    assert sample.qid == "w1"
    assert sample.answer == "20 March 851"
    assert len(sample.contexts) == 2
    assert ("Lothair II", 1) in sample.supporting_facts


def test_parse_popqa_possible_answers_string_and_paragraphs():
    rec = {
        "id": 4222362,
        "question": "What is George Rankin's occupation?",
        "possible_answers": '["politician", "political leader"]',
        "subj": "George Rankin",
        "prop": "occupation",
        "obj": "politician",
        "paragraphs": [
            {
                "title": "George Rankin",
                "text": "George Rankin was an Australian soldier and politician.",
                "is_supporting": True,
            }
        ],
    }

    sample = _parse_popqa_record(rec)

    assert sample.qid == "4222362"
    assert sample.answer == "politician"
    assert len(sample.contexts) == 1
    assert sample.contexts[0].title == "George Rankin"
    assert sample.supporting_facts == [("George Rankin", 0)]
