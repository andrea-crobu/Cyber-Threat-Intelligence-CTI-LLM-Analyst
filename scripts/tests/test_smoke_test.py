from smoke_test import build_smoke_batch


def test_smoke_batch_holds_two_masked_examples():
    prepared = {
        "train": {
            "input_ids": [[7, 8, 9], [7, 8, 10], [7, 8, 11]],
            "labels": [[-100, 5, 1], [-100, 6, 1], [-100, 4, 1]],
        }
    }
    batch = build_smoke_batch(prepared)
    assert batch["input_ids"] == [[7, 8, 9], [7, 8, 10]]
    assert batch["labels"] == [[-100, 5, 1], [-100, 6, 1]]
    assert batch["labels"][0][0] == -100
