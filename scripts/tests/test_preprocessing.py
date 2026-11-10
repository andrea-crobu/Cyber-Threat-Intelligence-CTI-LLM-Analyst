import pandas as pd
import pytest

from cti_preprocessing import PROMPT_INTRO, PROMPT_OUTRO, CTIPreprocessor

THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"


def test_prompt_text_is_the_project_wording(stub_tokenizer):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    assert preprocessor.build_prompt_text("Report body.") == (
        "You are a skilled AI Agent capable of doing CTI Analysis.\n\nGiven this threat report:\n"
        "Report body."
        "\n\nYou will extract the main entities and their relations; finally, you will generate a diagnosis of the threat.\n"
    )
    assert preprocessor.build_prompt_text("x") == f"{PROMPT_INTRO}x{PROMPT_OUTRO}"


def test_prompt_is_wrapped_once_in_the_chat_template_with_thinking_disabled(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    prompts = preprocessor.format_input_prompt(report_frame)

    assert len(stub_tokenizer.template_calls) == len(report_frame)
    call = stub_tokenizer.template_calls[0]
    assert call["enable_thinking"] is False
    assert call["add_generation_prompt"] is True
    assert call["tokenize"] is False
    assert call["messages"] == [
        {"role": "user", "content": preprocessor.build_prompt_text(report_frame.loc[0, "text"])}
    ]

    prompt = prompts.loc[0, "prompt"]
    assert prompt.startswith("<|im_start|>user\n")
    assert prompt.endswith(f"<|im_start|>assistant\n{THINK_OPEN}\n\n{THINK_CLOSE}\n\n")
    assert prompt.count("<|im_start|>user") == 1


def test_entities_and_relations_are_formatted_like_the_project(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    row = preprocessor.process_entries_and_relations(report_frame.loc[0])
    assert row["entities"] == "NFlog (malware), PoisonIvy (malware), DragonOK (threat-actor)"
    assert row["relations"] == "NFlog to DragonOK (authored-by), PoisonIvy to DragonOK (authored-by)"


def test_empty_relation_list_yields_empty_string_not_nan(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    row = preprocessor.process_entries_and_relations(report_frame.loc[1])
    assert row["relations"] == ""
    assert row["entities"] == "Lazarus (threat-actor)"


def test_output_string_has_the_three_sections(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    processed = report_frame.apply(preprocessor.process_entries_and_relations, axis=1)
    output = preprocessor.format_output(processed).loc[0, "output"]
    assert output.startswith("\nEntities: NFlog (malware)")
    assert "\nRelations: NFlog to DragonOK (authored-by)" in output
    assert output.endswith("\nDiagnosis: DragonOK deploys malware.")


def test_training_inputs_append_the_masked_output_and_eos(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    inputs = preprocessor.preprocess_input_data(report_frame, for_training=True)
    labels = preprocessor.preprocess_output_data(report_frame)

    assert len(inputs) == len(labels) == len(report_frame)
    assert [len(a) for a in inputs] == [len(b) for b in labels]
    assert inputs[0][-1] == stub_tokenizer.eos_token_id
    assert labels[0][-1] == stub_tokenizer.eos_token_id


def test_labels_mask_exactly_the_prompt_tokens(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    prompts = preprocessor.format_input_prompt(report_frame)["prompt"]
    prompt_lengths = [len(stub_tokenizer(prompt).input_ids) for prompt in prompts]
    labels = preprocessor.preprocess_output_data(report_frame)

    for prompt_length, label in zip(prompt_lengths, labels):
        assert label[:prompt_length] == [-100] * prompt_length
        assert -100 not in label[prompt_length:]


def test_inference_inputs_are_prompt_only(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    inputs = preprocessor.preprocess_input_data(report_frame, for_training=False)
    prompt = preprocessor.format_input_prompt(report_frame).loc[0, "prompt"]
    assert inputs[0] == stub_tokenizer(prompt).input_ids
    assert inputs[0][-1] != stub_tokenizer.eos_token_id


def test_max_length_truncates_each_side_and_keeps_lengths_equal(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer, max_length=5)
    inputs = preprocessor.preprocess_input_data(report_frame, for_training=True)
    labels = preprocessor.preprocess_output_data(report_frame)
    assert [len(a) for a in inputs] == [len(b) for b in labels]
    assert all(call["max_length"] == 5 for call in stub_tokenizer.encode_calls)


def test_no_max_length_means_no_truncation(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    preprocessor.preprocess_input_data(report_frame)
    assert all(call["truncation"] is False for call in stub_tokenizer.encode_calls)


def test_missing_columns_raise_a_named_error(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    with pytest.raises(ValueError, match="diagnosis"):
        preprocessor.preprocess_output_data(report_frame.drop(columns=["diagnosis"]))


def test_decode_drops_masked_tokens(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    labels = preprocessor.preprocess_output_data(report_frame)
    decoded = preprocessor.decode(labels[0])
    assert "-100" not in decoded
    assert "Entities" in decoded


def test_malformed_literal_raises_a_named_error(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    broken = report_frame.copy()
    broken.loc[0, "entities"] = "not a list"
    with pytest.raises(ValueError, match="entities"):
        preprocessor.process_entries_and_relations(broken.loc[0])


def test_decode_can_keep_the_chat_framing(stub_tokenizer, report_frame):
    preprocessor = CTIPreprocessor(stub_tokenizer)
    prompt_ids = preprocessor.tokenize(preprocessor.format_input_prompt(report_frame).loc[0, "prompt"])

    assert "<|im_end|>" not in preprocessor.decode(prompt_ids)
    assert "<|im_end|>" in preprocessor.decode(prompt_ids, preserve_special_tokens=True)
