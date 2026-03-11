from traitlets.config import Config

from ...preprocessors import ClearLLMAnswers, ClearLLMCriteria, LLMGrade


def test_shared_llm_delimiter_config_applies_to_all_preprocessors():
    c = Config()
    c.LLMDelimiterConfig.begin_question_delimiter = "BEGIN QUESTION CUSTOM"
    c.LLMDelimiterConfig.end_question_delimiter = "END QUESTION CUSTOM"
    c.LLMDelimiterConfig.begin_criteria_delimiter = "BEGIN CRITERIA CUSTOM"
    c.LLMDelimiterConfig.end_criteria_delimiter = "END CRITERIA CUSTOM"

    clear_answers = ClearLLMAnswers(config=c)
    clear_criteria = ClearLLMCriteria(config=c)
    llm_grade = LLMGrade(config=c)

    assert clear_answers.begin_question_delimiter == "BEGIN QUESTION CUSTOM"
    assert clear_answers.end_question_delimiter == "END QUESTION CUSTOM"
    assert clear_answers.begin_criteria_delimiter == "BEGIN CRITERIA CUSTOM"
    assert clear_answers.end_criteria_delimiter == "END CRITERIA CUSTOM"

    assert clear_criteria.begin_criteria_delimiter == "BEGIN CRITERIA CUSTOM"
    assert clear_criteria.end_criteria_delimiter == "END CRITERIA CUSTOM"

    assert llm_grade.begin_question_delimiter == "BEGIN QUESTION CUSTOM"
    assert llm_grade.end_question_delimiter == "END QUESTION CUSTOM"
    assert llm_grade.begin_criteria_delimiter == "BEGIN CRITERIA CUSTOM"
    assert llm_grade.end_criteria_delimiter == "END CRITERIA CUSTOM"


def test_per_class_delimiter_override_takes_precedence_over_shared():
    c = Config()
    c.LLMDelimiterConfig.begin_question_delimiter = "BEGIN QUESTION SHARED"
    c.LLMDelimiterConfig.end_question_delimiter = "END QUESTION SHARED"
    c.ClearLLMAnswers.begin_question_delimiter = "BEGIN QUESTION ANSWERS"

    clear_answers = ClearLLMAnswers(config=c)

    assert clear_answers.begin_question_delimiter == "BEGIN QUESTION ANSWERS"
    assert clear_answers.end_question_delimiter == "END QUESTION SHARED"


def test_default_delimiters_unchanged_without_shared_config():
    clear_answers = ClearLLMAnswers()
    clear_criteria = ClearLLMCriteria()
    llm_grade = LLMGrade()

    assert clear_answers.begin_question_delimiter == "BEGIN QUESTION_LLM"
    assert clear_answers.end_question_delimiter == "END QUESTION_LLM"
    assert clear_answers.begin_criteria_delimiter == "BEGIN CRITERIA_LLM"
    assert clear_answers.end_criteria_delimiter == "END CRITERIA_LLM"

    assert clear_criteria.begin_criteria_delimiter == "BEGIN CRITERIA_LLM"
    assert clear_criteria.end_criteria_delimiter == "END CRITERIA_LLM"

    assert llm_grade.begin_question_delimiter == "BEGIN QUESTION_LLM"
    assert llm_grade.end_question_delimiter == "END QUESTION_LLM"
    assert llm_grade.begin_criteria_delimiter == "BEGIN CRITERIA_LLM"
    assert llm_grade.end_criteria_delimiter == "END CRITERIA_LLM"
