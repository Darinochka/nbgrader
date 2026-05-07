from traitlets.config.loader import Config


_DELIMITER_KEYS = (
    "begin_question_delimiter",
    "end_question_delimiter",
    "begin_criteria_delimiter",
    "end_criteria_delimiter",
)


def merge_shared_llm_delimiter_config(cfg: Config, class_name: str) -> None:
    """Merge shared LLM delimiter config into a specific class section.

    Class-specific config values keep precedence over shared values.
    """
    if "LLMDelimiterConfig" not in cfg:
        return

    shared_cfg = cfg.LLMDelimiterConfig
    class_cfg = cfg[class_name]

    for key in _DELIMITER_KEYS:
        if key in shared_cfg and key not in class_cfg:
            class_cfg[key] = shared_cfg[key]
