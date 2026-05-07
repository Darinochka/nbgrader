import pytest
from nbformat.v4 import new_markdown_cell, new_code_cell

from ...utils import is_llm_graded, extract_hidden_grading_instructions


class TestIsLlmGraded:

    def test_not_llm_graded(self):
        """Test that a regular cell is not LLM graded"""
        cell = new_markdown_cell("Hello")
        assert not is_llm_graded(cell)

    def test_llm_graded_cell(self):
        """Test that a cell with llm_graded flag is detected"""
        cell = new_markdown_cell("Hello")
        cell.metadata['nbgrader'] = {
            'grade': True,
            'solution': True,
            'llm_graded': True,
            'grade_id': 'test',
            'points': 10
        }
        assert is_llm_graded(cell)

    def test_no_metadata(self):
        """Test that a cell without metadata is not LLM graded"""
        cell = new_markdown_cell("Hello")
        assert not is_llm_graded(cell)

    def test_llm_graded_false(self):
        """Test that a cell with llm_graded=False is not detected"""
        cell = new_markdown_cell("Hello")
        cell.metadata['nbgrader'] = {
            'grade': True,
            'solution': True,
            'llm_graded': False,
            'grade_id': 'test',
            'points': 10
        }
        assert not is_llm_graded(cell)


class TestExtractHiddenGradingInstructions:

    def test_extract_hidden_content(self):
        """Test extracting hidden content between delimiters"""
        source = """This is the visible question.

###
This is hidden grading instructions.
Grade based on correctness.
###
"""
        visible, hidden = extract_hidden_grading_instructions(source)
        assert "This is the visible question." in visible
        assert "This is hidden grading instructions." in hidden
        assert "Grade based on correctness." in hidden
        assert "###" not in visible
        assert "###" not in hidden

    def test_no_hidden_content(self):
        """Test when there's no hidden content"""
        source = "This is just a regular question with no hidden content."
        visible, hidden = extract_hidden_grading_instructions(source)
        assert visible == source
        assert hidden == ""

    def test_custom_delimiters(self):
        """Test with custom delimiters"""
        source = """Question here.

BEGIN
Hidden instructions
END
"""
        visible, hidden = extract_hidden_grading_instructions(
            source, "BEGIN", "END"
        )
        assert "Question here." in visible
        assert "Hidden instructions" in hidden

    def test_nested_delimiters_error(self):
        """Test that nested delimiters raise an error"""
        source = """Question

###
First level
###
Second level
###
"""
        with pytest.raises(ValueError, match="Nested begin delimiter"):
            extract_hidden_grading_instructions(source)

    def test_unmatched_begin_delimiter_error(self):
        """Test that unmatched begin delimiter raises an error"""
        source = """Question

###
Hidden content without end
"""
        with pytest.raises(ValueError, match="without matching end"):
            extract_hidden_grading_instructions(source)

    def test_unmatched_end_delimiter_error(self):
        """Test that unmatched end delimiter raises an error"""
        source = """Question

###
"""
        with pytest.raises(ValueError, match="without matching begin"):
            extract_hidden_grading_instructions(source)

    def test_multiple_lines_hidden(self):
        """Test extracting multi-line hidden content"""
        source = """What is 2+2?

###
The answer should be 4.
Give full points for correct answer.
Give 0 points for incorrect answer.
###
"""
        visible, hidden = extract_hidden_grading_instructions(source)
        assert "What is 2+2?" in visible
        assert "The answer should be 4." in hidden
        assert "Give full points" in hidden
        assert "Give 0 points" in hidden