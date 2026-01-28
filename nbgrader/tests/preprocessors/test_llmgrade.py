import pytest
from unittest.mock import Mock, patch, MagicMock
from nbformat.v4 import new_notebook, new_markdown_cell

from ...preprocessors import SaveCells, LLMGrade, SaveAutoGrades
from ...api import Gradebook
from ...utils import compute_checksum
from .base import BaseTestPreprocessor
from .. import create_grade_and_solution_cell


@pytest.fixture
def preprocessors():
    return (SaveCells(), LLMGrade(), SaveAutoGrades())


@pytest.fixture
def gradebook(request, db):
    gb = Gradebook(db)
    gb.add_assignment("ps0")
    gb.add_student("bar")

    def fin():
        gb.close()
    request.addfinalizer(fin)

    return gb


@pytest.fixture
def resources(db):
    return {
        "nbgrader": {
            "db_url": db,
            "assignment": "ps0",
            "notebook": "test",
            "student": "bar"
        }
    }


class TestLLMGrade(BaseTestPreprocessor):

    def test_skip_non_llm_cells(self, preprocessors, gradebook, resources):
        """Test that non-LLM cells are skipped"""
        cell = create_grade_and_solution_cell("hello", "markdown", "foo", 1)
        nb = new_notebook()
        nb.cells.append(cell)
        preprocessors[0].preprocess(nb, resources)
        gradebook.add_submission("ps0", "bar")
        
        # Should not call LLM API
        with patch.object(preprocessors[1], '_call_llm_api') as mock_api:
            preprocessors[1].preprocess(nb, resources)
            mock_api.assert_not_called()

    def test_skip_without_api_key(self, preprocessors, gradebook, resources):
        """Test that LLM grading is skipped when API key is not set"""
        cell = create_grade_and_solution_cell("hello", "markdown", "foo", 1)
        cell.metadata.nbgrader['llm_graded'] = True
        nb = new_notebook()
        nb.cells.append(cell)
        preprocessors[0].preprocess(nb, resources)
        gradebook.add_submission("ps0", "bar")
        
        # Set API key to empty
        preprocessors[1].llm_api_key = ""
        
        # Should not call LLM API
        with patch.object(preprocessors[1], '_call_llm_api') as mock_api:
            preprocessors[1].preprocess(nb, resources)
            mock_api.assert_not_called()

    @patch('nbgrader.preprocessors.llmgrade.OPENAI_AVAILABLE', True)
    @patch('nbgrader.preprocessors.llmgrade.REQUESTS_AVAILABLE', False)
    def test_llm_grade_success(self, preprocessors, gradebook, resources):
        """Test successful LLM grading"""
        # Create source cell with hidden instructions
        source_cell = create_grade_and_solution_cell(
            """What is 2+2?

###
The answer should be 4.
Give full points for correct answer.
###""",
            "markdown", "foo", 10
        )
        source_cell.metadata.nbgrader['llm_graded'] = True
        
        # Create student cell
        student_cell = create_grade_and_solution_cell("2+2 = 4", "markdown", "foo", 10)
        student_cell.metadata.nbgrader['llm_graded'] = True
        
        nb = new_notebook()
        nb.cells.append(source_cell)
        preprocessors[0].preprocess(nb, resources)  # SaveCells
        
        # Replace with student cell
        nb.cells[0] = student_cell
        gradebook.add_submission("ps0", "bar")
        
        # Mock OpenAI API response
        preprocessors[1].llm_api_key = "test-key"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "10"
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('nbgrader.preprocessors.llmgrade.OpenAI', return_value=mock_client):
            preprocessors[1].preprocess(nb, resources)  # LLMGrade
            preprocessors[2].preprocess(nb, resources)  # SaveAutoGrades
        
        # Check that grade was saved
        grade_cell = gradebook.find_grade("foo", "test", "ps0", "bar")
        assert grade_cell.auto_score == 10.0
        assert not grade_cell.needs_manual_grade

    @patch('nbgrader.preprocessors.llmgrade.OPENAI_AVAILABLE', True)
    @patch('nbgrader.preprocessors.llmgrade.REQUESTS_AVAILABLE', False)
    def test_llm_grade_partial_score(self, preprocessors, gradebook, resources):
        """Test LLM grading with partial score"""
        source_cell = create_grade_and_solution_cell(
            """Explain the concept.

###
Give 5 points for mentioning key concept.
Give 0 points if no mention.
###""",
            "markdown", "foo", 10
        )
        source_cell.metadata.nbgrader['llm_graded'] = True
        
        student_cell = create_grade_and_solution_cell("Partial answer", "markdown", "foo", 10)
        student_cell.metadata.nbgrader['llm_graded'] = True
        
        nb = new_notebook()
        nb.cells.append(source_cell)
        preprocessors[0].preprocess(nb, resources)
        
        nb.cells[0] = student_cell
        gradebook.add_submission("ps0", "bar")
        
        preprocessors[1].llm_api_key = "test-key"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "5"
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('nbgrader.preprocessors.llmgrade.OpenAI', return_value=mock_client):
            preprocessors[1].preprocess(nb, resources)
            preprocessors[2].preprocess(nb, resources)
        
        grade_cell = gradebook.find_grade("foo", "test", "ps0", "bar")
        assert grade_cell.auto_score == 5.0

    @patch('nbgrader.preprocessors.llmgrade.OPENAI_AVAILABLE', True)
    @patch('nbgrader.preprocessors.llmgrade.REQUESTS_AVAILABLE', False)
    def test_llm_grade_api_error(self, preprocessors, gradebook, resources):
        """Test handling of API errors"""
        source_cell = create_grade_and_solution_cell("Question", "markdown", "foo", 10)
        source_cell.metadata.nbgrader['llm_graded'] = True
        
        student_cell = create_grade_and_solution_cell("Answer", "markdown", "foo", 10)
        student_cell.metadata.nbgrader['llm_graded'] = True
        
        nb = new_notebook()
        nb.cells.append(source_cell)
        preprocessors[0].preprocess(nb, resources)
        
        nb.cells[0] = student_cell
        gradebook.add_submission("ps0", "bar")
        
        preprocessors[1].llm_api_key = "test-key"
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with patch('nbgrader.preprocessors.llmgrade.OpenAI', return_value=mock_client):
            preprocessors[1].preprocess(nb, resources)
            preprocessors[2].preprocess(nb, resources)
        
        # Should mark for manual review
        grade_cell = gradebook.find_grade("foo", "test", "ps0", "bar")
        assert grade_cell.auto_score is None
        assert grade_cell.needs_manual_grade

    @patch('nbgrader.preprocessors.llmgrade.OPENAI_AVAILABLE', True)
    @patch('nbgrader.preprocessors.llmgrade.REQUESTS_AVAILABLE', False)
    def test_llm_grade_invalid_response(self, preprocessors, gradebook, resources):
        """Test handling of invalid LLM response"""
        source_cell = create_grade_and_solution_cell("Question", "markdown", "foo", 10)
        source_cell.metadata.nbgrader['llm_graded'] = True
        
        student_cell = create_grade_and_solution_cell("Answer", "markdown", "foo", 10)
        student_cell.metadata.nbgrader['llm_graded'] = True
        
        nb = new_notebook()
        nb.cells.append(source_cell)
        preprocessors[0].preprocess(nb, resources)
        
        nb.cells[0] = student_cell
        gradebook.add_submission("ps0", "bar")
        
        preprocessors[1].llm_api_key = "test-key"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not a number"
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('nbgrader.preprocessors.llmgrade.OpenAI', return_value=mock_client):
            preprocessors[1].preprocess(nb, resources)
            preprocessors[2].preprocess(nb, resources)
        
        # Should mark for manual review
        grade_cell = gradebook.find_grade("foo", "test", "ps0", "bar")
        assert grade_cell.auto_score is None
        assert grade_cell.needs_manual_grade

    @patch('nbgrader.preprocessors.llmgrade.OPENAI_AVAILABLE', True)
    @patch('nbgrader.preprocessors.llmgrade.REQUESTS_AVAILABLE', False)
    def test_llm_grade_out_of_range(self, preprocessors, gradebook, resources):
        """Test handling of score out of range"""
        source_cell = create_grade_and_solution_cell("Question", "markdown", "foo", 10)
        source_cell.metadata.nbgrader['llm_graded'] = True
        
        student_cell = create_grade_and_solution_cell("Answer", "markdown", "foo", 10)
        student_cell.metadata.nbgrader['llm_graded'] = True
        
        nb = new_notebook()
        nb.cells.append(source_cell)
        preprocessors[0].preprocess(nb, resources)
        
        nb.cells[0] = student_cell
        gradebook.add_submission("ps0", "bar")
        
        preprocessors[1].llm_api_key = "test-key"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "15"  # More than max_points
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('nbgrader.preprocessors.llmgrade.OpenAI', return_value=mock_client):
            preprocessors[1].preprocess(nb, resources)
            preprocessors[2].preprocess(nb, resources)
        
        # Should use 0 and mark for manual review
        grade_cell = gradebook.find_grade("foo", "test", "ps0", "bar")
        assert grade_cell.auto_score == 0.0
        assert grade_cell.needs_manual_grade