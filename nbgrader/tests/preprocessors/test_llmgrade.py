import copy
import pytest
from unittest.mock import patch, MagicMock

from nbformat.v4 import new_notebook

from ...preprocessors import SaveCells, LLMGrade, SaveAutoGrades
from ...preprocessors.clearlmanswers import ClearLLMAnswers
from ...preprocessors.clearlmcriteria import ClearLLMCriteria
from ...api import Gradebook
from .base import BaseTestPreprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
            "student": "bar",
        }
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_source_nb(base: BaseTestPreprocessor):
    """Load the LLM source notebook (instructor version with all delimiters)."""
    return base._read_nb("files/test_llm_source.ipynb", validate=False)


def _find_llm_cell(nb, grade_id: str):
    """Return the first cell whose nbgrader grade_id matches."""
    for cell in nb.cells:
        meta = cell.metadata.get("nbgrader", {})
        if meta.get("grade_id") == grade_id:
            return cell
    raise KeyError(f"Cell with grade_id={grade_id!r} not found")


# ---------------------------------------------------------------------------
# Tests: ClearLLMAnswers (student notebook generation)
# ---------------------------------------------------------------------------

class TestClearLLMAnswers(BaseTestPreprocessor):
    """ClearLLMAnswers replaces the full LLM-graded cell with a stub."""

    @pytest.fixture
    def preprocessor(self):
        return ClearLLMAnswers()

    def test_question_removed_from_student_notebook(self, preprocessor):
        """Question text must not appear in the released student notebook."""
        nb = _load_source_nb(self)
        preprocessor.preprocess(nb, {})

        cell = _find_llm_cell(nb, "llm_essay_q1")
        assert "BEGIN QUESTION_LLM" not in cell.source
        assert "END QUESTION_LLM" not in cell.source
        assert "supervised" not in cell.source

    def test_criteria_removed_from_student_notebook(self, preprocessor):
        """Grading criteria must not appear in the released student notebook."""
        nb = _load_source_nb(self)
        preprocessor.preprocess(nb, {})

        cell = _find_llm_cell(nb, "llm_essay_q1")
        assert "BEGIN CRITERIA_LLM" not in cell.source
        assert "END CRITERIA_LLM" not in cell.source
        assert "labeled data" not in cell.source

    def test_cell_replaced_with_stub(self, preprocessor):
        """LLM markdown cells must be replaced with the text stub."""
        nb = _load_source_nb(self)
        preprocessor.preprocess(nb, {})

        for cell in nb.cells:
            if cell.metadata.get("nbgrader", {}).get("llm_graded") and cell.cell_type == "markdown":
                assert cell.source == preprocessor.text_stub

    def test_non_llm_cells_unchanged(self, preprocessor):
        """Regular (non-LLM) cells must not be modified."""
        nb = _load_source_nb(self)
        original_intro = nb.cells[0].source
        preprocessor.preprocess(nb, {})
        assert nb.cells[0].source == original_intro

    def test_code_cell_replaced_with_code_stub(self, preprocessor):
        """LLM-graded code cells must be replaced with the language-appropriate code stub."""
        nb = _load_source_nb(self)
        preprocessor.preprocess(nb, {})

        cell = _find_llm_cell(nb, "llm_code_q1")
        assert cell.source == preprocessor.code_stub["python"]

    def test_code_cell_question_and_criteria_not_in_stub(self, preprocessor):
        """After preprocessing, delimiters, question text and criteria must not appear in the code cell."""
        nb = _load_source_nb(self)
        preprocessor.preprocess(nb, {})

        cell = _find_llm_cell(nb, "llm_code_q1")
        assert "BEGIN QUESTION_LLM" not in cell.source
        assert "END QUESTION_LLM" not in cell.source
        assert "BEGIN CRITERIA_LLM" not in cell.source
        assert "END CRITERIA_LLM" not in cell.source
        assert "square" not in cell.source


# ---------------------------------------------------------------------------
# Tests: ClearLLMCriteria (alternative: keep question, strip criteria only)
# ---------------------------------------------------------------------------

class TestClearLLMCriteria(BaseTestPreprocessor):
    """ClearLLMCriteria removes only the criteria block, question stays."""

    @pytest.fixture
    def preprocessor(self):
        return ClearLLMCriteria()

    def test_criteria_removed(self, preprocessor):
        """Criteria block must be stripped from every LLM cell."""
        nb = _load_source_nb(self)
        preprocessor.preprocess(nb, {})

        cell = _find_llm_cell(nb, "llm_essay_q1")
        assert "BEGIN CRITERIA_LLM" not in cell.source
        assert "END CRITERIA_LLM" not in cell.source
        assert "labeled data" not in cell.source

    def test_question_still_visible(self, preprocessor):
        """Question text must remain after criteria are removed."""
        nb = _load_source_nb(self)
        preprocessor.preprocess(nb, {})

        cell = _find_llm_cell(nb, "llm_essay_q1")
        assert "supervised" in cell.source

    def test_criteria_removed_from_code_cell(self, preprocessor):
        """Criteria block must be stripped from LLM-graded code cells."""
        nb = _load_source_nb(self)
        preprocessor.preprocess(nb, {})

        cell = _find_llm_cell(nb, "llm_code_q1")
        assert "BEGIN CRITERIA_LLM" not in cell.source
        assert "END CRITERIA_LLM" not in cell.source
        assert "x * x" not in cell.source

    def test_question_still_visible_in_code_cell(self, preprocessor):
        """Question text must remain in code cell after criteria are removed."""
        nb = _load_source_nb(self)
        preprocessor.preprocess(nb, {})

        cell = _find_llm_cell(nb, "llm_code_q1")
        assert "square" in cell.source


# ---------------------------------------------------------------------------
# Tests: LLMGrade prompt assembly
# ---------------------------------------------------------------------------

class TestLLMGradePromptAssembly(BaseTestPreprocessor):
    """Verify that the LLM prompt contains the right pieces."""

    @pytest.fixture
    def grader(self):
        return LLMGrade()

    def test_extract_question(self, grader):
        """_extract_question returns only the question text, no delimiters."""
        source = (
            "### BEGIN QUESTION_LLM\n"
            "What is supervised learning?\n"
            "### END QUESTION_LLM\n"
            "\n"
            "### BEGIN CRITERIA_LLM\n"
            "Give 10 points for a correct answer.\n"
            "### END CRITERIA_LLM"
        )
        question = grader._extract_question(source)
        assert "What is supervised learning?" in question
        assert "BEGIN QUESTION_LLM" not in question
        assert "END QUESTION_LLM" not in question
        assert "CRITERIA" not in question

    def test_extract_criteria(self, grader):
        """_extract_criteria returns only the criteria text, no delimiters."""
        source = (
            "### BEGIN QUESTION_LLM\n"
            "What is supervised learning?\n"
            "### END QUESTION_LLM\n"
            "\n"
            "### BEGIN CRITERIA_LLM\n"
            "Give 10 points for a correct answer.\n"
            "### END CRITERIA_LLM"
        )
        criteria = grader._extract_criteria(source)
        assert "Give 10 points" in criteria
        assert "BEGIN CRITERIA_LLM" not in criteria
        assert "END CRITERIA_LLM" not in criteria
        assert "QUESTION" not in criteria

    def test_prompt_contains_question_criteria_and_answer(self, grader):
        """The formatted prompt must include question, criteria and student answer."""
        question = "What is supervised learning?"
        criteria = "Give 10 points for a correct answer."
        student_answer = "Supervised learning uses labeled data."
        max_points = 10.0

        prompt = grader.llm_prompt_template.format(
            question=question,
            student_answer=student_answer,
            grading_instructions=criteria,
            max_points=max_points,
        )

        assert question in prompt
        assert criteria in prompt
        assert student_answer in prompt
        assert str(int(max_points)) in prompt


# ---------------------------------------------------------------------------
# Tests: LLMGrade end-to-end with mocked LLM API
# ---------------------------------------------------------------------------

class TestLLMGradeWithMock(BaseTestPreprocessor):
    """Full pipeline test: SaveCells → LLMGrade (mocked) → SaveAutoGrades."""

    @pytest.fixture
    def preprocessors(self):
        return (SaveCells(), LLMGrade(), SaveAutoGrades())

    def _setup_source_in_db(self, preprocessors, resources):
        """Run SaveCells on the source notebook to populate the DB."""
        nb = _load_source_nb(self)
        preprocessors[0].preprocess(nb, resources)
        return nb

    def _make_submitted_nb(self, grade_id: str, student_answer: str, points: float):
        """Build a minimal submitted notebook with a single student-answered cell."""
        from nbformat.v4 import new_markdown_cell
        cell = new_markdown_cell(source=student_answer)
        cell.metadata["nbgrader"] = {
            "grade": True,
            "grade_id": grade_id,
            "locked": False,
            "points": points,
            "schema_version": 4,
            "solution": True,
            "task": False,
            "llm_graded": True,
        }
        nb = new_notebook()
        nb.cells.append(cell)
        nb.metadata["kernelspec"] = {"language": "python", "name": "python3",
                                     "display_name": "Python 3"}
        return nb

    def _make_submitted_nb_code(self, grade_id: str, student_code: str, points: float):
        """Build a minimal submitted notebook with a single student code cell."""
        from nbformat.v4 import new_code_cell
        cell = new_code_cell(source=student_code)
        cell.metadata["nbgrader"] = {
            "grade": True,
            "grade_id": grade_id,
            "locked": False,
            "points": points,
            "schema_version": 4,
            "solution": True,
            "task": False,
            "llm_graded": True,
        }
        nb = new_notebook()
        nb.cells.append(cell)
        nb.metadata["kernelspec"] = {"language": "python", "name": "python3",
                                     "display_name": "Python 3"}
        return nb

    def test_llm_grade_returns_score_from_api(self, preprocessors, gradebook, resources):
        """LLMGrade must call the API and store the returned score."""
        self._setup_source_in_db(preprocessors, resources)
        gradebook.add_submission("ps0", "bar")

        submitted = self._make_submitted_nb(
            "llm_essay_q1",
            "Supervised learning uses labeled data (e.g. classification). "
            "Unsupervised learning works with unlabeled data (e.g. clustering).",
            10.0,
        )

        preprocessors[1].llm_api_key = "test-key"

        with patch.object(preprocessors[1], "_call_llm_api", return_value="8") as mock_api:
            preprocessors[1].preprocess(submitted, resources)
            preprocessors[2].preprocess(submitted, resources)

        mock_api.assert_called_once()
        prompt_used = mock_api.call_args[0][0]
        assert "supervised" in prompt_used.lower()
        assert "labeled data" in prompt_used.lower()
        assert "labeled data" in prompt_used.lower()

        grade = gradebook.find_grade("llm_essay_q1", "test", "ps0", "bar")
        assert grade.auto_score == 8.0
        assert not grade.needs_manual_grade

    def test_prompt_contains_student_answer(self, preprocessors, gradebook, resources):
        """The actual student answer must appear in the prompt sent to the LLM."""
        self._setup_source_in_db(preprocessors, resources)
        gradebook.add_submission("ps0", "bar")

        student_answer = "My unique answer about neural networks and backpropagation."
        submitted = self._make_submitted_nb("llm_essay_q1", student_answer, 10.0)

        preprocessors[1].llm_api_key = "test-key"

        with patch.object(preprocessors[1], "_call_llm_api", return_value="5") as mock_api:
            preprocessors[1].preprocess(submitted, resources)

        prompt_used = mock_api.call_args[0][0]
        assert student_answer in prompt_used

    def test_prompt_contains_source_question_and_criteria(self, preprocessors, gradebook, resources):
        """Question and criteria extracted from the DB source must appear in the prompt."""
        self._setup_source_in_db(preprocessors, resources)
        gradebook.add_submission("ps0", "bar")

        submitted = self._make_submitted_nb("llm_essay_q1", "Some answer.", 10.0)
        preprocessors[1].llm_api_key = "test-key"

        with patch.object(preprocessors[1], "_call_llm_api", return_value="3") as mock_api:
            preprocessors[1].preprocess(submitted, resources)

        prompt_used = mock_api.call_args[0][0]
        # Question text from source notebook
        assert "supervised" in prompt_used.lower()
        # Criteria text from source notebook
        assert "labeled data" in prompt_used.lower()

    def test_failed_api_call_marks_for_manual_review(self, preprocessors, gradebook, resources):
        """When the API call fails the cell must be flagged for manual review."""
        self._setup_source_in_db(preprocessors, resources)
        gradebook.add_submission("ps0", "bar")

        submitted = self._make_submitted_nb("llm_essay_q1", "Some answer.", 10.0)
        preprocessors[1].llm_api_key = "test-key"

        with patch.object(preprocessors[1], "_call_llm_api", return_value=None):
            preprocessors[1].preprocess(submitted, resources)
            preprocessors[2].preprocess(submitted, resources)

        grade = gradebook.find_grade("llm_essay_q1", "test", "ps0", "bar")
        assert grade.auto_score is None
        assert grade.needs_manual_grade

    def test_non_llm_cells_are_skipped(self, preprocessors, gradebook, resources):
        """LLMGrade must not call the API for cells without llm_graded=True."""
        gradebook.add_submission("ps0", "bar")

        from nbformat.v4 import new_markdown_cell
        cell = new_markdown_cell(source="Regular answer")
        cell.metadata["nbgrader"] = {
            "grade": True,
            "grade_id": "regular_q",
            "locked": False,
            "points": 5,
            "schema_version": 4,
            "solution": True,
            "task": False,
        }
        nb = new_notebook()
        nb.cells.append(cell)

        preprocessors[1].llm_api_key = "test-key"
        with patch.object(preprocessors[1], "_call_llm_api") as mock_api:
            preprocessors[1].preprocess(nb, resources)

        mock_api.assert_not_called()

    def test_missing_api_key_skips_grading(self, preprocessors, gradebook, resources):
        """LLMGrade must skip all grading when no API key is configured."""
        self._setup_source_in_db(preprocessors, resources)
        gradebook.add_submission("ps0", "bar")

        submitted = self._make_submitted_nb("llm_essay_q1", "Some answer.", 10.0)
        preprocessors[1].llm_api_key = ""

        with patch.object(preprocessors[1], "_call_llm_api") as mock_api:
            preprocessors[1].preprocess(submitted, resources)

        mock_api.assert_not_called()

    def test_llm_grade_code_cell_returns_score(self, preprocessors, gradebook, resources):
        """LLMGrade must grade code cells and store the score just like markdown cells."""
        self._setup_source_in_db(preprocessors, resources)
        gradebook.add_submission("ps0", "bar")

        submitted = self._make_submitted_nb_code(
            "llm_code_q1",
            "def square(x):\n    return x * x",
            5.0,
        )

        preprocessors[1].llm_api_key = "test-key"

        with patch.object(preprocessors[1], "_call_llm_api", return_value="4") as mock_api:
            preprocessors[1].preprocess(submitted, resources)
            preprocessors[2].preprocess(submitted, resources)

        mock_api.assert_called_once()
        prompt_used = mock_api.call_args[0][0]
        assert "square" in prompt_used.lower()
        assert "x * x" in prompt_used or "x ** 2" in prompt_used

        grade = gradebook.find_grade("llm_code_q1", "test", "ps0", "bar")
        assert grade.auto_score == 4.0
        assert not grade.needs_manual_grade
