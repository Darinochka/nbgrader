import os
import shutil
import pytest
import nbformat
from os.path import join

from .. import run_nbgrader
from .base import BaseTestApp


# Absolute path to the LLM source notebook used as instructor input.
_LLM_SOURCE_NB = os.path.join(
    os.path.dirname(__file__), '..', 'preprocessors', 'files', 'test_llm_source.ipynb'
)


@pytest.mark.usefixtures("temp_cwd")
class TestLLMGenerateAssignment(BaseTestApp):
    """Integration tests: generate_assignment produces correct student notebook for LLM cells."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _setup_and_generate(self, course_dir: str) -> str:
        """Copy LLM source notebook into source/, run generate_assignment, return released path."""
        dest = join(course_dir, "source", "ps1", "test_llm_source.ipynb")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(_LLM_SOURCE_NB, dest)

        run_nbgrader(["db", "assignment", "add", "ps1"])
        run_nbgrader(["generate_assignment", "ps1"])

        return join(course_dir, "release", "ps1", "test_llm_source.ipynb")

    def _load_released(self, course_dir: str):
        """Run pipeline and return (raw_str, parsed_notebook)."""
        released_path = self._setup_and_generate(course_dir)
        raw = self._file_contents(released_path)
        with open(released_path) as f:
            nb = nbformat.read(f, as_version=4)
        return raw, nb

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_released_notebook_is_created(self, course_dir, temp_cwd):
        """generate_assignment must produce the released notebook file."""
        released_path = self._setup_and_generate(course_dir)
        assert os.path.isfile(released_path)

    def test_no_llm_delimiters_in_released_notebook(self, course_dir, temp_cwd):
        """No LLM delimiter markers must appear in the released notebook."""
        raw, _ = self._load_released(course_dir)

        assert "BEGIN QUESTION_LLM" not in raw
        assert "END QUESTION_LLM" not in raw
        assert "BEGIN CRITERIA_LLM" not in raw
        assert "END CRITERIA_LLM" not in raw

    def test_question_text_removed_from_released_notebook(self, course_dir, temp_cwd):
        """Instructor question text must not be visible in the released notebook."""
        raw, _ = self._load_released(course_dir)

        # From llm_essay_q1 question block
        assert "supervised" not in raw
        # From llm_essay_q2 question block
        assert "overfitting" not in raw
        # From llm_code_q1 question block (comment inside code cell)
        assert "square" not in raw

    def test_criteria_text_removed_from_released_notebook(self, course_dir, temp_cwd):
        """Grading criteria text must not be visible in the released notebook."""
        raw, _ = self._load_released(course_dir)

        # From llm_essay_q1 criteria
        assert "labeled data" not in raw
        # From llm_essay_q2 criteria
        assert "regularisation" not in raw
        # From llm_code_q1 criteria
        assert "x * x" not in raw

    def test_llm_markdown_cells_replaced_with_text_stub(self, course_dir, temp_cwd):
        """LLM-graded markdown cells must be replaced with the text stub."""
        _, nb = self._load_released(course_dir)

        llm_markdown_cells = [
            c for c in nb.cells
            if c.cell_type == "markdown" and c.metadata.get("nbgrader", {}).get("llm_graded")
        ]
        assert llm_markdown_cells, "expected at least one LLM markdown cell in released notebook"

        for cell in llm_markdown_cells:
            grade_id = cell.metadata["nbgrader"]["grade_id"]
            assert cell.source == "YOUR ANSWER HERE", (
                f"Markdown LLM cell {grade_id!r} was not replaced with text stub; got: {cell.source!r}"
            )

    def test_llm_code_cells_replaced_with_code_stub(self, course_dir, temp_cwd):
        """LLM-graded code cells must be replaced with the Python code stub."""
        _, nb = self._load_released(course_dir)

        llm_code_cells = [
            c for c in nb.cells
            if c.cell_type == "code" and c.metadata.get("nbgrader", {}).get("llm_graded")
        ]
        assert llm_code_cells, "expected at least one LLM code cell in released notebook"

        expected_stub = "# YOUR CODE HERE\nraise NotImplementedError()"
        for cell in llm_code_cells:
            grade_id = cell.metadata["nbgrader"]["grade_id"]
            assert cell.source == expected_stub, (
                f"Code LLM cell {grade_id!r} was not replaced with code stub; got: {cell.source!r}"
            )

    def test_non_llm_cells_are_unchanged(self, course_dir, temp_cwd):
        """Regular (non-LLM) cells must not be modified by the pipeline."""
        _, nb = self._load_released(course_dir)

        intro_cell = nb.cells[0]
        assert intro_cell.metadata.get("nbgrader", {}).get("llm_graded") is None
        assert "Machine Learning Essay Questions" in intro_cell.source
