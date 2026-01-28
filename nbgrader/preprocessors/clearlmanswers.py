import re

from traitlets import Unicode, Dict, Bool
from textwrap import dedent

from . import NbGraderPreprocessor
from .. import utils
from nbformat.notebooknode import NotebookNode
from nbconvert.exporters.exporter import ResourcesDict
from typing import Tuple


class ClearLLMAnswers(NbGraderPreprocessor):
    """Preprocessor for clearing LLM graded answer cells during generate_assignment"""

    code_stub = Dict(
        dict(python="# YOUR CODE HERE\nraise NotImplementedError()",
             R="# YOUR CODE HERE\nfail()",
             matlab="% YOUR CODE HERE\nerror('No Answer Given!')",
             octave="% YOUR CODE HERE\nerror('No Answer Given!')",
             sas="/* YOUR CODE HERE */\n %notImplemented;",
             java="// YOUR CODE HERE"),
        help="The code snippet that will replace code answers in LLM graded cells"
    ).tag(config=True)

    text_stub = Unicode(
        "YOUR ANSWER HERE",
        help="The text snippet that will replace written answers in LLM graded cells"
    ).tag(config=True)

    begin_question_delimiter = Unicode(
        "BEGIN QUESTION_LLM",
        help="The delimiter marking the beginning of the question in LLM graded cells"
    ).tag(config=True)

    end_question_delimiter = Unicode(
        "END QUESTION_LLM",
        help="The delimiter marking the end of the question in LLM graded cells"
    ).tag(config=True)

    begin_criteria_delimiter = Unicode(
        "BEGIN CRITERIA_LLM",
        help="The delimiter marking the beginning of criteria (for skipping)"
    ).tag(config=True)

    end_criteria_delimiter = Unicode(
        "END CRITERIA_LLM",
        help="The delimiter marking the end of criteria (for skipping)"
    ).tag(config=True)

    def preprocess(self, nb: NotebookNode, resources: ResourcesDict) -> Tuple[NotebookNode, ResourcesDict]:
        language = nb.metadata.get("kernelspec", {}).get("language", "python")
        if language not in self.code_stub:
            raise ValueError(
                "language '{}' has not been specified in "
                "ClearLLMAnswers.code_stub".format(language))

        resources["language"] = language
        nb, resources = super(ClearLLMAnswers, self).preprocess(nb, resources)
        if 'celltoolbar' in nb.metadata:
            del nb.metadata['celltoolbar']
        return nb, resources

    def _extract_question_region(self, cell: NotebookNode) -> Tuple[str, bool]:
        """Extract the question region from the cell.
        
        Returns:
            Tuple of (question_text, found_question_region)
        """
        lines = cell.source.split("\n")
        question_lines = []
        in_question = False
        found_question = False
        before_question_lines = []

        for line in lines:
            # Skip criteria regions (they should already be removed by ClearLLMCriteria)
            # But check anyway in case they weren't removed yet
            if "BEGIN CRITERIA_LLM" in line or "END CRITERIA_LLM" in line:
                continue
            if self.begin_question_delimiter in line:
                if in_question:
                    raise RuntimeError("Encountered nested begin question statements")
                in_question = True
                found_question = True
                # Don't add the delimiter line itself
                # Add any content before question region
                if before_question_lines:
                    question_lines.extend(before_question_lines)
                    before_question_lines = []
            elif self.end_question_delimiter in line:
                in_question = False
                # Don't add the delimiter line itself
            elif in_question:
                question_lines.append(line)
            elif not found_question:
                # Before question region starts, keep the content (for backward compatibility)
                before_question_lines.append(line)

        if in_question:
            raise RuntimeError("No end question statement found")

        # If no question region found, use all content before criteria
        if not found_question and before_question_lines:
            question_lines = before_question_lines

        return "\n".join(question_lines), found_question

    def preprocess_cell(self,
                        cell: NotebookNode,
                        resources: ResourcesDict,
                        cell_index: int
                        ) -> Tuple[NotebookNode, ResourcesDict]:
        # Only process LLM graded cells
        if not utils.is_llm_graded(cell):
            return cell, resources

        language = resources["language"]
        
        # Extract question region (this also removes criteria regions)
        question_text, found_question = self._extract_question_region(cell)
        
        # Clean up question text (remove empty lines at start/end)
        question_text = question_text.strip()
        
        if cell.cell_type == 'code':
            stub = self.code_stub[language]
        else:
            stub = self.text_stub
        
        # If question region was found, keep it and add stub
        if found_question and question_text:
            # Combine question with stub
            cell.source = question_text + "\n\n" + stub
        else:
            # No question region found or empty, replace entire cell with stub
            # This handles the case where question is not explicitly marked
            cell.source = stub

        return cell, resources