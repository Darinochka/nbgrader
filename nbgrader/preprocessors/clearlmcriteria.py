import re

from traitlets import Unicode, Bool
from textwrap import dedent

from . import NbGraderPreprocessor
from .. import utils
from nbformat.notebooknode import NotebookNode
from nbconvert.exporters.exporter import ResourcesDict
from typing import Tuple


class ClearLLMCriteria(NbGraderPreprocessor):
    """Preprocessor for removing LLM criteria from cells during generate_assignment"""

    begin_criteria_delimiter = Unicode(
        "BEGIN CRITERIA_LLM",
        help="The delimiter marking the beginning of LLM grading criteria"
    ).tag(config=True)

    end_criteria_delimiter = Unicode(
        "END CRITERIA_LLM",
        help="The delimiter marking the end of LLM grading criteria"
    ).tag(config=True)

    enforce_metadata = Bool(
        True,
        help=dedent(
            """
            Whether or not to complain if cells containing LLM criteria regions
            are not marked as LLM graded cells. WARNING: this will potentially cause
            things to break if you are using the full nbgrader pipeline. ONLY
            disable this option if you are only ever planning to use nbgrader
            assign.
            """
        )
    ).tag(config=True)

    def _remove_criteria_region(self, cell: NotebookNode) -> bool:
        """Find a region in the cell that is delimited by
        `self.begin_criteria_delimiter` and `self.end_criteria_delimiter` (e.g.
        ### BEGIN CRITERIA_LLM and ### END CRITERIA_LLM). Remove that region.

        This modifies the cell in place, and then returns True if a
        criteria region was removed, and False otherwise.
        """
        # pull out the cell input/source
        lines = cell.source.split("\n")

        new_lines = []
        in_criteria = False
        removed_criteria = False

        for line in lines:
            # begin the criteria area
            if self.begin_criteria_delimiter in line:
                # check to make sure this isn't a nested BEGIN CRITERIA_LLM
                # region
                if in_criteria:
                    raise RuntimeError(
                        "Encountered nested begin criteria statements")
                in_criteria = True
                removed_criteria = True
                # Don't add the delimiter line itself

            # end the criteria area
            elif self.end_criteria_delimiter in line:
                in_criteria = False
                # Don't add the delimiter line itself

            # add lines as long as it's not in the criteria region
            elif not in_criteria:
                new_lines.append(line)

        # we finished going through all the lines, but didn't find a
        # matching END CRITERIA_LLM statement
        if in_criteria:
            raise RuntimeError("No end criteria statement found")

        # replace the cell source
        cell.source = "\n".join(new_lines)

        return removed_criteria

    def preprocess(self, nb: NotebookNode, resources: ResourcesDict) -> Tuple[NotebookNode, ResourcesDict]:
        nb, resources = super(ClearLLMCriteria, self).preprocess(nb, resources)
        if 'celltoolbar' in nb.metadata:
            del nb.metadata['celltoolbar']
        return nb, resources

    def preprocess_cell(self,
                        cell: NotebookNode,
                        resources: ResourcesDict,
                        cell_index: int
                        ) -> Tuple[NotebookNode, ResourcesDict]:
        # remove criteria regions
        removed_criteria = self._remove_criteria_region(cell)

        # determine whether the cell is an LLM graded cell
        is_llm_graded = utils.is_llm_graded(cell)

        # check that it is marked as an LLM graded cell if we remove a criteria
        # region -- if it's not, then this is a problem
        if not is_llm_graded and removed_criteria:
            if self.enforce_metadata:
                raise RuntimeError(
                    "LLM criteria region detected in a non-LLM graded cell; "
                    "please make sure all LLM criteria regions are within "
                    "'LLM graded answer' cells."
                )

        return cell, resources