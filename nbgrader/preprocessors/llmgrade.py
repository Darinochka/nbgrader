import re
from typing import Tuple, Optional
from textwrap import dedent

from traitlets import Unicode, Float
from nbformat.notebooknode import NotebookNode
from nbconvert.exporters.exporter import ResourcesDict

from . import NbGraderPreprocessor
from .. import utils
from ..api import Gradebook, MissingEntry


try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Fallback to requests if openai is not available
    try:
        import requests
        REQUESTS_AVAILABLE = True
    except ImportError:
        REQUESTS_AVAILABLE = False


class LLMGrade(NbGraderPreprocessor):
    """Preprocessor for grading cells using LLM API"""

    llm_api_key = Unicode(
        "",
        help=dedent(
            """
            API key for the LLM service. If not set, LLM grading will be skipped.
            """
        )
    ).tag(config=True)

    llm_api_base = Unicode(
        "https://api.openai.com/v1",
        help=dedent(
            """
            Base URL for the LLM API. Defaults to OpenAI API endpoint.
            Can be set to use other OpenAI-compatible APIs.
            """
        )
    ).tag(config=True)

    llm_model = Unicode(
        "gpt-4",
        help=dedent(
            """
            Model name to use for LLM grading.
            """
        )
    ).tag(config=True)

    llm_prompt_template = Unicode(
        dedent("""
        You are grading a student's answer to a question.

        Question: {question}

        Student's Answer: {student_answer}

        Grading Instructions (hidden from student): {grading_instructions}

        Maximum Points: {max_points}

        Please evaluate the student's answer based on the grading instructions and return ONLY a number between 0 and {max_points} representing the points the student should receive. Do not include any explanation, just the number.
        """),
        help=dedent(
            """
            Template for the prompt sent to the LLM. Should contain placeholders:
            - {question}: The visible question text
            - {student_answer}: The student's answer
            - {grading_instructions}: The hidden grading instructions
            - {max_points}: Maximum points for this question
            """
        )
    ).tag(config=True)

    begin_question_delimiter = Unicode(
        "BEGIN QUESTION_LLM",
        help=dedent(
            """
            Delimiter marking the beginning of the question in LLM graded cells.
            """
        )
    ).tag(config=True)

    end_question_delimiter = Unicode(
        "END QUESTION_LLM",
        help=dedent(
            """
            Delimiter marking the end of the question in LLM graded cells.
            """
        )
    ).tag(config=True)

    begin_criteria_delimiter = Unicode(
        "BEGIN CRITERIA_LLM",
        help=dedent(
            """
            Delimiter marking the beginning of grading criteria in LLM graded cells.
            Note: This should match ClearLLMCriteria.begin_criteria_delimiter.
            """
        )
    ).tag(config=True)

    end_criteria_delimiter = Unicode(
        "END CRITERIA_LLM",
        help=dedent(
            """
            Delimiter marking the end of grading criteria in LLM graded cells.
            Note: This should match ClearLLMCriteria.end_criteria_delimiter.
            """
        )
    ).tag(config=True)

    llm_timeout = Float(
        30.0,
        help=dedent(
            """
            Timeout in seconds for LLM API calls.
            """
        )
    ).tag(config=True)

    def preprocess(self, nb: NotebookNode, resources: ResourcesDict) -> Tuple[NotebookNode, ResourcesDict]:
        # pull information from the resources
        self.notebook_id = resources['nbgrader']['notebook']
        self.assignment_id = resources['nbgrader']['assignment']
        self.student_id = resources['nbgrader']['student']
        self.db_url = resources['nbgrader']['db_url']

        # Initialize llm_scores dict in resources
        if 'llm_scores' not in resources['nbgrader']:
            resources['nbgrader']['llm_scores'] = {}

        # connect to the database
        self.gradebook = Gradebook(self.db_url)

        # Check if LLM API is available
        if not self.llm_api_key:
            self.log.warning("LLM API key not configured. LLM grading will be skipped.")
            return nb, resources

        if not OPENAI_AVAILABLE and not REQUESTS_AVAILABLE:
            self.log.error("Neither 'openai' nor 'requests' library is available. LLM grading will be skipped.")
            return nb, resources

        with self.gradebook:
            # process the cells
            nb, resources = super(LLMGrade, self).preprocess(nb, resources)

        return nb, resources

    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """Call the LLM API and return the response text."""
        try:
            if OPENAI_AVAILABLE:
                self.log.info("Calling OpenAI API with base URL %s and model %s", self.llm_api_base, self.llm_model)
                client = OpenAI(
                    api_key=self.llm_api_key,
                    base_url=self.llm_api_base,
                    # timeout=self.llm_timeout
                )
                response = client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,  # Use deterministic responses for grading
                )
                return response.choices[0].message.content.strip()
            elif REQUESTS_AVAILABLE:
                self.log.info("Calling requests API with base URL %s", self.llm_api_base)
                # Fallback to direct HTTP requests
                url = f"{self.llm_api_base}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.llm_model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0
                }
                response = requests.post(url, json=data, headers=headers, timeout=self.llm_timeout)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                return None
        except Exception as e:
            self.log.error("Error calling LLM API: %s", str(e))
            return None

    def _extract_question(self, cell_source: str) -> str:
        """Extract the question text from cell source."""
        lines = cell_source.split("\n")
        question_lines = []
        in_question = False

        for line in lines:
            if self.begin_question_delimiter in line:
                in_question = True
            elif self.end_question_delimiter in line:
                in_question = False
            elif in_question:
                question_lines.append(line)

        if in_question:
            self.log.warning("Question region not properly closed")
        
        return "\n".join(question_lines).strip()

    def _extract_criteria(self, cell_source: str) -> str:
        """Extract the grading criteria from cell source."""
        lines = cell_source.split("\n")
        criteria_lines = []
        in_criteria = False

        for line in lines:
            if self.begin_criteria_delimiter in line:
                in_criteria = True
            elif self.end_criteria_delimiter in line:
                in_criteria = False
            elif in_criteria:
                criteria_lines.append(line)

        if in_criteria:
            self.log.warning("Criteria region not properly closed")
        
        return "\n".join(criteria_lines).strip()

    def _parse_llm_score(self, response: str, max_points: float) -> Optional[float]:
        """Parse the LLM response to extract a numeric score."""
        if not response:
            return None

        # Try to extract a number from the response
        # Look for numbers that could be scores
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            try:
                score = float(numbers[0])
                # Validate score is within range
                if 0 <= score <= max_points:
                    return score
                else:
                    self.log.warning(
                        "LLM returned score %f which is outside valid range [0, %f]. Using 0.",
                        score, max_points
                    )
                    return 0.0
            except ValueError:
                pass

        self.log.warning(
            "Could not parse LLM response as a score: %s. Response will be treated as needing manual review.",
            response
        )
        return None

    def _grade_with_llm(self, cell: NotebookNode) -> Optional[float]:
        """Grade a cell using LLM."""
        if not self.llm_api_key:
            return None

        try:
            # Get the source cell from database to extract hidden instructions
            grade_id = cell.metadata['nbgrader']['grade_id']
            source_cell = self.gradebook.find_source_cell(
                grade_id,
                self.notebook_id,
                self.assignment_id
            )

            # Extract question and criteria from source cell
            question_text = self._extract_question(source_cell.source)
            criteria_text = self._extract_criteria(source_cell.source)

            # Get student's answer from current cell
            student_answer = cell.source

            # Get max points
            max_points = float(cell.metadata['nbgrader']['points'])

            # Format the prompt
            prompt = self.llm_prompt_template.format(
                question=question_text,
                student_answer=student_answer,
                grading_instructions=criteria_text,
                max_points=max_points
            )

            # Call LLM API
            self.log.info("Calling LLM API for cell %s", grade_id)
            response = self._call_llm_api(prompt)

            if response is None:
                self.log.warning("LLM API call failed for cell %s", grade_id)
                return None

            # Parse the score
            score = self._parse_llm_score(response, max_points)
            if score is not None:
                self.log.info("LLM graded cell %s: %f / %f", grade_id, score, max_points)
            else:
                self.log.warning("Could not parse LLM response for cell %s", grade_id)

            return score

        except MissingEntry:
            self.log.warning("Source cell not found for grade_id %s", grade_id)
            return None
        except Exception as e:
            self.log.error("Error grading cell %s with LLM: %s", grade_id, str(e))
            return None

    def preprocess_cell(self,
                        cell: NotebookNode,
                        resources: ResourcesDict,
                        cell_index: int
                        ) -> Tuple[NotebookNode, ResourcesDict]:
        # Only process LLM graded cells
        if not utils.is_llm_graded(cell):
            return cell, resources

        # Grade the cell using LLM
        score = self._grade_with_llm(cell)

        # Store the score in resources dict (not in cell metadata to avoid schema validation errors)
        # SaveAutoGrades will pick it up and save it to the database
        grade_id = cell.metadata['nbgrader']['grade_id']
        if 'llm_scores' not in resources['nbgrader']:
            resources['nbgrader']['llm_scores'] = {}
        
        if score is not None:
            resources['nbgrader']['llm_scores'][grade_id] = score
        else:
            # Mark as None for manual review
            resources['nbgrader']['llm_scores'][grade_id] = None
            self.log.warning(
                "LLM grading failed for cell %s. Marking for manual review.",
                grade_id
            )

        return cell, resources