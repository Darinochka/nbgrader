The ``nbgrader_config.py`` file
===============================

.. seealso::

    :doc:`config_options`
        A list of all config options for ``nbgrader_config.py``.

The ``nbgrader_config.py`` file is the main place for you to specify options
for configuring nbgrader. Normally, it should be located in the same directory
as where you are running the nbgrader commands, or you can place it in one of a
number of other locations on your system. These locations correspond to the
configuration directories that Jupyter itself looks in; you can find out what
these are by running ``jupyter --paths``.

Things get a bit more complicated in certain setups, so this document aims to clarify how to setup the ``nbgrader_config.py`` file in multiple different scenarios.

Using ``nbgrader_config.py``
----------------------------

To set a configuration option in the config file, you need to use the ``c``
variable which actually stores the config. For example::

    c = get_config()
    c.CourseDirectory.course_id = "course101"

To get an example config file, you can run ``nbgrader generate_config``.


Use Case 1: nbgrader and ``jupyter notebook`` run in the same directory
-----------------------------------------------------------------------

The easiest way to use nbgrader and the formgrader extension is to run both
from the same directory. For example::

    nbgrader quickstart course101
    cd ./course101
    jupyter notebook

In this case, there should be a ``nbgrader_config.py`` file in the directory
``./course101``, which corresponds both to the directory where the notebook is
running and the directory where the nbgrader commands will be run.

As mentioned above, you can actually put the ``nbgrader_config.py`` file in any of the directories listed by ``jupyter --paths`` in the "Config" section.


Use Case 2: nbgrader and ``jupyter notebook`` run in separate directories
-------------------------------------------------------------------------

.. warning::

    The nbgrader course directory must be a subdirectory of where you run the
    Jupyter notebook.

A common use case is to run the notebook server from the root of your home
directory, which is likely not the place where you will be running nbgrader
from. In this case, you will need to tell the nbgrader extensions---which run
as part of the notebook server---where to find your course directory. In this
case, you want *two* ``nbgrader_config.py`` files: one for your main course directory (where you run the nbgrader commands) and one that specifies for the notebook where the course directory is.

For example, the ``nbgrader_config.py`` that the notebook knows about could be placed in ``~/.jupyter/nbgrader_config.py``, and it would include a path to where the main course directory is::

    c = get_config()
    c.CourseDirectory.root = "/path/to/course/directory"

Then you would additionally have a config file at ``/path/to/course/directory/nbgrader_config.py``.

Use Case 3: using config from a specific sub directory
------------------------------------------------------

.. warning::

    This option should not be used with a multiuser Jupyterlab instance, as it modifies
    certain objects in the running instance, and can probably prevent other users
    from using *formgrader* correctly. Also, if you have a JupyterHub installation,
    you should use the settings described in the following section.

You may need to use a dedicated configuration file for each course without configuring
JupyterHub for all courses. In this case, the config file used will be the one from the
current directory in the filebrowser panel, instead of the one from the directory where
the jupyter server started.

This option is not enabled by default. It can be enabled by using the settings panel:
*Nbgrader -> Formgrader* and check *Allow local nbgrader config file*.

A new item is displayed in the *nbgrader* menu (or in the command palette), to open
formgrader from the local director: *Formgrader (local)*.

.. warning::

    If paths are used in the configuration file, note that the root of the relative
    paths will always be the directory where the jupyter server was started, and not
    the directory containing the ``nbgrader_config.py`` file.

Use Case 4: nbgrader and JupyterHub
-----------------------------------

.. seealso::

    :doc:`jupyterhub_config`
        Further information on using nbgrader with JupyterHub.

The setup of ``nbgrader_config.py`` files gets a bit more complicated when you
are running a shared server with JupyterHub. In this case, you will likely need (at least) three separate ``nbgrader_config.py`` files:

1. A global ``nbgrader_config.py`` (for example, placed in a global path like ``/usr/local/etc/jupyter`` or ``/etc/jupyter``, but check ``jupyter --paths`` to see which ones are valid on your system). This global file should include information relevant to all students, instructors, and formgraders, such as the location of the exchange directory.

2. An ``nbgrader_config.py`` that tells the notebook server running the formgrader where the course directory is located (as in Use Case 2). The options in this config file will only be relevant for the formgrader, and not any other user accounts.

3. An ``nbgrader_config.py`` file in the course directory itself. The options in this config file will only be relevant for the formgrader, and not any other user accounts.

LLM Grading Configuration
--------------------------

nbgrader supports LLM (Large Language Model) grading for answer cells. To use this feature, you need to configure the LLM API settings in your ``nbgrader_config.py`` file.

Here is an example configuration for using OpenAI's API::

    c = get_config()
    
    # LLM API configuration
    c.LLMGrade.llm_api_key = "your-api-key-here"
    c.LLMGrade.llm_api_base = "https://api.openai.com/v1"
    c.LLMGrade.llm_model = "gpt-4"
    
    # Optional: Customize the prompt template
    c.LLMGrade.llm_prompt_template = """
    You are grading a student's answer to a question.
    
    Question: {question}
    
    Student's Answer: {student_answer}
    
    Grading Instructions (hidden from student): {grading_instructions}
    
    Maximum Points: {max_points}
    
    Please evaluate the student's answer based on the grading instructions and return ONLY a number between 0 and {max_points} representing the points the student should receive. Do not include any explanation, just the number.
    """
    
    # Optional: Customize delimiters for question and criteria regions
    # Default: 'BEGIN QUESTION_LLM'
    c.LLMGrade.begin_question_delimiter = 'BEGIN QUESTION_LLM'
    # Default: 'END QUESTION_LLM'
    c.LLMGrade.end_question_delimiter = 'END QUESTION_LLM'
    # Default: 'BEGIN CRITERIA_LLM'
    c.ClearLLMCriteria.begin_criteria_delimiter = 'BEGIN CRITERIA_LLM'
    # Default: 'END CRITERIA_LLM'
    c.ClearLLMCriteria.end_criteria_delimiter = 'END CRITERIA_LLM'
    
    # Optional: Set timeout for API calls (in seconds)
    c.LLMGrade.llm_timeout = 30.0

For using other OpenAI-compatible APIs (such as local models or other providers), you can change the ``llm_api_base`` to point to your API endpoint::

    c.LLMGrade.llm_api_base = "https://your-api-endpoint.com/v1"

**Note:** You need to install either the ``openai`` Python package (recommended) or the ``requests`` package for LLM grading to work. If neither is available, LLM grading will be skipped with a warning.

LLM Graded Cell Structure
--------------------------

LLM graded cells use a special structure with two types of regions:

1. **Question region** (visible to students): between ``### BEGIN QUESTION_LLM`` and ``### END QUESTION_LLM``
2. **Criteria region** (hidden from students, used by LLM): between ``### BEGIN CRITERIA_LLM`` and ``### END CRITERIA_LLM``

Example LLM graded cell in source version::

    ### BEGIN QUESTION_LLM
    Explain what recursion is in programming. Provide an example.
    ### END QUESTION_LLM

    ### BEGIN CRITERIA_LLM
    Grading criteria:
    - 10 points: Complete explanation with correct example
    - 7 points: Good explanation but incomplete example
    - 4 points: Partial understanding, no example
    - 0 points: Incorrect answer
    ### END CRITERIA_LLM

After running ``nbgrader generate_assignment``, students will see::

    Explain what recursion is in programming. Provide an example.

    YOUR ANSWER HERE

The criteria region is completely removed and only the question remains visible.

**Note:** The delimiters can be written with or without the ``###`` prefix. Both ``### BEGIN QUESTION_LLM`` and ``BEGIN QUESTION_LLM`` will work, as long as the delimiter text appears in the line.

