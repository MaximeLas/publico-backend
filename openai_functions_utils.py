from pydantic import BaseModel, Field



class ImplicitQuestion(BaseModel):
    '''An implicit question to be answered to fill in any missing information required to make the answer comprehensive.'''

    question: str = Field(
        None,
        description='The question to be answered to fill in missing information.',
    )


def function_for_comprehensiveness_check(missing_information: str, implicit_questions: list[ImplicitQuestion]):
    '''
    Check for comprehensiveness of an answer to a grant application question and return missing information and implicit questions.

    Args:
        missing_information: Description of the information that should be in a good grant application answer to the given question but is missing from this answer.
        implicit_questions: Questions to be answered to fill in any missing information required to make the answer comprehensive.
    '''

    print(f"Missing information: {missing_information}")
    for i, question in enumerate(implicit_questions):
        print(f"Question {i + 1}: {question.question}")
