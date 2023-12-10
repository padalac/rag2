get_question_template = """\
You are a University Professor creating a test for advanced students. For each context, create a question that is specific to the context. Avoid creating generic or general questions.

question: A question about the context.

Format the output as JSON with the following keys:
question

context: {context}
"""

get_answer_template = """\
You are a University Professor creating a test for advanced students. For each question and context, create an answer.

answer: answer the question using the context.

Format the output as JSON with the following keys:
answer

question: {question}
context: {context}
"""