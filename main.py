# import torch
# from llama_index.llms import LlamaCPP
# from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
# from llama_cpp import Llama

# # Initialize the LlamaCPP model with Mistral 7B
# llm = LlamaCPP(
#     model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q3_K_M.gguf',
#     model_path=None,  # No local path since we're using a URL
#     temperature=0.1,
#     max_new_tokens=256,
#     context_window=3900,  # Adjust according to the context window you need
#     generate_kwargs={},
#     model_kwargs={"n_gpu_layers": -1},  # Ensure this matches your hardware setup
#     messages_to_prompt=messages_to_prompt,
#     completion_to_prompt=completion_to_prompt,
#     verbose=True,
# )

# # Function to generate interview questions (adapted from your original prompt)
# def generate_questions(prompt):
#     response = llm(prompt)
#     return response

# # Generate question based on score logic
# def generate_interview_question(score):
#     if score <= 50:
#         prompt = """
#         Conduct a structured interview with a candidate consisting of a maximum of 20 questions divided into three sections: Generic Questions, Technical Questions, and Leadership Questions.
#         Start from the introduction and proceed with other questions.
#         The questions must be concise and under 30 words each.
        
#         Generic Questions:
#         Start with an introductory question about the candidate's background.
#         Follow with questions about their education, motivation, and relevant experiences.
        
#         Technical Questions:
#         Focus on specific skills relevant to the candidate's expertise such as programming languages or frameworks.
        
#         Leadership Questions:
#         Explore leadership experience only if the candidate’s profile indicates relevant experience.
#         """
#     else:
#         # Assuming 'previous_question' is a string from previous interview interaction
#         prompt = f"""
#         Generate the next interview question based on the previous question and the candidate's responses. 
#         The new question should explore related skills, knowledge, or experience to assess the candidate’s depth of understanding.
        
#         Previous Question: {previous_question}
#         """

#     question = generate_questions(prompt)
#     return question

# # Evaluate answer and assign a score (you can replace this with actual logic if required)
# def evaluate_answer(user_answer, question):
#     evaluation_prompt = f"""
#     Evaluate the following answer based on the interview question provided. 
#     Rate the answer's accuracy, relevance, and completeness as a percentage score.

#     Question: {question}
#     Answer: {user_answer}

#     Provide only the score as a percentage (e.g., 85%).
#     """
#     score = generate_questions(evaluation_prompt)  # You may refine this to handle responses appropriately
#     return score

# # Example usage
# if __name__ == '__main__':
#     score = 45  # Example score (can be dynamically adjusted based on logic)
#     question = generate_interview_question(score)
#     print(f"Generated Question: {question}")

#     # Assuming we have a user_answer
#     user_answer = "My experience with Python involves creating full-stack web applications using Flask."
#     evaluation_score = evaluate_answer(user_answer, question)
#     print(f"Answer Score: {evaluation_score}")



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
import logging
import sys

# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout)) 

# Llama model setup
llm = LlamaCPP(
    model_path="C:/Mock-Test-Interview/mistral-7b-instruct-v0.1.Q3_K_M.gguf",
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": -1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

# Request model for validation
class InterviewRequest(BaseModel):
    introductory_message: str

# Define the route for interview generation
@app.post("/generate_interview")
async def generate_interview(request: InterviewRequest):
    try:
        # Customize the interview based on the provided introductory message
        user_input = request.introductory_message

        # Construct the prompt for Llama model
        prompt = f"""
        Conduct a structured interview with a candidate consisting of a maximum of 20 questions divided into three sections: Generic Questions, Technical Questions, and Leadership Questions.
        Make sure you start with an introductory part and then move to other questions.
        Ensure the questions you generate are concise and under 30 words each.

        Generic Questions:
        - Start with an introductory question about the candidate's background.
        - Follow with questions about their education, motivation for their field, and relevant experiences.
        - Adapt subsequent questions based on their responses, ensuring clarity and brevity.

        Technical Questions:
        - Focus on specific skills relevant to the candidate's expertise such as programming languages or frameworks.
        - Ask them to explain their experience with particular projects or tools.
        - Use concise follow-up questions based on their answers to explore their knowledge further.

        Leadership Questions:
        - Explore their leadership style and experience managing teams, only if their profile indicates relevant experience.
        - Inquire about specific challenges they faced in leadership roles and how they addressed them.
        - Maintain engagement with concise follow-up questions as needed.

        The goal is to create an engaging and responsive interview that allows the candidate to showcase their knowledge and experience while keeping the conversation fluid and relevant.
        """

        # Add user's introductory message for dynamic prompts
        prompt += f"\nIntroductory Message: {user_input}"

        # Generate response from Llama model
        response = llm.complete(prompt)

        # Return the generated response
        return {"response": response.text}

    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))