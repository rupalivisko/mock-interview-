# from flask import Flask, request, jsonify,session
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import ConversationChain
# from datetime import timedelta
# import secrets


# app = Flask(__name__)

# app.secret_key = secrets.token_hex(16) 
# app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=15)
# api_key = "AIzaSyBHAzGIkIzv1HyeKyqRMLUdc0_Vs1cRXlY"

# #  Langchain intigrate.
# llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0.7, google_api_key=api_key)
# conversation_chain = ConversationChain(llm=llm)


# # generate question from prompt.
# def generate_questions(prompt):
#     response = conversation_chain.run(input=prompt)
#     return response


# # Genarte question from post api.   
# @app.route("/generate_question", methods=["POST"])
# def generate_question():
#     score=session.get("score",0)
 
#     data = request.get_json()
#     name = data.get("user_name", "")
#     designation = data.get("user_profile", "")
#     experience = data.get("user_experience_ext", "")
#     education = data.get("user_qualification_ext", "")
#     skills = data.get("user_skills", "")
#     experties= data.get("user_expertise", "")
#     if "count_answer"  not in session:
#         session["count_answer"]=0
   
#     # Prompt for generate question.
#     if score<=50:
#         prompt = f"""
#         Conduct a structured interview with a candidate consisting of a maximum of 20 questions divided into three sections: Generic Questions, Technical Questions, and Leadership Questions.
#         Make sure you Start from intoductive part and then move to other questions.
#         Make sure the question you generate should not be in large chunk's must be under 30 words.

#         Generic Questions:

#         Start with an introductory question about the candidate's background.
#         Follow with questions about their education, motivation for their field, and relevant experiences.
#         Adapt subsequent questions based on their responses ensuring clarity and brevity without punctuation.
#         Technical Questions:

#         Focus on specific skills relevant to the candidate's expertise such as programming languages or frameworks.
#         Ask them to explain their experience with particular projects or tools.
#         Use concise follow-up questions based on their answers to explore their knowledge further.
#         Leadership Questions:

#         Explore their leadership style and experience managing teams only if their profile indicates relevant experience.
#         Inquire about specific challenges they faced in leadership roles and how they addressed them.
#         Maintain engagement with concise follow-up questions as needed.


#         The goal is to create an engaging and responsive interview that allows the candidate to showcase their knowledge and experience while keeping the conversation fluid and relevant.
#         """
#     else:
        
#         # Prompt for generate question according to previous question.
#         prompt = f"""Generate the next interview question based on the previous question and the candidate's responses. The new question should explore related skills, knowledge, or experience to assess the candidate’s depth of understanding in a specific area.

#         Previous Question: {session.get("question", "")}

#         Please make sure the new question builds upon this context, delving deeper into topics relevant to the candidate’s role. Only provide a single, relevant follow-up question."""
   
#     question = generate_questions(prompt)
#     session['question'] = question  


#     return jsonify({"question": question})



# def answer_score(user_answer):
#     question = session.get("question", "")
    
#     if not question:
#         return {"error": "No question found in session"}
    
#     # Evaluation prompt
#     evaluation_prompt = f"""
#     Evaluate the following answer based on the interview question provided. Rate the answer's accuracy (while considering the generic response from human), relevance, and completeness as a percentage score.

#     Question: {question}
#     Answer: {user_answer}

#     Provide only the score as a percentage (e.g., 85%) without any extra text.
#     """
    
#     # Get evaluation response from model
#     evaluation_response = conversation_chain.run(input=evaluation_prompt)
    
#     # Convert response to an integer score, handle any ValueError
#     try:
#         score = int(evaluation_response.strip().replace("%", ""))
#     except ValueError:
#         score = 0  
    
#     # Store score in session
#     session['score'] = score
    
#     return {"score": score}


# # Get Answer score According to question 
# @app.route("/answer_score", methods=["POST"])
# def user_answer_score():
#     data = request.get_json()
#     user_answer = data.get("answer", "")
#     session["count_answer"]+=1
#     session['answer'] = user_answer 
#     result = answer_score(user_answer)
#     print(result["score"])
#     if "result_score" not in session:
#         session["result_score"] = []
    
#     session["result_score"].append(result["score"])
   

#     return jsonify(result)


# # Show Result and Total Answers.
# @app.route('/result', methods=["GET"])
# def result():
#     result= session.get("result_score", "")
#     total_answer= session.get("count_answer")
#     if not result:
#         overall_score = 0  
#     else:
#         total = sum(result)
#         length = len(result)
#         overall_score = total / length
#     session.clear()
#     return jsonify({"user_score": overall_score,"total_answer":total_answer})

# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, request, jsonify, session
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datetime import timedelta
# import secrets
# import torch
# import os

# app = Flask(__name__)

# # Secret key for session management
# app.secret_key = secrets.token_hex(16)
# app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=15)

# # Local model path (replace with the actual path to your local model directory)
# model_path = "mistral-7b-instruct-v0.1.Q3_K_M.gguf"  # Update this with the correct path to your model

# # Check if the local model path exists
# if not os.path.exists(model_path):
#     print(f"Error: Model path {model_path} does not exist.")
#     model_loaded = False
# else:
#     try:
#         # Specify device (CPU or GPU if available)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Load the tokenizer and model from the local path (no Hugging Face token needed)
#         tokenizer = AutoTokenizer.from_pretrained(model_path)  # Local path to tokenizer
#         model = AutoModelForCausalLM.from_pretrained(model_path).to(device)  # Local path to model
#         model_loaded = True
#         print("Model loaded successfully.")
#     except Exception as e:
#         print("Error loading model:", str(e))
#         model_loaded = False

# # Function to generate questions using the local Mistral model
# def generate_questions(prompt):
#     if not model_loaded:
#         return "Model not loaded. Please check your setup."
    
#     # Tokenize the input and send to device
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_length=50)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# # Endpoint to generate interview questions
# @app.route("/generate_question", methods=["POST"])
# def generate_question():
#     score = session.get("score", 0)
    
#     if "count_answer" not in session:
#         session["count_answer"] = 0
    
#     # Prompt for generating question
#     if score <= 50:
#         prompt = """
#         Conduct a structured interview with a candidate consisting of a maximum of 20 questions divided into three sections: Generic Questions, Technical Questions, and Leadership Questions.
#         Start with an introductory question, then proceed to others.
#         Each question should be under 30 words.
        
#         Generic Questions:
#         Begin with an introductory question about the candidate's background.
#         Follow with questions about education, motivation, and relevant experiences.
        
#         Technical Questions:
#         Focus on skills relevant to their expertise, like programming languages or frameworks.
#         Ask about experience with projects or tools.
        
#         Leadership Questions:
#         Explore leadership style and experience, if relevant.
#         Inquire about challenges in leadership roles.
        
#         The goal is to create an engaging interview that lets the candidate showcase knowledge while keeping the conversation fluid.
#         """
#     else:
#         # Prompt for generating follow-up question
#         prompt = f"""Generate the next interview question based on the previous question and the candidate's responses. The new question should explore related skills, knowledge, or experience to assess the candidate’s depth of understanding in a specific area.
        
#         Previous Question: {session.get("question", "")}

#         Provide a single, relevant follow-up question."""
    
#     question = generate_questions(prompt)
#     session['question'] = question

#     return jsonify({"question": question})

# # Function to evaluate the answer
# def answer_score(user_answer):
#     question = session.get("question", "")
    
#     if not question:
#         return {"error": "No question found in session"}
    
#     # Evaluation prompt
#     evaluation_prompt = f"""
#     Evaluate the following answer based on the interview question provided. Rate the answer's accuracy, relevance, and completeness as a percentage score.

#     Question: {question}
#     Answer: {user_answer}

#     Provide only the score as a percentage (e.g., 85%) without any extra text.
#     """
    
#     evaluation_response = generate_questions(evaluation_prompt)
    
#     # Convert response to an integer score, handle any ValueError
#     try:
#         score = int(evaluation_response.strip().replace("%", ""))
#     except ValueError:
#         score = 0  
    
#     # Store score in session
#     session['score'] = score
    
#     return {"score": score}

# # Endpoint to evaluate user answer score
# @app.route("/answer_score", methods=["POST"])
# def user_answer_score():
#     data = request.get_json()
#     user_answer = data.get("answer", "")
#     session["count_answer"] += 1
#     session['answer'] = user_answer 
#     result = answer_score(user_answer)
    
#     if "result_score" not in session:
#         session["result_score"] = []
    
#     session["result_score"].append(result["score"])

#     return jsonify(result)

# # Endpoint to show final result and total answers
# @app.route('/result', methods=["GET"])
# def result():
#     result = session.get("result_score", [])
#     total_answer = session.get("count_answer", 0)
#     overall_score = sum(result) / len(result) if result else 0
#     session.clear()
#     return jsonify({"user_score": overall_score, "total_answer": total_answer})

# if __name__ == '__main__':
#     app.run(debug=True)




# import torch
# import os
# import requests
# import llama_index.llms
# from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt

# # Define the URL for the Mistral-7B model
# model_url = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q3_K_M.gguf'

# # Define a local path where the model will be saved
# model_path = './mistral-7b-instruct-v0.1.gguf'

# # Download the model if it's not already present locally
# if not os.path.exists(model_path):
#     print("Downloading Mistral-7B model...")
#     response = requests.get(model_url, stream=True)
    
#     # Check if the request was successful
#     if response.status_code == 200:
#         with open(model_path, 'wb') as f:
#             for chunk in response.iter_content(chunk_size=1024):
#                 if chunk:
#                     f.write(chunk)
#         print(f"Model downloaded successfully to {model_path}")
#     else:
#         print(f"Error downloading model: {response.status_code}")
# else:
#     print(f"Model already exists at {model_path}")

# # Set up the LlamaCPP model
# llm = LlamaCPP(
#     model_url=None,  # Not needed since we are using a local path
#     model_path=model_path,  # Provide the local path where model is saved
#     temperature=0.1,
#     max_new_tokens=256,
#     context_window=3900,
#     generate_kwargs={},
#     model_kwargs={"n_gpu_layers": -1},
#     messages_to_prompt=messages_to_prompt,
#     completion_to_prompt=completion_to_prompt,
#     verbose=True,
# )

# # Example prompt to test the model
# prompt = "What is the capital of France?"

# # Generate the response
# response = llm(prompt)
# print("Response from Mistral-7B:", response)





from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Local path to the downloaded model (adjust as needed)
model_path = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q3_K_M.gguf' 

# Load model and tokenizer from the local path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example prompt to test the model
prompt = "What is the capital of France?"

# Tokenize the input and generate a response
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Response from Mistral-7B:", response)
