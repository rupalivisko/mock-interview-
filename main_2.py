
from flask import Flask, request, jsonify
import logging
import sys

# Assuming all necessary imports from llama_index and llama_cpp are in place
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt

# Initialize Flask app
app = Flask(__name__)

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

# Define the route for interview generation
@app.route('/generate_interview', methods=['POST'])
def generate_interview():
    try:
        # Get request data (question details)
        data = request.get_json()

        # Ensure the request contains necessary keys
        if 'introductory_message' not in data:
            return jsonify({'error': 'Introductory message missing from request'}), 400

        # Customize the interview based on the provided introductory message
        user_input = data.get('introductory_message', '')

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
        return jsonify({'response': response.text}), 200

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

