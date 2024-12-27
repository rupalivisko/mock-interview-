# Mock-Test-Interview in Mistral-7b -instruct-v0.1
Required python libraries and packages
torch==2.5.1
python-dotenv==1.0.1
llama-index==0.9.47
llama_cpp_python==0.2.44
transformers==4.46.3
langchain-community==0.3.7

issues

'!CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.44 --force-reinstall --no-cache-dir'
this commamd shows this error - 'CMAKE_ARGS' is not recognized as an internal or external command,operable program or batch file.
when run this in local system but work fine in Google colab 

solution
to resovle this issue or run this in local system install this separately 'pip install llama-cpp-python==0.2.44'
