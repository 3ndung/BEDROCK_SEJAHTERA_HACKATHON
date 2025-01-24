import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.middleware.sessions import SessionMiddleware
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from datetime import datetime
import re

# for calling function 
import requests
from langchain.tools import tool
#from langchain.agents import initialize_agent, Tool

################ function ######################################

@tool
def get_payment_list(customer_id: str) -> dict:
    """
    Retrieves payment information for a given customer ID, returning only selected fields.

    Args:
        customer_id (str): The ID of the customer.

    Returns:
        dict: The response from the API containing selected payment details.
    """
    url = ""
    headers = {
        "Authorization": "",
        "Content-Type": "application/json",
        "Cookie": "",
    }
    payload = {"customer_id": customer_id}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_data = response.json()

        # Extract specific fields from the response
        if isinstance(response_data, dict) and "result" in response_data:
            result = response_data["result"]
            selected_fields = {
                "so_no": result.get("so_no"),
                "customer_id": result.get("customer_id"),
                "payment_amount": result.get("payment_amount"),
                "payment_status": result.get("payment_status"),
                "payment_date": result.get("payment_date"),
                "payment_time": result.get("payment_time"),
            }
            parsed_response = (
                f"result: {selected_fields}\n"
                f"message: {response_data.get('message', 'No message')}\n"
                f"status: {response_data.get('status', False)}"
            )
        else:
            parsed_response = "Unexpected response format"

        return parsed_response
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}



def cariix(wo_id: str) -> dict:
    """
    Retrieves information about a Work Order (WO) status.

    Args:
        wo_id (str): The ID of the Work Order.

    Returns:
        dict: The response from the API containing work order details or an error message.
    """
    url = ''
    headers = {
        'Authorization': ''
    }
    params = {'wo_id': wo_id}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        try:
            response_json = response.json()  # Attempt to parse JSON
        except ValueError:
            return {"error": "Invalid JSON response from server"}
        
        # Extract data safely
        result = {
            'wo_id': response_json.get("result", {}).get("wo_id", ""),
            'wo_status': response_json.get("result", {}).get("wo_status", ""),
            'so': response_json.get("result", {}).get("so_no", "-"),
            'wo_last_update': response_json.get("result", {}).get("wo_last_update", "")
        }
        return result
    
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP error occurred: {http_err}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error occurred"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"An error occurred: {req_err}"}



################ End function ##################################

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Add session middleware for managing chat history
app.add_middleware(
    SessionMiddleware,
    secret_key="",
    session_cookie="session_cookie"
)

BUCKET_NAME = ""

aws_access_key_id=""
aws_secret_access_key="/s"
aws_session_token=""



# AWS credentials (replace with environment variables in production)
AWS_ACCESS_KEY_ID = aws_access_key_id
AWS_SECRET_ACCESS_KEY = aws_secret_access_key
AWS_SESSION_TOKEN = aws_session_token
BUCKET_NAME = BUCKET_NAME

# Initialize AWS clients
try:
    s3_client = boto3.client(
        service_name="s3",
        region_name="us-west-2",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )

    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )
except ClientError as e:
    raise Exception(f"Failed to create AWS client: {e}")

# Initialize Bedrock embeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock_client
)

# Temporary folder path for downloaded files
folder_path = "/tmp/"

# Load FAISS index from S3
def load_index():
    s3_client.download_file(BUCKET_NAME, "my_faiss.faiss", f"{folder_path}my_faiss.faiss")
    s3_client.download_file(BUCKET_NAME, "my_faiss.pkl", f"{folder_path}my_faiss.pkl")

# Get LLM instance
def get_llm():
    return BedrockLLM(
        model_id="meta.llama3-70b-instruct-v1:0",
        client=bedrock_client
    )

# Generate response from question
def get_response(llm, vectorstore, question):
    prompt_template = """
    Human: Please use the given context to provide a concise answer to the question.
    If you don't know the answer, just say that you don't know; don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    #answer = qa({"query": question})
    answer = qa.invoke({"query": question})

    return answer["result"]

# Startup event to load FAISS index
@app.on_event("startup")
async def startup_event():
    load_index()

# Process user question
@app.post("/", response_class=HTMLResponse)
async def process_question(request: Request, question: str = Form(...)):
    try:
        question = question.strip()
        nasi = ""
        if not question:
            raise ValueError("Please enter a valid question.")

        if "chat_history" not in request.session:
            request.session["chat_history"] = []

        faiss_index = FAISS.load_local(
            index_name="my_faiss",
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True,
        )

        llm = get_llm()
        answer = get_response(llm, faiss_index, question)

        if "so" in question.lower() and 'verified unpaid' in question.lower():
            match = re.search(r'\bso (\d+)\b', question)
            match = "60"+str(match.group(1))
            print(match)

            customer_id = match  # Extracted or predefined customer_id
            payment_data = get_payment_list(customer_id)
            nasi += f"Wait Sekejap ya --->\n\nPayment Details: {payment_data}"

        elif "WO" in question and 'status' in question.lower():
            match = re.search(r'WO-\w+-\d{6}-\d{4}', question)
            print(f'{match} #INI RESPONSE 1')
            match = str(match.group(0))
            print(match)

            wo_id = match  # Extracted or predefined customer_id
            wo = cariix(wo_id)
            nasi += f"Wait Sekejap ya --->\n\nWO Status: {wo}"    

        # WO-\w+-\d{6}-\d{4}    

        chat_history = request.session["chat_history"]
        chat_history.append({"role": "user", "content": question, "timestamp": datetime.now().strftime("%H:%M")})
        chat_history.append({"role": "assistant", "content": answer, "timestamp": datetime.now().strftime("%H:%M")})
        chat_history.append({"role": "assistant", "content": nasi, "timestamp": datetime.now().strftime("%H:%M")})
        #try:
        #    pass
        #    #if len(nasi) !=0 :
        #    #    chat_history.append({"role": "assistant", "content": nasi, "timestamp": datetime.now().strftime("%H:%M")})
        #except:
        #    pass        

        request.session["chat_history"] = chat_history

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "answer": answer, "question": question, "chat_history": chat_history},
        )
    except ValueError as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e), "chat_history": request.session.get("chat_history", [])},
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "An unexpected error occurred. Please try again.", "chat_history": []},
            status_code=500,
        )

# Display the chat interface
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if "chat_history" not in request.session:
        request.session["chat_history"] = []
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "answer": None, "question": None, "chat_history": request.session["chat_history"]},
    )

# Clear chat history
@app.post("/clear", response_class=HTMLResponse)
async def clear_history(request: Request):
    request.session["chat_history"] = []
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "answer": None, "question": None, "chat_history": []},
    )
