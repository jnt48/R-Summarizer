from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise Exception("GOOGLE_API_KEY is not set in your environment variables.")

# Configure Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize FastAPI app
app = FastAPI(title="Edith AI Service", version="1.0.0", description="Edith: Your smart generative AI assistant")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class TextData(BaseModel):
    text: str

def generate_content(prompt: str) -> str:
    response = model.generate_content([prompt])
    return response.text.strip()

# ============================================
# 1️⃣ Summarize Text Route
# ============================================
@app.post("/edith/summarize")
def summarize_text(data: TextData):
    prompt = (
        "Summarize the following text concisely, highlighting key points and main ideas:\n\n"
        + data.text
    )
    try:
        summary = generate_content(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")
    
    return {"summary": summary}

# ============================================
# 2️⃣ Abuse Check Route
# ============================================
@app.post("/edith/abuse-check")
def check_abuse(data: TextData):
    prompt = (
        "Analyze the following text. If it contains abusive or offensive language, respond with only 'BAD'. "
        "If it is clean and safe, respond with only 'GOOD'. No extra words:\n\n"
        + data.text
    )
    try:
        result = generate_content(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking abuse: {str(e)}")
    
    result = result.strip().upper()
    if "BAD" in result:
        return {"result": "BAD"}
    else:
        return {"result": "GOOD"}

# ============================================
# 3️⃣ Fact Check Route
# ============================================
@app.post("/edith/fact-check")
def fact_check(data: TextData):
    prompt = (
        "Carefully fact-check the following text. "
        "Respond with a verdict (True, False, or Mixed) and include suggestions or corrections if needed:\n\n"
        + data.text
    )
    try:
        result = generate_content(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fact-checking: {str(e)}")
    
    return {"fact_check_result": result}

# ============================================
# 4️⃣ Ask AI Question Route
# ============================================
@app.post("/edith/ask")
def ask_ai(data: TextData):
    prompt = (
        "You are Edith, a helpful AI assistant. Please answer the following user question clearly and helpfully:\n\n"
        + data.text
    )
    try:
        answer = generate_content(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")
    
    return {"answer": answer}

# ============================================
# 5 Improve-Question Route
# ============================================

@app.post("/edith/improve-question")
def improve_question(data: TextData):
    prompt = (
        "Analyze the following question text and suggest improvements to make it clearer and more detailed. "
        "Suggest better phrasing, additional details if missing, and tag suggestions:\n\n"
        + data.text
    )
    try:
        improvement = generate_content(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error improving question: {str(e)}")
    
    return {"improvement_suggestions": improvement}


# ============================================
# Run with uvicorn if needed
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)