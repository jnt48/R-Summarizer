from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the Google API Key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise Exception("GOOGLE_API_KEY is not set in your environment variables.")

# Configure Gemini AI with the API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize FastAPI app
app = FastAPI(title="Blog Summary API", version="1.0.0")

# Enable CORS to allow frontend to interact with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data model for blog data input
class BlogData(BaseModel):
    id: str
    title: str
    slug: str
    excerpt: str
    content: str
    coverImage: str
    author: str  # Assuming a simple string; adjust as needed
    categories: List[str]  # List of category names
    publishedAt: str
    readTime: int
    likes: int
    comments: List[str]  # Simplified to a list of comment strings

def generate_blog_summary(prompt: str) -> str:
    """
    Uses Gemini AI to generate a concise summary of a blog post.
    """
    # Construct a prompt that instructs the model to provide a clear summary.
    full_prompt = (
        "Please provide a concise and insightful summary for the following blog post. "
        "Include key points such as the main ideas, value propositions, and any unique insights:\n" + prompt
    )
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([full_prompt])
    return response.text.strip()

@app.post("/summarize")
def summarize_blog(blog: BlogData):
    """
    Endpoint to generate a summary for a given blog post.
    """
    # Build a text block with key details from the blog to pass to Gemini AI
    blog_text = (
        f"Title: {blog.title}\n"
        f"Excerpt: {blog.excerpt}\n"
        f"Content: {blog.content}\n"
        f"Author: {blog.author}\n"
        f"Published At: {blog.publishedAt}\n"
        f"Read Time: {blog.readTime} minutes\n"
        f"Likes: {blog.likes}\n"
    )
    try:
        summary = generate_blog_summary(blog_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")
    
    return {"summary": summary}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
