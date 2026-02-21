from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import sys
from io import StringIO
import traceback

# Gemini
# from google import genai
# from google.genai import types

import google.generativeai as genai
from pydantic import BaseModel
from typing import List

# ---------------- App ----------------
app = FastAPI(title="Code Interpreter with AI Error Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Request / Response Models ----------------
class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    error: List[int]
    result: str

class ErrorAnalysis(BaseModel):
    error_lines: List[int]

# ---------------- Tool: Execute Python Code ----------------
def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact stdout or traceback.
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code, {})
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout

# ---------------- AI Error Analysis (ONLY on error) ----------------
# def analyze_error_with_ai(code: str, tb: str) -> List[int]:
#     client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

#     prompt = f"""
# Analyze the Python code and traceback below.
# Identify the exact line number(s) in the CODE where the error occurred.

# CODE:
# {code}

# TRACEBACK:
# {tb}

# Return only the line numbers.
# """

#     response = client.models.generate_content(
#         model="gemini-2.0-flash-exp",
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             response_mime_type="application/json",
#             response_schema=types.Schema(
#                 type=types.Type.OBJECT,
#                 properties={
#                     "error_lines": types.Schema(
#                         type=types.Type.ARRAY,
#                         items=types.Schema(type=types.Type.INTEGER),
#                     )
#                 },
#                 required=["error_lines"],
#             ),
#         ),
#     )

#     parsed = ErrorAnalysis.model_validate_json(response.text)
#     return parsed.error_lines

def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
Analyze the Python code and traceback below.
Identify the exact line number(s) in the CODE where the error occurred.

CODE:
{code}

TRACEBACK:
{tb}

Return only the line numbers as a JSON list under key "error_lines".
Example:
{{ "error_lines": [3] }}
"""

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={"response_mime_type": "application/json"},
    )

    response = model.generate_content(prompt)

    data = response.candidates[0].content.parts[0].text
    parsed = ErrorAnalysis.model_validate_json(data)
    return parsed.error_lines

# ---------------- API Endpoint ----------------
@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(req: CodeRequest):
    execution = execute_python_code(req.code)

    # ✅ No error → no AI call
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    # ❌ Error → AI analysis
    try:
        lines = analyze_error_with_ai(req.code, execution["output"])
    except Exception:
        # Safety fallback if AI fails
        lines = []

    return {
        "error": lines,
        "result": execution["output"],
    }