"""
Alternate version of main.py that uses the improved RAG implementation
"""
print("Running:", __file__)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import requests
requests.packages.urllib3.disable_warnings()
requests.sessions.Session.verify = False

import traceback
from flask import Flask, request, jsonify, render_template, Response, send_from_directory, session
import json
import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from flask import Flask, request, jsonify, render_template_string, Response, send_from_directory, session

load_dotenv()


def get_sas_token():
    token = os.getenv("SAS_TOKEN")
    if not token:
        from generate_sas_token import generate_sas_token
        token = generate_sas_token()
        os.environ["SAS_TOKEN"] = token
        print("Generated new SAS token")
        print(f"SAS_TOKEN={token}")
    return token

# Import the improved RAG implementation
from rag_assistant_v2 import FlaskRAGAssistantV2
from db_manager import DatabaseManager
from openai import AzureOpenAI
from config import get_cost_rates
from openai_service import OpenAIService
from rag_improvement_logging import setup_improvement_logging

# Set up dedicated logging for the improved implementation
logger = setup_improvement_logging()

# Add file handler with absolute path for main application logs
file_handler = logging.FileHandler('logs/main_alternate.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream logs to stdout for visibility
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.info("Alternate Flask RAG application starting up with improved procedural content handling")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key-for-sessions")


# Dictionary to store RAG assistant instances by session ID
rag_assistants = {}

# Function to get or create a RAG assistant for a session
def get_rag_assistant(session_id):
    """Get or create a RAG assistant for the given session ID"""
    if session_id not in rag_assistants:
        logger.info(f"Creating new RAG assistant for session {session_id}")
        rag_assistants[session_id] = FlaskRAGAssistantV2()
    return rag_assistants[session_id]

@app.route("/", methods=["GET"])
def index():
    logger.info("Index page accessed")
    # Generate a session ID if one doesn't exist
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
        logger.info(f"New session created: {session['session_id']}")
    
    token = get_sas_token()
    return render_template("index.html", sas_token=token)

@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json()
    logger.info("DEBUG - Incoming /api/query payload: %s", json.dumps(data))
    user_query = data.get("query", "")
    is_enhanced = data.get("is_enhanced", False)
    logger.info(f"API query received: {user_query}")
    
    # Get the session ID
    session_id = session.get('session_id')
    if not session_id:
        session_id = os.urandom(16).hex()
        session['session_id'] = session_id
        logger.info(f"Created new session ID: {session_id}")
    
    # Extract any settings from the request
    settings = data.get("settings", {})
    logger.info(f"DEBUG - Request settings: {json.dumps(settings)}")
    
    try:
        # Get or create the RAG assistant for this session
        rag_assistant = get_rag_assistant(session_id)
        
        # Update settings if provided
        if settings:
            for key, value in settings.items():
                if hasattr(rag_assistant, key):
                    setattr(rag_assistant, key, value)
            
            # If model is updated, update the deployment name
            if "model" in settings:
                rag_assistant.deployment_name = settings["model"]
        
        logger.info(f"DEBUG - Using model: {rag_assistant.deployment_name}")
        logger.info(f"DEBUG - Temperature: {rag_assistant.temperature}")
        logger.info(f"DEBUG - Max tokens: {rag_assistant.max_completion_tokens}")
        logger.info(f"DEBUG - Top P: {rag_assistant.top_p}")
        
        answer, cited_sources, _, evaluation, context = rag_assistant.generate_rag_response(user_query, is_enhanced=is_enhanced)
        logger.info(f"API query response generated for: {user_query}")
        logger.info(f"DEBUG - Response length: {len(answer)}")
        logger.info(f"DEBUG - Number of cited sources: {len(cited_sources)}")
        
        try:
            # Log using existing log_rag_query method by unpacking relevant fields
            vote_id = DatabaseManager.log_rag_query(
                query=user_query,
                response=answer,
                sources=cited_sources,
                context=context,
                sql_query=None
            )
            logger.info(f"RAG query logged with ID: {vote_id}")
        except Exception as log_exc:
            logger.error(f"Failed to log RAG query: {log_exc}", exc_info=True)
        
        return jsonify({
            "answer": answer,
            "sources": cited_sources,
            "evaluation": evaluation
        })
    except Exception as e:
        logger.error(f"Error in api_query: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/api/clear_history", methods=["POST"])
def api_clear_history():
    """Clear the conversation history for the current session"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in rag_assistants:
            logger.info(f"Clearing conversation history for session {session_id}")
            rag_assistants[session_id].clear_conversation_history()
            return jsonify({"success": True})
        else:
            logger.warning(f"No active session found to clear history")
            return jsonify({"success": True, "message": "No active session found"})
    except Exception as e:
        logger.error(f"Error clearing conversation history: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

# Serve static files from the 'static' folder
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

# Serve static files from the 'assets' folder
@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory("assets", filename)

# Feedback submission endpoint
@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json()
    logger.debug("Incoming feedback payload: %s", json.dumps(data))
    try:
        vote_id = DatabaseManager.save_feedback(data)
        logger.info(f"Feedback saved with ID: {vote_id}")
        # Fallback: write feedback to local JSON file
        with open("logs/feedback_fallback.jsonl", "a") as f:
            f.write(json.dumps({"vote_id": vote_id, **data}) + "\n")
        return jsonify({"success": True, "vote_id": vote_id})
    except Exception as e:
        logger.error("Error saving feedback: %s", str(e), exc_info=True)
        # Fallback: write raw feedback to local JSON file
        with open("logs/feedback_fallback.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Improved Flask RAG application')
    parser.add_argument('--port', type=int, default=int(os.environ.get("PORT", 5004)),
                        help='Port to run the server on (default: 5004)')
    args = parser.parse_args()
    
    port = args.port
    logger.info(f"Starting Improved Flask app on port {port}")

    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
