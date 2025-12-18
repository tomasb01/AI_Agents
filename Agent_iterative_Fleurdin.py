"""
FLEURDIN AI - Iterativní RAG Agent
===================================

Inteligentní agent pro aromatherapii s multi-step reasoning.

Workflow:
1. Mandatory Clarification - zjistí problém, příčinu, symptomy
2. VectorSearch Loop (max 3x) - hledá v databázi esenciálních olejů
3. TavilySearch Fallback (max 3x) - web search při selhání
4. Final Answer nebo Apology + Email notification

Author: Claude Code + Tomáš
Date: 2025-12-17
"""

import json
import os
import sys
import smtplib
import numpy as np
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from sentence_transformers import SentenceTransformer

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ============================================
# CONFIGURATION
# ============================================

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

# Tavily Configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Gmail SMTP Configuration
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

# Data Configuration
DATA_PATH = "/Users/atlas/Projects/Fleurdin_AI/4-RAG_Pipeline/chunked_data_FIXED.json"

# Agent Configuration
MAX_VECTOR_ITERATIONS = 3
MAX_TAVILY_ITERATIONS = 3
RELEVANCE_THRESHOLD = 0.6  # Minimum cosine similarity score

# Embedding Model
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ============================================
# STATE DEFINITION
# ============================================

class AgentState(TypedDict):
    """State pro LangGraph workflow"""
    # User input
    original_question: str

    # Clarification
    has_problem: bool
    has_cause: bool
    has_symptoms: bool
    problem: str
    cause: str
    symptoms: str
    clarified_question: str

    # Vector search
    vector_search_attempts: List[Dict[str, Any]]
    vector_iteration: int
    vector_satisfied: bool

    # Tavily search
    tavily_search_attempts: List[Dict[str, Any]]
    tavily_iteration: int
    tavily_satisfied: bool

    # Results
    best_docs: List[Dict[str, Any]]
    web_context: str
    final_answer: str

    # Status
    status: str  # "success" | "failed"
    conversation_log: List[str]

# ============================================
# HELPER FUNCTIONS
# ============================================

def load_vector_data():
    """Načte chunked data z JSON souboru"""
    print(f"📂 Loading data from: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: File not found: {DATA_PATH}")
        sys.exit(1)

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = data['chunks']
    print(f"✅ Loaded {len(chunks)} chunks")
    return chunks

def vector_search(query: str, chunks: List[dict], top_k: int = 5):
    """
    Semantic search v chunk datech pomocí cosine similarity

    Returns:
        List of tuples: [(chunk, score), ...]
    """
    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Generate query embedding
    query_embedding = model.encode(query)

    # Calculate similarities
    results = []
    for chunk in chunks:
        chunk_embedding = np.array(chunk['embedding'])
        similarity = cosine_similarity(
            [query_embedding],
            [chunk_embedding]
        )[0][0]
        results.append((chunk, float(similarity)))

    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]

def send_email_notification(state: AgentState):
    """Odešle email notifikaci při selhání agenta"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = "Fleurdin AI - Nepodařilo se najít odpověď"

        # Email body
        body = f"""
Dobrý den,

Agent nebyl schopen najít uspokojivou odpověď na dotaz uživatele.

═══════════════════════════════════════════════════════════

DOTAZ UŽIVATELE:
{state['original_question']}

UPŘESNĚNÉ INFORMACE:
- Problém: {state.get('problem', 'N/A')}
- Příčina: {state.get('cause', 'N/A')}
- Symptomy: {state.get('symptoms', 'N/A')}

═══════════════════════════════════════════════════════════

VYZKOUŠENÉ POKUSY:

VectorSearch ({len(state['vector_search_attempts'])}x):
"""

        for i, attempt in enumerate(state['vector_search_attempts'], 1):
            body += f"\n  {i}. Query: {attempt['query'][:100]}..."
            body += f"\n     Best score: {attempt['max_score']:.3f}"
            body += f"\n     Satisfied: {attempt['satisfied']}\n"

        body += f"\nTavilySearch ({len(state['tavily_search_attempts'])}x):\n"

        for i, attempt in enumerate(state['tavily_search_attempts'], 1):
            body += f"\n  {i}. Query: {attempt['query'][:100]}..."
            body += f"\n     Satisfied: {attempt['satisfied']}\n"

        body += "\n═══════════════════════════════════════════════════════════\n\n"

        # Add best results found
        if state['best_docs']:
            body += "NEJLEPŠÍ NALEZENÉ VÝSLEDKY:\n\n"
            for i, (doc, score) in enumerate(state['best_docs'][:3], 1):
                body += f"{i}. {doc.get('name', 'Unknown')} (score: {score:.3f})\n"
                body += f"   {doc.get('text', '')[:200]}...\n\n"

        body += "\n---\nFleurdin AI Agent\n"
        body += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.send_message(msg)

        print("✅ Email notification sent successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

def save_conversation_log(state: AgentState):
    """Uloží conversation log do TXT souboru"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"conversation_log_{timestamp}.txt"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("FLEURDIN AI - CONVERSATION LOG\n")
            f.write("="*70 + "\n\n")

            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Status: {state['status']}\n\n")

            f.write("-"*70 + "\n")
            f.write("ORIGINAL QUESTION:\n")
            f.write("-"*70 + "\n")
            f.write(f"{state['original_question']}\n\n")

            f.write("-"*70 + "\n")
            f.write("CLARIFICATION:\n")
            f.write("-"*70 + "\n")
            f.write(f"Problém: {state.get('problem', 'N/A')}\n")
            f.write(f"Příčina: {state.get('cause', 'N/A')}\n")
            f.write(f"Symptomy: {state.get('symptoms', 'N/A')}\n")
            f.write(f"Clarified question: {state.get('clarified_question', 'N/A')}\n\n")

            f.write("-"*70 + "\n")
            f.write(f"VECTOR SEARCH ATTEMPTS ({len(state['vector_search_attempts'])}):\n")
            f.write("-"*70 + "\n")
            for i, attempt in enumerate(state['vector_search_attempts'], 1):
                f.write(f"\nAttempt {i}:\n")
                f.write(f"  Query: {attempt['query']}\n")
                f.write(f"  Max Score: {attempt['max_score']:.3f}\n")
                f.write(f"  Results Count: {len(attempt['results'])}\n")
                f.write(f"  Satisfied: {attempt['satisfied']}\n")

            f.write("\n" + "-"*70 + "\n")
            f.write(f"TAVILY SEARCH ATTEMPTS ({len(state['tavily_search_attempts'])}):\n")
            f.write("-"*70 + "\n")
            for i, attempt in enumerate(state['tavily_search_attempts'], 1):
                f.write(f"\nAttempt {i}:\n")
                f.write(f"  Query: {attempt['query']}\n")
                f.write(f"  Satisfied: {attempt['satisfied']}\n")

            f.write("\n" + "-"*70 + "\n")
            f.write("FINAL ANSWER:\n")
            f.write("-"*70 + "\n")
            f.write(f"{state['final_answer']}\n\n")

            f.write("="*70 + "\n")
            f.write("END OF LOG\n")
            f.write("="*70 + "\n")

        print(f"✅ Conversation log saved: {filename}")
        return filename

    except Exception as e:
        print(f"❌ Failed to save log: {e}")
        return None

# ============================================
# LANGRAPH NODES
# ============================================

# Initialize LLM
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# Load chunks
CHUNKS = load_vector_data()

def check_clarification_node(state: AgentState) -> AgentState:
    """
    Node 1: Zkontroluje, jestli dotaz obsahuje problém, příčinu a symptomy
    """
    print("\n" + "="*70)
    print("🔍 STEP 1: Checking clarification needs...")
    print("="*70)

    messages = [
        SystemMessage(content="""Analyzuj uživatelský dotaz a zkontroluj:
1. Je specifikován konkrétní problém/potíž? (např. "bolí mě hlava", "mám nespavost")
2. Je uvedena možná příčina? (např. "kvůli stresu", "po nemoci", "únava")
3. Jsou uvedeny další symptomy? (např. "a taky mě bolí v krku")

Odpověz POUZE ve formátu JSON bez dalšího textu:
{
  "has_problem": true/false,
  "has_cause": true/false,
  "has_symptoms": true/false,
  "problem": "popis problému nebo null",
  "cause": "popis příčiny nebo null",
  "symptoms": "popis symptomů nebo null"
}"""),
        HumanMessage(content=f"Dotaz: {state['original_question']}")
    ]

    response = llm.invoke(messages)

    # Parse JSON response
    try:
        result = json.loads(response.content)
        state['has_problem'] = result.get('has_problem', False)
        state['has_cause'] = result.get('has_cause', False)
        state['has_symptoms'] = result.get('has_symptoms', False)
        state['problem'] = result.get('problem', '')
        state['cause'] = result.get('cause', '')
        state['symptoms'] = result.get('symptoms', '')
    except json.JSONDecodeError:
        print("⚠️  Warning: Failed to parse LLM response, assuming clarification needed")
        state['has_problem'] = False
        state['has_cause'] = False
        state['has_symptoms'] = False

    print(f"\n📊 Analysis:")
    print(f"  Has problem: {state['has_problem']}")
    print(f"  Has cause: {state['has_cause']}")
    print(f"  Has symptoms: {state['has_symptoms']}")

    return state

def clarify_question_node(state: AgentState) -> AgentState:
    """
    Node 2: Interaktivně se ptá uživatele na upřesnění
    """
    print("\n" + "="*70)
    print("💬 STEP 2: Asking for clarification...")
    print("="*70)

    missing_info = []
    if not state['has_problem']:
        missing_info.append("konkrétní problém")
    if not state['has_cause']:
        missing_info.append("možná příčina")
    if not state['has_symptoms']:
        missing_info.append("další symptomy")

    print(f"\n❓ Rozumím vašemu dotazu, ale potřebuji více informací:")

    # Ask for problem
    if not state['has_problem']:
        problem = input("   • Jaký konkrétní problém řešíte? (např. bolest hlavy, nespavost): ").strip()
        state['problem'] = problem

    # Ask for cause
    if not state['has_cause']:
        cause = input("   • Víte, co může být příčinou? (např. stres, únava, nemoc): ").strip()
        state['cause'] = cause

    # Ask for symptoms
    if not state['has_symptoms']:
        symptoms = input("   • Máte i jiné obtíže? (pokud ne, napište 'ne'): ").strip()
        state['symptoms'] = symptoms if symptoms.lower() != 'ne' else ''

    # Create clarified question
    clarified_parts = [state['original_question']]

    if state['problem']:
        clarified_parts.append(f"Problém: {state['problem']}")
    if state['cause']:
        clarified_parts.append(f"Příčina: {state['cause']}")
    if state['symptoms']:
        clarified_parts.append(f"Další symptomy: {state['symptoms']}")

    state['clarified_question'] = ". ".join(clarified_parts)

    print(f"\n✅ Clarified question: {state['clarified_question']}")

    return state

def vector_search_node(state: AgentState) -> AgentState:
    """
    Node 3: Provede vector search v databázi
    """
    state['vector_iteration'] = state.get('vector_iteration', 0) + 1

    print("\n" + "="*70)
    print(f"🔎 STEP 3: Vector Search (Attempt {state['vector_iteration']}/{MAX_VECTOR_ITERATIONS})")
    print("="*70)

    query = state.get('clarified_question', state['original_question'])

    print(f"\n📝 Query: {query}")
    print(f"🔍 Searching in {len(CHUNKS)} chunks...")

    # Perform search
    results = vector_search(query, CHUNKS, top_k=5)

    # Extract scores
    max_score = results[0][1] if results else 0.0

    print(f"\n📊 Results:")
    print(f"  Found: {len(results)} documents")
    print(f"  Best score: {max_score:.3f}")
    print(f"  Relevance threshold: {RELEVANCE_THRESHOLD}")

    # Display top results
    print(f"\n🏆 Top results:")
    for i, (doc, score) in enumerate(results[:3], 1):
        print(f"  {i}. {doc.get('name', 'Unknown')} (score: {score:.3f})")

    # Store attempt
    attempt = {
        'iteration': state['vector_iteration'],
        'query': query,
        'results': [(doc, score) for doc, score in results],
        'max_score': max_score,
        'satisfied': False  # Will be updated in evaluation
    }

    if 'vector_search_attempts' not in state:
        state['vector_search_attempts'] = []
    state['vector_search_attempts'].append(attempt)

    # Store best docs
    state['best_docs'] = results

    return state

def evaluate_vector_node(state: AgentState) -> AgentState:
    """
    Node 4: LLM posoudí, jestli jsou výsledky z VectorSearch dostačující
    """
    print("\n" + "="*70)
    print("⚖️  STEP 4: Evaluating vector search results...")
    print("="*70)

    # Get last attempt results
    last_attempt = state['vector_search_attempts'][-1]
    results = last_attempt['results']
    max_score = last_attempt['max_score']

    # Check score threshold
    if max_score < RELEVANCE_THRESHOLD:
        print(f"\n❌ Score {max_score:.3f} < threshold {RELEVANCE_THRESHOLD}")
        print("   Results are not relevant enough")
        state['vector_satisfied'] = False
        last_attempt['satisfied'] = False
        return state

    # Prepare docs for LLM evaluation
    docs_text = "\n\n".join([
        f"Document {i+1} (score: {score:.3f}):\n{doc.get('text', '')[:300]}..."
        for i, (doc, score) in enumerate(results[:3])
    ])

    messages = [
        SystemMessage(content="""Jsi evaluátor kvality výsledků vyhledávání.
Posouď, jestli poskytnuté dokumenty obsahují dostatečné informace pro odpověď na dotaz uživatele.

Odpověz POUZE ve formátu JSON:
{
  "satisfied": true/false,
  "reason": "krátké zdůvodnění"
}"""),
        HumanMessage(content=f"""Dotaz uživatele: {state.get('clarified_question', state['original_question'])}

Nalezené dokumenty:
{docs_text}

Obsahují tyto dokumenty dostatečné informace pro kvalitní odpověď?""")
    ]

    response = llm.invoke(messages)

    try:
        result = json.loads(response.content)
        satisfied = result.get('satisfied', False)
        reason = result.get('reason', '')

        state['vector_satisfied'] = satisfied
        last_attempt['satisfied'] = satisfied

        print(f"\n📊 Evaluation:")
        print(f"  Satisfied: {satisfied}")
        print(f"  Reason: {reason}")

    except json.JSONDecodeError:
        print("⚠️  Warning: Failed to parse evaluation, assuming not satisfied")
        state['vector_satisfied'] = False
        last_attempt['satisfied'] = False

    return state

def tavily_search_node(state: AgentState) -> AgentState:
    """
    Node 5: Provede web search pomocí Tavily
    """
    state['tavily_iteration'] = state.get('tavily_iteration', 0) + 1

    print("\n" + "="*70)
    print(f"🌐 STEP 5: Tavily Web Search (Attempt {state['tavily_iteration']}/{MAX_TAVILY_ITERATIONS})")
    print("="*70)

    query = state.get('clarified_question', state['original_question'])

    print(f"\n📝 Query: {query}")
    print(f"🔍 Searching web...")

    try:
        # Initialize Tavily search
        tavily = TavilySearchResults(max_results=3)
        results = tavily.invoke(query)

        # Format results
        web_context = "\n\n".join([
            f"Source {i+1}:\n{result.get('content', '')}"
            for i, result in enumerate(results)
        ])

        state['web_context'] = web_context

        print(f"\n📊 Results:")
        print(f"  Found: {len(results)} web sources")

        # Store attempt
        attempt = {
            'iteration': state['tavily_iteration'],
            'query': query,
            'results': results,
            'satisfied': False  # Will be updated in evaluation
        }

        if 'tavily_search_attempts' not in state:
            state['tavily_search_attempts'] = []
        state['tavily_search_attempts'].append(attempt)

    except Exception as e:
        print(f"❌ Tavily search failed: {e}")
        state['web_context'] = ""
        state['tavily_satisfied'] = False

    return state

def evaluate_tavily_node(state: AgentState) -> AgentState:
    """
    Node 6: LLM posoudí, jestli jsou výsledky z Tavily dostačující
    """
    print("\n" + "="*70)
    print("⚖️  STEP 6: Evaluating Tavily search results...")
    print("="*70)

    last_attempt = state['tavily_search_attempts'][-1]

    if not state.get('web_context'):
        print("\n❌ No web context available")
        state['tavily_satisfied'] = False
        last_attempt['satisfied'] = False
        return state

    messages = [
        SystemMessage(content="""Jsi evaluátor kvality webových výsledků.
Posouď, jestli poskytnuté webové zdroje obsahují dostatečné informace pro odpověď na dotaz.

Odpověz POUZE ve formátu JSON:
{
  "satisfied": true/false,
  "reason": "krátké zdůvodnění"
}"""),
        HumanMessage(content=f"""Dotaz uživatele: {state.get('clarified_question', state['original_question'])}

Webové zdroje:
{state['web_context'][:1000]}...

Obsahují tyto zdroje dostatečné informace?""")
    ]

    response = llm.invoke(messages)

    try:
        result = json.loads(response.content)
        satisfied = result.get('satisfied', False)
        reason = result.get('reason', '')

        state['tavily_satisfied'] = satisfied
        last_attempt['satisfied'] = satisfied

        print(f"\n📊 Evaluation:")
        print(f"  Satisfied: {satisfied}")
        print(f"  Reason: {reason}")

    except json.JSONDecodeError:
        print("⚠️  Warning: Failed to parse evaluation, assuming not satisfied")
        state['tavily_satisfied'] = False
        last_attempt['satisfied'] = False

    return state

def generate_final_answer_node(state: AgentState) -> AgentState:
    """
    Node 7: Vygeneruje finální odpověď pro uživatele
    """
    print("\n" + "="*70)
    print("✨ STEP 7: Generating final answer...")
    print("="*70)

    # Prepare context from vector search
    vector_context = ""
    if state.get('best_docs'):
        vector_context = "\n\n".join([
            f"{doc.get('name', 'Unknown')}:\n{doc.get('text', '')}"
            for doc, score in state['best_docs'][:3]
        ])

    # Prepare full context
    full_context = f"""DATABÁZE ESENCIÁLNÍCH OLEJŮ:
{vector_context}

WEBOVÉ ZDROJE:
{state.get('web_context', 'Žádné webové zdroje')}"""

    messages = [
        SystemMessage(content="""Jsi zkušený aromaterapeut a expert na přírodní medicínu.

DŮLEŽITÉ POKYNY:
1. Odpovídej POUZE na základě poskytnutého kontextu
2. Odpovídaj STEJNÝM JAZYKEM jako otázka uživatele (čeština/slovenština)
3. Používej přirozený, vstřícný tón - jako expert který radí
4. Doporuč KOMBINACI esenciálních olejů A bylinných přípravků (pokud jsou v kontextu)
5. Pro každé doporučení uveď:
   - Konkrétní názvy (esenciální oleje + bylinky)
   - Jak je používat (inhalace, masáž, difuzér, čaj, tinktura)
   - Případná upozornění
6. Nepiš to jako seznam z databáze, ale jako radu od zkušeného terapeuta"""),
        HumanMessage(content=f"""Klient se ptá: {state.get('clarified_question', state['original_question'])}

Kontext:
{full_context}

Poskytni kompletní, přátelskou a odbornou odpověď:""")
    ]

    response = llm.invoke(messages)
    state['final_answer'] = response.content
    state['status'] = 'success'

    print("\n✅ Final answer generated")

    return state

def generate_apology_node(state: AgentState) -> AgentState:
    """
    Node 8: Omluva uživateli + odeslání emailu + logging
    """
    print("\n" + "="*70)
    print("😔 STEP 8: Generating apology and sending notification...")
    print("="*70)

    # Generate apology message
    apology = f"""Omlouváme se, ale nepodařilo se nám najít uspokojivou odpověď na váš dotaz.

Váš dotaz: {state['original_question']}

Problém: {state.get('problem', 'N/A')}
Příčina: {state.get('cause', 'N/A')}
Symptomy: {state.get('symptoms', 'N/A')}

Vyzkoušeli jsme:
• {len(state.get('vector_search_attempts', []))}x hledání v naší databázi esenciálních olejů
• {len(state.get('tavily_search_attempts', []))}x webové vyhledávání

Bohužel jsme nenašli dostatečně relevantní informace pro kvalitní odpověď.

Prosím zanechte nám zde svůj kontakt na email či mobil, ozveme se Vám zpět. 

Váš tým Fleurdin AI"""

    state['final_answer'] = apology
    state['status'] = 'failed'

    print("\n📧 Sending email notification...")
    email_sent = send_email_notification(state)

    if email_sent:
        print("✅ Email sent successfully")
    else:
        print("❌ Failed to send email")

    return state

# ============================================
# CONDITIONAL EDGES
# ============================================

def route_after_clarification(state: AgentState) -> str:
    """Rozhodne, jestli potřebujeme clarification"""
    needs_clarification = not (
        state.get('has_problem', False) and
        state.get('has_cause', False) and
        state.get('has_symptoms', False)
    )

    if needs_clarification:
        print("\n➡️  Route: Need clarification")
        return "clarify"
    else:
        print("\n➡️  Route: Skip clarification, go to vector search")
        # Set clarified_question same as original
        state['clarified_question'] = state['original_question']
        return "vector_search"

def route_after_vector_evaluation(state: AgentState) -> str:
    """Rozhodne co dělat po Vector Search evaluation"""

    if state.get('vector_satisfied', False):
        print("\n➡️  Route: Vector search satisfied → Final answer")
        return "generate_final_answer"

    if state.get('vector_iteration', 0) < MAX_VECTOR_ITERATIONS:
        print(f"\n➡️  Route: Try vector search again ({state['vector_iteration']}/{MAX_VECTOR_ITERATIONS})")
        return "ask_refinement"

    print("\n➡️  Route: Max vector iterations reached → Try Tavily")
    return "tavily_search"

def route_after_tavily_evaluation(state: AgentState) -> str:
    """Rozhodne co dělat po Tavily Search evaluation"""

    if state.get('tavily_satisfied', False):
        print("\n➡️  Route: Tavily search satisfied → Final answer")
        return "generate_final_answer"

    if state.get('tavily_iteration', 0) < MAX_TAVILY_ITERATIONS:
        print(f"\n➡️  Route: Try Tavily again ({state['tavily_iteration']}/{MAX_TAVILY_ITERATIONS})")
        return "tavily_search"

    print("\n➡️  Route: Max Tavily iterations reached → Apology + Email")
    return "generate_apology"

def ask_user_refinement_node(state: AgentState) -> AgentState:
    """Zeptá se uživatele na upřesnění dotazu pro další VectorSearch"""
    print("\n" + "="*70)
    print("💬 Asking user for query refinement...")
    print("="*70)

    print("\n❓ Nenašli jsme dostatečně relevantní výsledky.")
    print("   Můžete prosím upřesnit váš dotaz nebo přidat více informací?")

    refinement = input("\n   Upřesnění: ").strip()

    # Update clarified question
    if refinement:
        state['clarified_question'] = f"{state['clarified_question']}. {refinement}"

    print(f"\n✅ Updated query: {state['clarified_question']}")

    return state

# ============================================
# BUILD LANGGRAPH WORKFLOW
# ============================================

def build_graph():
    """Sestaví LangGraph workflow"""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("check_clarification", check_clarification_node)
    workflow.add_node("clarify_question", clarify_question_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("evaluate_vector", evaluate_vector_node)
    workflow.add_node("ask_refinement", ask_user_refinement_node)
    workflow.add_node("tavily_search", tavily_search_node)
    workflow.add_node("evaluate_tavily", evaluate_tavily_node)
    workflow.add_node("generate_final_answer", generate_final_answer_node)
    workflow.add_node("generate_apology", generate_apology_node)

    # Add edges
    workflow.add_edge(START, "check_clarification")

    # Conditional: clarification needed?
    workflow.add_conditional_edges(
        "check_clarification",
        route_after_clarification,
        {
            "clarify": "clarify_question",
            "vector_search": "vector_search"
        }
    )

    workflow.add_edge("clarify_question", "vector_search")
    workflow.add_edge("vector_search", "evaluate_vector")

    # Conditional: after vector evaluation
    workflow.add_conditional_edges(
        "evaluate_vector",
        route_after_vector_evaluation,
        {
            "generate_final_answer": "generate_final_answer",
            "ask_refinement": "ask_refinement",
            "tavily_search": "tavily_search"
        }
    )

    workflow.add_edge("ask_refinement", "vector_search")
    workflow.add_edge("tavily_search", "evaluate_tavily")

    # Conditional: after tavily evaluation
    workflow.add_conditional_edges(
        "evaluate_tavily",
        route_after_tavily_evaluation,
        {
            "generate_final_answer": "generate_final_answer",
            "tavily_search": "tavily_search",
            "generate_apology": "generate_apology"
        }
    )

    workflow.add_edge("generate_final_answer", END)
    workflow.add_edge("generate_apology", END)

    return workflow.compile()

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Hlavní funkce"""
    print("\n" + "="*70)
    print("🌿 FLEURDIN AI - Iterativní RAG Agent")
    print("="*70)

    # Validate configuration
    if not OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    if not TAVILY_API_KEY:
        print("⚠️  WARNING: TAVILY_API_KEY not found in .env")
        print("   Tavily search will not work")

    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        print("⚠️  WARNING: Gmail credentials not found in .env")
        print("   Email notifications will not work")

    # Build graph
    print("\n📊 Building workflow graph...")
    graph = build_graph()
    print("✅ Graph built successfully")

    # Get user question
    print("\n" + "-"*70)
    question = input("\n💬 Zadejte váš dotaz (nebo 'exit' pro ukončení): ").strip()

    if question.lower() in ['exit', 'quit', 'konec']:
        print("\n👋 Nashledanou!")
        return

    # Initialize state
    initial_state = {
        'original_question': question,
        'vector_search_attempts': [],
        'tavily_search_attempts': [],
        'vector_iteration': 0,
        'tavily_iteration': 0,
        'conversation_log': []
    }

    # Run workflow
    print("\n🚀 Starting agent workflow...\n")

    try:
        final_state = graph.invoke(initial_state)

        # Display final answer
        print("\n" + "="*70)
        print("📝 FINAL ANSWER:")
        print("="*70)
        print(f"\n{final_state['final_answer']}\n")

        # Save conversation log
        print("\n" + "-"*70)
        print("💾 Saving conversation log...")
        log_file = save_conversation_log(final_state)

        if log_file:
            print(f"✅ Log saved to: {log_file}")

        print("\n" + "="*70)
        print(f"✅ Session completed with status: {final_state['status']}")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Session interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
