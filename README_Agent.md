# 🌿 Fleurdin AI - Iterativní RAG Agent

Inteligentní agent pro aromatherapii s multi-step reasoning a adaptivním vyhledáváním.

**Vytvořeno:** 2025-12-17
**Autor:** Claude Code + Tomáš
**Framework:** LangChain + LangGraph
**Model:** OpenAI GPT-4-mini

---

## 📋 Obsah

- [Přehled](#-přehled)
- [Architektura](#-architektura)
- [Workflow](#-workflow)
- [Instalace](#-instalace)
- [Konfigurace](#-konfigurace)
- [Použití](#-použití)
- [Features](#-features)
- [Struktura projektu](#-struktura-projektu)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Přehled

Tento agent implementuje pokročilý RAG (Retrieval-Augmented Generation) systém s následujícími vlastnostmi:

- **Mandatory Clarification** - Vždy zjistí problém, příčinu a symptomy od uživatele
- **Iterativní VectorSearch** - Až 3 pokusy s upřesněním dotazu
- **Fallback na Tavily** - Web search při selhání databázového vyhledávání
- **Email notifikace** - Automatické upozornění při selhání agenta
- **Conversation logging** - Kompletní záznam všech interakcí

---

## 🏗️ Architektura

### Produkční architektura (plánovaná)

```
┌─────────────────────────────────────────┐
│  User Interface                          │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  LangGraph Agent                         │
│  • Mandatory Clarification               │
│  • VectorSearch Tool                     │
│  • TavilySearch Tool                     │
│  • Email Notification                    │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Vector Database: Qdrant Cloud           │
│  • Collection: essential_oils            │
│  • Collection: herb_knowledge            │
│  • Vector dim: 384                       │
│  • Chunks: 3,505                         │
└──────────────────────────────────────────┘
```

### ⚠️ Aktuální implementace (Development)

**Pro účely vývoje a testování používáme místo Qdrant lokální JSON:**

```
┌─────────────────────────────────────────┐
│  LangGraph Agent                         │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  JSON File: chunked_data_FIXED.json      │
│  • Format: Chunks with embeddings        │
│  • Vector dim: 384                       │
│  • Model: paraphrase-multilingual-       │
│           MiniLM-L12-v2                  │
└──────────────────────────────────────────┘
```

**Poznámka:** Produkční verze použije přímé připojení k Qdrant Cloud místo JSON souboru.

---

## 🔄 Workflow

Agent pracuje v následujícím workflow:

```
START
  ↓
┌─────────────────────────────────────────────────┐
│ 1. CHECK CLARIFICATION                          │
│    • LLM analyzuje dotaz                        │
│    • Zkontroluje: problém, příčinu, symptomy    │
└────────────────┬────────────────────────────────┘
                 ↓
        ┌────────┴────────┐
        │                 │
    COMPLETE         INCOMPLETE
        │                 │
        ↓                 ↓
    SKIP         ┌──────────────────┐
                 │ 2. CLARIFICATION │
                 │    • Interaktivní│
                 │      dotazy      │
                 └────────┬─────────┘
                          │
        ┌─────────────────┘
        ↓
┌─────────────────────────────────────────────────┐
│ 3. VECTOR SEARCH LOOP (max 3x)                  │
│    • VectorSearchTool → cosine similarity       │
│    • Evaluate results (LLM)                     │
│    • Satisfied? → FINAL ANSWER                  │
│    • Not satisfied? → Ask user refinement       │
└────────────────┬────────────────────────────────┘
                 ↓
         ┌───────┴────────┐
         │                │
     SATISFIED      NOT SATISFIED
         │          (after 3x)
         │                │
         ↓                ↓
  FINAL ANSWER   ┌──────────────────┐
                 │ 4. TAVILY SEARCH │
                 │    (max 3x)      │
                 └────────┬─────────┘
                          │
                  ┌───────┴────────┐
                  │                │
              SATISFIED      NOT SATISFIED
                  │          (after 3x)
                  ↓                │
          FINAL ANSWER             ↓
                          ┌──────────────────┐
                          │ 5. APOLOGY       │
                          │    • Email       │
                          │    • Log         │
                          └──────────────────┘
```

### Detailní kroky

**Krok 1: Mandatory Clarification**
- Agent zkontroluje, jestli dotaz obsahuje:
  - ✅ Konkrétní problém (např. "bolí mě hlava")
  - ✅ Možná příčina (např. "kvůli stresu")
  - ✅ Další symptomy (např. "a mám nevolnost")
- Pokud něco chybí → interaktivně se ptá uživatele

**Krok 2-4: Vector Search Loop**
- Max 3 pokusy
- Po každém pokusu LLM vyhodnotí relevanci
- Pokud není spokojený → zeptá se uživatele na upřesnění
- Relevance threshold: 0.6 (cosine similarity)

**Krok 5-6: Tavily Search Fallback**
- Aktivuje se, pokud VectorSearch selhal
- Max 3 pokusy web search
- LLM vyhodnotí kvalitu webových zdrojů

**Krok 7: Final Answer**
- LLM vygeneruje přátelskou, odbornou odpověď
- Kombinuje databázové + webové zdroje

**Krok 8: Apology + Email**
- Pokud ani Tavily nepomohl (3x neúspěch)
- Omluva uživateli
- Email notifikace na zadaný Gmail
- Kompletní conversation log do TXT

---

## 🚀 Instalace

### 1. Systémové požadavky

- Python 3.10+
- pip nebo uv

### 2. Nainstalovat dependencies

```bash
# Pomocí pip
pip install -r requirements.txt

# Nebo pomocí uv (doporučeno)
uv pip install -r requirements.txt
```

### 3. Stáhnout embedding model

Při prvním spuštění se automaticky stáhne model:
- `paraphrase-multilingual-MiniLM-L12-v2`
- Velikost: ~470 MB
- HuggingFace cache: `~/.cache/huggingface`

---

## ⚙️ Konfigurace

### 1. Vytvoř `.env` soubor

```bash
cp .env.example .env
```

### 2. Doplň API keys a credentials

Otevři `.env` a doplň:

```bash
# OpenAI API
OPENAI_API_KEY=sk-proj-...

# Tavily API (webové vyhledávání)
TAVILY_API_KEY=tvly-...

# Gmail SMTP (pro email notifikace)
GMAIL_USER=tvuj-email@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
RECIPIENT_EMAIL=kam-poslat-notifikace@gmail.com
```

### 3. Získání API keys

#### OpenAI API Key
1. Jdi na: https://platform.openai.com/api-keys
2. Vytvoř nový API key
3. Zkopíruj do `.env`

#### Tavily API Key
1. Registruj se na: https://tavily.com/
2. FREE tier: 1,000 searches/měsíc zdarma
3. Zkopíruj API key do `.env`

#### Gmail App Password
1. Jdi do Google Account → Security → 2-Step Verification
2. Scroll dolů na "App passwords"
3. Vytvoř nový app password pro "Mail"
4. Zkopíruj 16-místný kód (bez mezer) do `.env`

**Poznámka:** Musíš mít zapnuté 2FA (two-factor authentication) na Google účtu.

---

## 💻 Použití

### Základní spuštění

```bash
python Agent_iterative_Fleurdin.py
```

### Příklad session

```
🌿 FLEURDIN AI - Iterativní RAG Agent
======================================================================

📊 Building workflow graph...
✅ Graph built successfully

----------------------------------------------------------------------

💬 Zadejte váš dotaz (nebo 'exit' pro ukončení): Bolí mě hlava

🚀 Starting agent workflow...

======================================================================
🔍 STEP 1: Checking clarification needs...
======================================================================

📊 Analysis:
  Has problem: True
  Has cause: False
  Has symptoms: False

======================================================================
💬 STEP 2: Asking for clarification...
======================================================================

❓ Rozumím vašemu dotazu, ale potřebuji více informací:
   • Víte, co může být příčinou? (např. stres, únava, nemoc): stres
   • Máte i jiné symptomy? (pokud ne, napište 'ne'): únava

✅ Clarified question: Bolí mě hlava. Příčina: stres. Další symptomy: únava

======================================================================
🔎 STEP 3: Vector Search (Attempt 1/3)
======================================================================

📝 Query: Bolí mě hlava. Příčina: stres. Další symptomy: únava
🔍 Searching in 3505 chunks...

📊 Results:
  Found: 5 documents
  Best score: 0.782
  Relevance threshold: 0.6

🏆 Top results:
  1. Levandule (score: 0.782)
  2. Máta peprná (score: 0.741)
  3. Heřmánek (score: 0.698)

======================================================================
⚖️  STEP 4: Evaluating vector search results...
======================================================================

📊 Evaluation:
  Satisfied: True
  Reason: Dokumenty obsahují relevantní informace o olejích pro bolest hlavy

======================================================================
✨ STEP 7: Generating final answer...
======================================================================

✅ Final answer generated

======================================================================
📝 FINAL ANSWER:
======================================================================

Pro vaši bolest hlavy způsobenou stresem a únavou doporučuji následující:

🌿 ESENCIÁLNÍ OLEJE:

1. **Levandule** (Lavandula angustifolia)
   - Uklidňuje nervový systém a pomáhá při stresové bolesti hlavy
   - Použití: 2-3 kapky na spánky, nebo inhalace z kapesníku
   - Můžete také použít v difuzéru (5-8 kapek)

2. **Máta peprná** (Mentha piperita)
   - Osvěžuje a uvolňuje napětí v hlavě
   - Použití: 1 kapku s nosným olejem na čelo a spánky
   - Pozor: Nepoužívat u dětí pod 6 let

...

💾 Saving conversation log...
✅ Log saved to: conversation_log_2025-12-17_18-45-30.txt

======================================================================
✅ Session completed with status: success
======================================================================
```

---

## ✨ Features

### 1. Mandatory Clarification
- Agent **vždy** zjistí kompletní kontext
- Ptá se pouze pokud informace chybí
- LLM-driven detekce missing info

### 2. Iterativní VectorSearch
- Max 3 pokusy s upřesněním
- Cosine similarity search (threshold 0.6)
- Top 5 nejrelevantnějších chunků

### 3. Smart Evaluation
- LLM posuzuje kvalitu výsledků
- Rozhoduje o spokojenosti agenta
- Adaptivní strategie vyhledávání

### 4. Fallback na Web Search
- Tavily API pro aktuální informace
- Aktivuje se pouze při selhání VectorSearch
- Max 3 pokusy

### 5. Email Notifications
- Gmail SMTP
- Automatické při selhání (3x Tavily failed)
- Obsahuje kompletní shrnutí session

### 6. Conversation Logging
- TXT formát
- Timestamp každé session
- Všechny pokusy + výsledky
- Stored: `conversation_log_YYYY-MM-DD_HH-MM-SS.txt`

### 7. Professional Output
- Přátelský, odborný tón
- Kombinace databázových + webových zdrojů
- Konkrétní doporučení s použitím

---

## 📁 Struktura projektu

```
4-RAG_Pipeline/
│
├── Agent_iterative_Fleurdin.py    # ⭐ Hlavní script
├── README_Agent.md                # 📚 Tato dokumentace
├── requirements.txt               # 📦 Python dependencies
├── .env.example                   # 🔑 Template pro config
├── .env                           # 🔐 Tvoje API keys (gitignore)
│
├── chunked_data_FIXED.json        # 💾 Vector databáze (40 MB)
│                                  # 3,505 chunků s embeddings
│
└── conversation_log_*.txt         # 📝 Conversation logy
```

---

## 🔧 Troubleshooting

### ❌ "OPENAI_API_KEY not found"

**Problém:** `.env` soubor není vytvořen nebo neobsahuje API key

**Řešení:**
```bash
# 1. Zkopíruj template
cp .env.example .env

# 2. Edituj .env a doplň API key
nano .env  # nebo vim, code, atd.

# 3. Restartuj script
python Agent_iterative_Fleurdin.py
```

---

### ❌ "File not found: chunked_data_FIXED.json"

**Problém:** JSON soubor není na správné cestě

**Řešení:**
```python
# V Agent_iterative_Fleurdin.py uprav cestu:
DATA_PATH = "/tvoje/cesta/k/chunked_data_FIXED.json"
```

---

### ❌ "Failed to send email"

**Problém:** Špatné Gmail credentials nebo není zapnuté 2FA

**Řešení:**
1. Zkontroluj Gmail 2FA: https://myaccount.google.com/security
2. Vytvoř nový App Password
3. Zkopíruj do `.env` (bez mezer)
4. Ověř `GMAIL_USER` je správný email

---

### ❌ "Tavily search failed"

**Problém:** Neplatný nebo chybějící Tavily API key

**Řešení:**
1. Registruj se: https://tavily.com/
2. Zkopíruj API key
3. Doplň do `.env`: `TAVILY_API_KEY=tvly-...`

**Poznámka:** Agent funguje i bez Tavily, ale při selhání VectorSearch nebude mít fallback.

---

### ⚠️ Embedding model download je pomalý

**Problém:** První spuštění stahuje 470 MB model

**Řešení:**
- Je to normální, stane se jen jednou
- Model se cachuje do `~/.cache/huggingface`
- Počkej 2-5 minut (závisí na rychlosti připojení)

---

### ❌ "ImportError: No module named X"

**Problém:** Chybějící dependencies

**Řešení:**
```bash
# Reinstaluj všechny dependencies
pip install -r requirements.txt

# Nebo s uv
uv pip install -r requirements.txt
```

---

## 🔜 Roadmap

### Aktuální verze (v1.0)
- ✅ Mandatory Clarification
- ✅ Iterativní VectorSearch (JSON file)
- ✅ Tavily fallback
- ✅ Email notifications
- ✅ TXT logging

### Plánované features (v2.0)
- ⏳ Migrace na Qdrant Cloud
- ⏳ Multi-collection search
- ⏳ Tier filtering (free/premium)
- ⏳ Web UI (Streamlit/Gradio)
- ⏳ Conversation history persistence
- ⏳ Multi-language support
- ⏳ Advanced analytics dashboard

---

## 📞 Support

**Projekt:** Fleurdin AI
**Web:** www.fleurdin.cz
**Email:** info@fleurdin.cz

---

## 📄 License

Proprietary - Fleurdin AI © 2025

---

**Vytvořeno s ❤️ pomocí Claude Code**
