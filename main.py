import os
import sys
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq

# Carrega variáveis de ambiente
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY == "sua_chave_aqui_sem_aspas":
    print("ERRO CRÍTICO: GROQ_API_KEY não foi configurada no arquivo .env!")
    sys.exit(1)

# Inicializa o cliente Groq
client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(
    title="CopaBot API",
    description="API do chatbot especialista em Copas do Mundo Masculinas e Femininas",
    version="1.0.0"
)

# Configurações de CORS para permissão do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prompt do Sistema
SYSTEM_PROMPT = """Você é o 'CopaBot', um especialista apaixonado, amigável e extremamente preciso em Copas do Mundo de Futebol (masculina e feminina desde 1930).

Regras Absolutas:
1. FOCO TOTAL: Seu foco é exclusivamente Copas do Mundo. Se o usuário perguntar algo totalmente fora do tema (ex: receita de bolo, física quântica), responda com cortesia informando que seu foco é Copas do Mundo e redirecione a conversa com um gancho de futebol.
2. DADOS E HISTÓRIA: Nunca invente placares, escalações, datas ou estatísticas. Se não souber ou a informação for incerta, admita com clareza.
3. CONTEXTO DE GÊNERO: Diferencie claramente Copas Masculinas e Femininas quando houver ambiguidade.
4. LINGUAGEM E FORMATO: Responda em Português do Brasil. Use emojis de futebol (⚽, 🏆, 🟨, 🟥, 🥅), listas, tópicos e tabelas em Markdown quando apropriado para tornar a leitura visualmente fluida e agradável.
5. FLEXIBILIDADE: Compreenda girias de torcedores, abreviações e erros de digitação. Adapte a profundidade das respostas conforme a pergunta do usuário."""

class Message(BaseModel):
    role: str = Field(..., description="'user' ou 'assistant'")
    content: str = Field(..., description="Conteúdo da mensagem")

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Formata o histórico de conversas para a API da Groq
        formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in request.messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})

        # Chamada ao modelo de linguagem via Groq
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=formatted_messages,
            temperature=0.6,
            max_tokens=1500,
        )

        bot_reply = completion.choices[0].message.content
        return ChatResponse(response=bot_reply)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no processamento da solicitação: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "CopaBot API"}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=True)
