import { ChangeEvent, FormEvent, useMemo, useState } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";

type UploadResponse = {
  success: boolean;
  document_id: string;
  filename: string;
  total_chunks: number;
  message: string;
};

type Source = {
  text: string;
  page: number;
  chunk_id: string;
  distance: number;
};

type ChatResponse = {
  answer: string;
  sources: Source[];
  chunk_used: number;
  document_id?: string;
};

type ChatMessage = {
  role: "system" | "user" | "assistant";
  text: string;
  sources?: Source[];
};

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") return payload.detail;
    if (typeof payload?.error === "string") return payload.error;
  } catch {
    // Ignore JSON parsing errors and use fallback.
  }
  return `Request failed with status ${response.status}`;
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [activeDocumentId, setActiveDocumentId] = useState("");
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "system", text: "Upload a PDF, then ask questions about it." }
  ]);

  const canUpload = useMemo(() => !!selectedFile && !uploading, [selectedFile, uploading]);
  const canAsk = useMemo(() => question.trim().length > 0 && !asking, [question, asking]);

  function onFilePicked(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setError("");
    setStatus("");

    if (!file) {
      setSelectedFile(null);
      return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setSelectedFile(null);
      setError("Only PDF files are allowed.");
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setSelectedFile(null);
      setError("File is too large. Max size is 50MB.");
      return;
    }

    setSelectedFile(file);
    setStatus(`Selected: ${file.name}`);
  }

  async function onUploadClick() {
    if (!selectedFile) return;

    setUploading(true);
    setError("");
    setStatus("Uploading and processing PDF...");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE}/api/documents/upload`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }

      const data = (await response.json()) as UploadResponse;
      setActiveDocumentId(data.document_id);
      setStatus(`Uploaded: ${data.filename} | Chunks: ${data.total_chunks}`);
      setMessages((prev) =>
        prev.concat({
          role: "system",
          text: `Document ready. ID: ${data.document_id}`
        })
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setStatus("");
    } finally {
      setUploading(false);
    }
  }

  async function onAskSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const questionText = question.trim();
    if (!questionText || asking) return;

    setQuestion("");
    setError("");
    setMessages((prev) => prev.concat({ role: "user", text: questionText }));
    setAsking(true);

    try {
      const payload = activeDocumentId
        ? { question: questionText, document_id: activeDocumentId }
        : { question: questionText };

      const response = await fetch(`${API_BASE}/api/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }

      const data = (await response.json()) as ChatResponse;
      setMessages((prev) =>
        prev.concat({
          role: "assistant",
          text: data.answer,
          sources: data.sources
        })
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat failed");
    } finally {
      setAsking(false);
    }
  }

  return (
    <main className="page">
      <header className="hero">
        <h1>Research Paper RAG Analyzer</h1>
        <p>Upload a paper, then ask grounded questions from retrieved chunks.</p>
      </header>

      <section className="panel">
        <h2>1) Upload PDF</h2>
        <div className="row">
          <input
            type="file"
            accept=".pdf,application/pdf"
            onChange={onFilePicked}
            disabled={uploading}
          />
          <button type="button" onClick={onUploadClick} disabled={!canUpload}>
            {uploading ? "Processing..." : "Upload"}
          </button>
        </div>
        {activeDocumentId ? <p className="doc-id">Active Document ID: {activeDocumentId}</p> : null}
      </section>

      <section className="panel">
        <h2>2) Ask Questions</h2>
        <form onSubmit={onAskSubmit} className="chat-form">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Example: What is the main methodology used in this paper?"
            rows={3}
          />
          <button type="submit" disabled={!canAsk}>
            {asking ? "Thinking..." : "Ask"}
          </button>
        </form>

        <div className="messages">
          {messages.map((message, index) => (
            <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
              <h3>{message.role.toUpperCase()}</h3>
              <p>{message.text}</p>

              {message.sources && message.sources.length > 0 ? (
                <ul className="sources">
                  {message.sources.map((source) => (
                    <li key={source.chunk_id}>
                      <strong>Page {source.page}</strong>
                      <span>{source.text.slice(0, 180)}...</span>
                    </li>
                  ))}
                </ul>
              ) : null}
            </article>
          ))}
        </div>
      </section>

      {status ? <p className="status">{status}</p> : null}
      {error ? <p className="status error">{error}</p> : null}
    </main>
  );
}

export default App;
