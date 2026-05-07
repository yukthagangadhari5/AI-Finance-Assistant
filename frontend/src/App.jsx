import { useState } from "react";

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0f172a",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "40px 20px",
        fontFamily: "sans-serif",
      }}
    >
      <h1 style={{ color: "white" }}>AI Finance Assistant</h1>
      <p style={{ color: "gray" }}>
        Powered by 9,189 RBI document chunks + Gemini AI
      </p>

      <div style={{ width: "100%", maxWidth: 820, display: "flex", gap: 12 }}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          style={{
            flex: 1,
            padding: "12px 14px",
            borderRadius: 10,
            border: "1px solid #334155",
            background: "#0b1220",
            color: "white",
          }}
        />
        <button
          onClick={handleSubmit}
          style={{
            padding: "12px 16px",
            borderRadius: 10,
            border: "1px solid #334155",
            background: "#1e293b",
            color: "white",
            cursor: "pointer",
          }}
        >
          Ask
        </button>
      </div>

      {loading ? (
        <p style={{ color: "gray", marginTop: 16 }}>Loading...</p>
      ) : null}

      {error ? (
        <p style={{ color: "#fca5a5", marginTop: 16 }}>{error}</p>
      ) : null}

      {result ? (
        <div
          style={{
            width: "100%",
            maxWidth: 820,
            marginTop: 20,
            background: "#111827",
            border: "1px solid #334155",
            borderRadius: 12,
            padding: 16,
            color: "white",
          }}
        >
          <h3 style={{ marginTop: 0 }}>Answer</h3>
          <p style={{ whiteSpace: "pre-wrap" }}>{result.answer}</p>

          <p>
            <strong>Confidence:</strong> {String(result.confidence)}
          </p>

          <div>
            <strong>Sources:</strong>
            <ul>
              {(Array.isArray(result.sources) ? result.sources : []).map(
                (s, idx) => (
                  <li key={`${s}-${idx}`}>{String(s)}</li>
                )
              )}
            </ul>
          </div>

          <p style={{ marginBottom: 0 }}>
            <strong>Tokens Used:</strong> {String(result.tokens_used)}
          </p>
        </div>
      ) : null}
    </div>
  );
}
