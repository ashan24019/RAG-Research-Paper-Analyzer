import React, { useState } from 'react'

export default function App() {
  const [sessionId, setSessionId] = useState(null)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState(null)

  return (
    <div style={{padding:20, fontFamily:'Arial'}}>
      <h1>Research Paper RAG (Frontend)</h1>
      <p>This is a minimal React scaffold. Use the Streamlit or API backend for processing.</p>

      <section style={{marginTop:20}}>
        <h2>Upload (use API or Streamlit)</h2>
        <p>Use the backend `/upload` endpoint to create a session. Then use the session ID below for questions.</p>
      </section>

      <section style={{marginTop:20}}>
        <h2>Ask a question</h2>
        <input placeholder="session id" value={sessionId || ''} onChange={e=>setSessionId(e.target.value)} style={{width:400, padding:8}} />
        <br/>
        <textarea placeholder="Your question" value={question} onChange={e=>setQuestion(e.target.value)} style={{width:600, height:100, marginTop:8}} />
        <br/>
        <button onClick={async ()=>{
          if(!sessionId || !question) return alert('provide session id and question')
          try{
            const res = await fetch('http://127.0.0.1:8000/ask', {
              method:'POST',
              headers:{'Content-Type':'application/json'},
              body: JSON.stringify({session_id: sessionId, query: question})
            })
            const data = await res.json()
            setAnswer(data.result)
          }catch(e){
            alert('Request failed: '+e)
          }
        }} style={{marginTop:8, padding:'8px 16px'}}>
          Get Answer
        </button>
      </section>

      {answer && (
        <section style={{marginTop:20}}>
          <h2>Answer</h2>
          <div style={{whiteSpace:'pre-wrap'}}>{answer}</div>
        </section>
      )}
    </div>
  )
}
