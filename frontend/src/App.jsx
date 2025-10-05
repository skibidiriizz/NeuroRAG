import React, {useState, useEffect, useRef} from 'react'
import axios from 'axios'

export default function App(){
  const [token, setToken] = useState(null)
  const [message, setMessage] = useState('')
  const wsRef = useRef()

  async function handleLogin(){
    try{
      const res = await axios.post('http://localhost:8000/auth/login', {username:'admin', password:'password'})
      setToken(res.data.access_token)
    }catch(e){
      alert('login failed')
    }
  }

  useEffect(()=>{
    if(token){
      const ws = new WebSocket('ws://localhost:8000/ws')
      ws.onopen = ()=> console.log('ws open')
      ws.onmessage = (e)=> setMessage(e.data)
      wsRef.current = ws
      return ()=> ws.close()
    }
  },[token])

  const send = ()=>{
    wsRef.current?.send('hello from frontend')
  }

  return (
    <div style={{padding:20}}>
      <h1>NeuroRAG Frontend</h1>
      {!token ? (
        <div>
          <button onClick={handleLogin}>Login (demo)</button>
        </div>
      ) : (
        <div>
          <p>Token: {token?.slice(0,20)}...</p>
          <button onClick={send}>Send WS</button>
          <div>Message: {message}</div>
        </div>
      )}
    </div>
  )
}
