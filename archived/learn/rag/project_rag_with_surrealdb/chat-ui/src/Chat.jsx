// File: src/Chat.jsx

import { useChat } from 'ai/react'

export default function () {
  const { messages, handleSubmit, input, handleInputChange } = useChat({
    api: 'http://localhost:8000/chat',
  })
  return (
    <form className="mt-12 flex w-full max-w-3xl px-10 flex-col" onSubmit={handleSubmit}>
      <input
        id="input"
        name="prompt"
        value={input}
        autoComplete='off'
        onChange={handleInputChange}
        placeholder="What's your next question?"
        className="mt-3 rounded border px-5 py-3 outline-none focus:border-black"
      />
      <button className="mt-3 max-w-max rounded border px-3 py-1 outline-none hover:bg-black hover:text-white" type="submit">
        Ask &rarr;
      </button>
      {messages.map((message, i) => (
        <div className="mt-3 border-t pt-3" key={i}>
          {message.content}
        </div>
      ))}
    </form>
  )
}