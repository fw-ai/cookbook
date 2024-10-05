// File: app/page.tsx

"use client";

import { useChat } from 'ai/react';

interface VideoObj {
  href: string;
  alt: string;
  src: string;
}

interface FlightObj {
  airline_logo: string;
  price: string | number;
  departure_airport_name: string;
  departure_airport_time: string;
  arrival_airport_name: string;
  arrival_airport_time: string;
}

export default function () {
  const { messages, handleSubmit, input, handleInputChange } = useChat({
    api: "http://localhost:8000/chat",
  });
  return (
    <div className="flex flex-col bg-white items-center w-screen min-h-screen text-black">
      <form
        onSubmit={handleSubmit}
        className="mt-12 flex w-full max-w-[300px] flex-col"
      >
        <input
          id="input"
          name="prompt"
          value={input}
          autoComplete="off"
          onChange={handleInputChange}
          placeholder="Flights from San Francisco to Amsterdam on 2024-12-06"
          className="mt-3 min-w-[500px] rounded border px-2 py-1 outline-none focus:border-black text-black"
        />
        <button
          type="submit"
          className="mt-3 max-w-max rounded border px-3 py-1 outline-none text-black hover:bg-black hover:text-white"
        >
          Ask &rarr;
        </button>
        {messages.map((message, i) =>
          message.role === "assistant" ? (
            <div key={`response_${i}`} className="mt-3 pt-3 flex flex-col">
              {JSON.parse(message.content)["videos"] ? (
                <>
                  {JSON.parse(message.content).videos.map(
                    (videoObj: VideoObj) => (
                      <div
                        key={videoObj.href + i}
                        className="mt-3 flex flex-col"
                      >
                        <img
                          loading="lazy"
                          alt={videoObj.alt}
                          src={videoObj.src}
                        />
                        <a target="_blank" href={videoObj.href}>
                          Watch &rarr;
                        </a>
                      </div>
                    )
                  )}
                </>
              ) : (
                JSON.parse(message.content).flights.map(
                  (flight: FlightObj, _: number) => (
                    <div
                      key={`flight_${_}_${i}`}
                      className="mt-3 flex flex-col"
                    >
                      <div className="flex flex-row items-center gap-x-3">
                        <img
                          loading="lazy"
                          className="size-10"
                          src={flight.airline_logo}
                        />
                        <span>USD {flight.price}</span>
                      </div>
                      <div className="flex flex-row items-center gap-x-3">
                        <span>{flight.departure_airport_name}</span>
                        <span>{flight.departure_airport_time}</span>
                      </div>
                      <div className="flex flex-row items-center gap-x-3">
                        <span>{flight.arrival_airport_name}</span>
                        <span>{flight.arrival_airport_time}</span>
                      </div>
                    </div>
                  )
                )
              )}
            </div>
          ) : (
            <div
              className="mt-3 border-t text-black pt-3"
              key={message.content + i}
            >
              {message.content}
            </div>
          )
        )}
      </form>
    </div>
  );
}
