import asyncio
import os
from dotenv import load_dotenv
import pyaudio

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

from openai import AsyncOpenAI
import pyttsx3

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
client = AsyncOpenAI()
engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()

# Initialize memory
conversation_history = [
    {"role": "system", "content": "You are a helpful voice assistant. Keep your answers conversational and concise."}
]
async def askLLM(prompt):
    global conversation_history
    
    # Add user message to memory
    conversation_history.append({"role": "user", "content": prompt})
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history,
        temperature=0.7,
    )
    
    reply = response.choices[0].message.content
    
    # Add assistant message to memory
    conversation_history.append({"role": "assistant", "content": reply})
    
    return reply


async def main():
    # Initialize Deepgram client
    dg_client = DeepgramClient(DEEPGRAM_API_KEY)
    
    dg_connection = dg_client.listen.asyncwebsocket.v("1")

    async def on_message(self, result, **kwargs):
        try:
            sentence = result.channel.alternatives[0].transcript
            if sentence:
                print(f"🗣️ You said: {sentence}")
                response = await askLLM(sentence)
                print(f"🤖 Assistant: {response}")
                speak(response)
        except Exception as e:
            # Deepgram occasionally sends other event types (like Metadata) that don't have transcripts
            pass
    
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

    options = LiveOptions(
        model="nova-2",
        language="en-US",
        smart_format=True,
        encoding="linear16",
        channels=1,
        sample_rate=44100,
    )

    await dg_connection.start(options)

    p = pyaudio.PyAudio()
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        input=True,
        frames_per_buffer=1024,
    )

    print("🎤 Listening... Press Ctrl+C to stop")

    try:
        while True:
            # Use to_thread to prevent stream.read from blocking the vital asyncio event loop
            data = await asyncio.to_thread(stream.read, 1024, exception_on_overflow=False)
            await dg_connection.send(data)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n🛑 Stopping...")
    
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        try:
            await dg_connection.finish()
        except Exception:
            pass
        print("✅ Done")

if __name__ == "__main__":
    asyncio.run(main())