import asyncio
import os
from dotenv import load_dotenv
import pyaudio

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

async def main():
    # Initialize Deepgram client
    dg_client = DeepgramClient(DEEPGRAM_API_KEY)
    
    # Connect to Deepgram streaming API
    dg_connection = dg_client.listen.asyncc.to_url(
        LiveOptions(
            model="nova-2",
            interim_results=True,
            smart_format=True,
        )
    )
    
    # Handle transcription events
    async def on_message(result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if sentence:
            print(f"User: {sentence}")
    
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
    
    # Start microphone stream
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024,
    )
    
    # Stream audio to Deepgram
    await dg_connection.start(dg_client.listen.asyncc.websocket_url)
    
    try:
        while True:
            data = stream.read(1024)
            await dg_connection.send(data)
    except KeyboardInterrupt:
        pass
    finally:
        await dg_connection.finish()
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(main())