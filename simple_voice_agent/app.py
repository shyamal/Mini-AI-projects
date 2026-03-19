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
    
    dg_connection = dg_client.listen.asyncwebsocket.v("1")

    async def on_message(self, result, **kwargs):
        try:
            sentence = result.channel.alternatives[0].transcript
            if sentence:
                print(f"🗣️ You said: {sentence}")
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