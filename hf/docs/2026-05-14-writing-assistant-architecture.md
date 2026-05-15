# Writing Assistant — Architecture Diagram

```
BROWSER                                FASTAPI SERVER                        MLX (Apple Silicon)
─────────────────────────────────────  ────────────────────────────────────  ───────────────────

User pastes text, picks a mode,
clicks "Run"
    │
    │  POST /generate
    │  { text: "...", mode: "summarize" }
    │ ─────────────────────────────────►
    │                                   async def generate(request):
    │                                       │
    │                                       │  creates asyncio.Queue
    │                                       │  ┌─────────┐
    │                                       └─►│  queue  │
    │                                          └────┬────┘
    │                                               │
    │                                               │  loop.run_in_executor()
    │                                               │  (hands off to thread pool)
    │                                               │ ─────────────────────────►
    │                                               │                            worker thread
    │                                               │                            calls stream_generate()
    │                                               │                            (blocking — runs on
    │                                               │                             Metal GPU kernels)
    │                                               │                                │
    │                                               │         token: "The"           │
    │                                          ┌────┴────┐  ◄─────────────────────── │
    │                                          │  queue  │                            │
    │                                          └────┬────┘       token: " key"        │
    │                                               │          ◄─────────────────────  │
    │  SSE stream (persistent HTTP connection)      │                            ... continues
    │  ◄────────────────────────────────────────    │
    │                                          StreamingResponse
    │  data: The\n\n                           reads from queue,
    │  data:  key\n\n                          formats each token
    │  data: ...\n\n                           as SSE message
    │  event: done\n\n                         sends "done" event
    │                                          when queue signals
    │                                          completion
    │
    │  EventSource receives each message
    │  appends token to output area
    │  on "done" event: re-enables button


─────────────────────────────────────────────────────────────────────────────────────────────────

SSE WIRE FORMAT (what travels over the HTTP connection):

  Each token:         data: Hello\n\n
  Completion signal:  event: done\ndata: \n\n

  The \n\n (double newline) is the message delimiter — the browser's
  EventSource parser uses it to know when one message ends and the next begins.

─────────────────────────────────────────────────────────────────────────────────────────────────

WHY THE QUEUE?

  stream_generate() is a synchronous blocking call — it cannot be awaited.
  If called directly inside an async endpoint it would freeze the event loop,
  blocking all other requests for the entire duration of generation.

  The queue bridges the two worlds:

    Thread (sync)  ──►  queue.put(token)   ──►  async endpoint  ──►  SSE response
                        (non-blocking)          await queue.get()

  The async endpoint yields control back to the event loop between each token,
  keeping the server responsive while the model runs in a background thread.
```
