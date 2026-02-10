# === IMPORTS: Standard library modules ===
import datetime  # For timestamp formatting in logs
import builtins  # Built-in Python functions (not actively used)
import asyncio  # Asynchronous programming for WebSocket handling
import json  # JSON serialization for log messages
import os  # Environment variable access
import threading  # Threading for model generation (runs alongside async WebSocket)
import traceback  # Error logging with full stack traces
from contextlib import (
    asynccontextmanager,
)  # FastAPI lifespan context manager (replaces deprecated @app.on_event)
from pathlib import Path  # Path manipulation for file operations
from queue import Empty, Queue  # Thread-safe queue for log buffering
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, cast  # Type hints

# === IMPORTS: Third-party libraries ===
import numpy as np  # Audio data processing (numpy arrays)
import torch  # PyTorch for model inference
from fastapi import FastAPI, WebSocket  # FastAPI framework and WebSocket support
from fastapi.responses import FileResponse  # For serving index.html
from fastapi.staticfiles import (
    StaticFiles,
)  # Static file serving (not used in this file)
from starlette.websockets import (
    WebSocketDisconnect,
    WebSocketState,
)  # WebSocket event handling

# === IMPORTS: Local project modules ===
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,  # Main TTS model class
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,  # Text/audio processor and tokenizer
)
from vibevoice.modular.streamer import AudioStreamer  # Custom audio streaming utility

import copy  # Deep copy for voice preset data

# === CONSTANTS ===
BASE = Path(__file__).parent  # Directory containing this file (demo/web/)
SAMPLE_RATE = 24_000  # Audio sample rate in Hz (24kHz)


def get_timestamp():
    """Generate timestamp string in format: YYYY-MM-DD HH:MM:SS.mmm with timezone conversion."""
    # Get current UTC time
    timestamp = (
        datetime.datetime.utcnow()
        .replace(tzinfo=datetime.timezone.utc)  # Add UTC timezone info
        .astimezone(
            datetime.timezone(datetime.timedelta(hours=8))
        )  # Convert to +8 timezone (Singapore/HK)
        .strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # Format and truncate microseconds to milliseconds
    )
    return timestamp


class StreamingTTSService:
    """
    Core TTS service class that manages model loading, voice presets, and audio generation.

    Architecture:
    - Loads VibeVoice model from HuggingFace
    - Manages voice presets (pre-computed speaker embeddings)
    - Handles streaming audio generation via generator functions
    - Converts audio chunks to PCM16 format for WebSocket transmission
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        inference_steps: int = 5,
    ) -> None:
        """
        Initialize the TTS service with model path and device configuration.

        Args:
            model_path: HuggingFace model ID or local path (e.g., "microsoft/VibeVoice-Realtime-0.5B")
            device: Inference device ("cpu", "cuda", "mps", or "mpx" for Apple silicon)
            inference_steps: Number of denoising steps (5-20, higher = better quality but slower)
        """
        # Keep model_path as string for HuggingFace repo IDs (Path() converts / to \ on Windows)
        self.model_path = model_path
        self.inference_steps = inference_steps  # Default denoising steps
        self.sample_rate = SAMPLE_RATE  # 24kHz audio output

        # Model components (initialized in load())
        self.processor: Optional[VibeVoiceStreamingProcessor] = (
            None  # Text/audio processor
        )
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = (
            None  # Main model
        )

        # Voice preset management
        self.voice_presets: Dict[str, Path] = {}  # Maps voice name → .pt file path
        self.default_voice_key: Optional[str] = None  # Currently selected voice
        self._voice_cache: Dict[
            str, Tuple[object, Path, str]
        ] = {}  # Cached voice embeddings

        # Device handling with fallback logic
        if device == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            device = "mps"  # 'mpx' is an alias for MPS (Apple silicon)
        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            device = "cpu"  # Fallback if MPS not supported
        self.device = device  # Final device choice
        self._torch_device = torch.device(device)  # PyTorch device object

    def load(self) -> None:
        """
        Load the TTS model and voice presets from disk.

        This method performs:
        1. Load the processor (tokenizer + feature extractor)
        2. Load the model with appropriate dtype/attention based on device
        3. Configure noise scheduler with SDE-DPM solver algorithm
        4. Set inference steps for denoising
        5. Load voice presets and set default voice

        Device-specific optimizations:
        - MPS (Apple silicon): float32, SDPA attention (no bfloat16 support)
        - CUDA: bfloat16, FlashAttention-2 (fastest), falls back to SDPA if unavailable
        - CPU: float32, SDPA attention
        """
        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        # Select data type and attention implementation based on device capabilities
        if self.device == "mps":
            load_dtype = torch.float32  # MPS doesn't support bfloat16
            device_map = None  # No device mapping needed for single device
            attn_impl_primary = "sdpa"  # MPS only supports SDPA attention
        elif self.device == "cuda":
            load_dtype = torch.bfloat16  # CUDA supports bfloat16 for faster inference
            device_map = "cuda"
            attn_impl_primary = "flash_attention_2"  # FlashAttention is fastest on CUDA
        else:
            load_dtype = torch.float32
            device_map = "cpu"
            attn_impl_primary = "sdpa"
        print(
            f"Using device: {device_map}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}"
        )

        # Load model with primary attention implementation
        try:
            self.model = (
                VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=device_map,
                    attn_implementation=attn_impl_primary,
                )
            )

            if self.device == "mps":
                self.model.to("mps")  # Move model to MPS device
        except Exception as e:
            # Fallback: try SDPA if FlashAttention fails
            if attn_impl_primary == "flash_attention_2":
                print(
                    "Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality."
                )

                self.model = (
                    VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=load_dtype,
                        device_map=self.device,
                        attn_implementation="sdpa",
                    )
                )
                print("Load model with SDPA successfully ")
            else:
                raise e

        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()

        # Configure noise scheduler with SDE-DPM algorithm for improved audio quality
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",  # Stochastic differential equation solver
            beta_schedule="squaredcos_cap_v2",  # Cosine beta schedule with clamping
        )
        # Set number of denoising steps (fewer = faster, more = better quality)
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        # Load voice presets from disk
        self.voice_presets = self._load_voice_presets()
        # Get default voice from environment variable or use fallback logic
        preset_name = os.environ.get("VOICE_PRESET")
        self.default_voice_key = self._determine_voice_key(preset_name)
        self._ensure_voice_cached(self.default_voice_key)  # Pre-cache the default voice

    def _load_voice_presets(self) -> Dict[str, Path]:
        """
        Load all voice preset files from the voices directory.

        Voice presets are pre-computed speaker embeddings stored as .pt files.
        Each preset contains cached prompt tokens and audio features for fast inference.

        Returns:
            Dict mapping voice name (file stem) to file path

        Raises:
            RuntimeError: If voices directory doesn't exist or no .pt files found
        """
        voices_dir = BASE.parent / "voices" / "streaming_model"
        if not voices_dir.exists():
            raise RuntimeError(f"Voices directory not found: {voices_dir}")

        presets: Dict[str, Path] = {}
        # Recursively find all .pt files in the voices directory
        for pt_path in voices_dir.rglob("*.pt"):
            presets[pt_path.stem] = pt_path  # Use filename without extension as key

        if not presets:
            raise RuntimeError(f"No voice preset (.pt) files found in {voices_dir}")

        print(f"[startup] Found {len(presets)} voice presets")
        return dict(sorted(presets.items()))  # Return sorted by voice name

    def _determine_voice_key(self, name: Optional[str]) -> str:
        """
        Determine which voice preset to use based on user request and availability.

        Priority order:
        1. Requested voice (if it exists in presets)
        2. Environment variable VOICE_PRESET (if set and valid)
        3. Default voice "en-Carter_man" (if available)
        4. First available voice alphabetically (fallback)

        Args:
            name: Requested voice key from user or environment

        Returns:
            Valid voice key that exists in voice_presets
        """
        if name and name in self.voice_presets:
            return name

        default_key = "en-Carter_man"
        if default_key in self.voice_presets:
            return default_key

        first_key = next(iter(self.voice_presets))
        print(f"[startup] Using fallback voice preset: {first_key}")
        return first_key

    def _ensure_voice_cached(self, key: str) -> Tuple[object, Path, str]:
        """
        Load and cache a voice preset if not already loaded.

        Voice presets contain pre-computed embeddings that define speaker characteristics.
        These are loaded once and cached to avoid repeated disk I/O.

        Args:
            key: Voice preset identifier (filename without extension)

        Returns:
            Tuple of (cached_outputs, preset_path, unused_string)

        Raises:
            RuntimeError: If requested voice doesn't exist in presets
        """
        if key not in self.voice_presets:
            raise RuntimeError(f"Voice preset {key!r} not found")

        # Cache lookup - only load from disk once per session
        if key not in self._voice_cache:
            preset_path = self.voice_presets[key]
            print(f"[startup] Loading voice preset {key} from {preset_path}")
            print(f"[startup] Loading prefilled prompt from {preset_path}")
            # Load torch tensor with cached embeddings
            # weights_only=False allows loading arbitrary Python objects (required for this model)
            prefilled_outputs = torch.load(
                preset_path,
                map_location=self._torch_device,
                weights_only=False,
            )
            self._voice_cache[key] = prefilled_outputs

        return self._voice_cache[key]

    def _get_voice_resources(
        self, requested_key: Optional[str]
    ) -> Tuple[str, object, Path, str]:
        """
        Get voice resources for a given voice key with fallback logic.

        This is a convenience method that combines:
        1. Voice key resolution (handle None/default/fallback)
        2. Voice caching/loading

        Args:
            requested_key: User-specified voice or None

        Returns:
            Tuple of (final_key, prefilled_outputs, preset_path, unused_string)

        Note: The return type annotation says 4 elements but only returns 2.
        This is a type hint mismatch that should be fixed.
        """
        key = (
            requested_key
            if requested_key and requested_key in self.voice_presets
            else self.default_voice_key
        )
        if key is None:
            key = next(iter(self.voice_presets))
            self.default_voice_key = key

        prefilled_outputs = self._ensure_voice_cached(key)
        return key, prefilled_outputs

    def _prepare_inputs(self, text: str, prefilled_outputs: object):
        """
        Prepare model inputs from raw text and cached voice embeddings.

        This method:
        1. Tokenizes the input text
        2. Combines with cached prompt from voice preset
        3. Prepares tensors for the target device (CPU/GPU)

        Args:
            text: Input text to synthesize
            prefilled_outputs: Cached voice embeddings from _ensure_voice_cached

        Returns:
            Dict of model inputs ready for inference:
            - input_ids: Tokenized text
            - attention_mask: Mask indicating real tokens vs padding
            - cached_prompt: Voice preset embeddings
            - Other model-specific inputs

        Raises:
            RuntimeError: If processor or model not initialized
        """
        if not self.processor or not self.model:
            raise RuntimeError("StreamingTTSService not initialized")

        processor_kwargs = {
            "text": text.strip(),  # Remove leading/trailing whitespace
            "cached_prompt": prefilled_outputs,  # Voice embeddings from cache
            "padding": True,  # Pad sequences to same length
            "return_tensors": "pt",  # Return PyTorch tensors
            "return_attention_mask": True,  # Include attention mask
        }

        processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)

        # Move all tensors to the target device (CPU/GPU/ MPS)
        prepared = {
            key: value.to(self._torch_device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
        return prepared

    def _run_generation(
        self,
        inputs,
        audio_streamer: AudioStreamer,
        errors,
        cfg_scale: float,
        do_sample: bool,
        temperature: float,
        top_p: float,
        refresh_negative: bool,
        prefilled_outputs,
        stop_event: threading.Event,
    ) -> None:
        """
        Run text-to-speech generation in a separate thread.

        This method performs the actual model inference to generate audio.
        It runs in a background thread (started by `stream()`) to allow
        streaming audio chunks back to the client via the AudioStreamer.

        Args:
            inputs: Prepared model inputs from _prepare_inputs()
            audio_streamer: Custom streamer that yields audio chunks
            errors: List to collect exceptions for error reporting
            cfg_scale: Classifier-Free Guidance scale (higher = more voice-like but less natural)
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter (cumulative probability threshold)
            refresh_negative: Whether to refresh negative prompts
            prefilled_outputs: Voice embeddings (deep copied to avoid mutation)
            stop_event: Threading event to signal generation should stop

        The generation process:
        1. Model processes inputs with configured parameters
        2. AudioStreamer receives audio chunks via callback
        3. Chunks are yielded back to the main thread via generator
        4. Errors are captured and re-raised in main thread
        """
        try:
            self.model.generate(
                **inputs,  # Unpack all prepared inputs (input_ids, attention_mask, etc.)
                max_new_tokens=None,  # Let model decide based on stop conditions
                cfg_scale=cfg_scale,  # CFG scale for voice quality
                tokenizer=self.processor.tokenizer,  # Tokenizer for text processing
                generation_config={
                    "do_sample": do_sample,
                    "temperature": temperature
                    if do_sample
                    else 1.0,  # No temp for greedy
                    "top_p": top_p if do_sample else 1.0,  # No top_p for greedy
                },
                audio_streamer=audio_streamer,  # Callback to stream audio chunks
                stop_check_fn=stop_event.is_set,  # Function to check for stop signal
                verbose=False,  # Don't print generation progress
                refresh_negative=refresh_negative,  # Refresh negative prompts
                all_prefilled_outputs=copy.deepcopy(
                    prefilled_outputs
                ),  # Deep copy voice data
            )
        except Exception as exc:  # pragma: no cover - diagnostic logging
            errors.append(exc)  # Capture error for main thread
            traceback.print_exc()  # Print full stack trace to console
            audio_streamer.end()  # Signal streamer to stop

    def stream(
        self,
        text: str,
        cfg_scale: float = 1.5,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.9,
        refresh_negative: bool = True,
        inference_steps: Optional[int] = None,
        voice_key: Optional[str] = None,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        """
        Generate audio from text with streaming output.

        This is a generator function that yields audio chunks as they are produced.
        It runs model inference in a background thread and yields audio data
        through an AudioStreamer callback mechanism.

        Args:
            text: Input text to synthesize
            cfg_scale: Classifier-Free Guidance (1.3-3.0, default 1.5)
            do_sample: Use sampling instead of greedy decoding (default False)
            temperature: Sampling temperature when do_sample=True (default 0.9)
            top_p: Nucleus sampling probability threshold (default 0.9)
            refresh_negative: Refresh negative prompts during generation (default True)
            inference_steps: Override default denoising steps (None = use default)
            voice_key: Specific voice preset to use (None = use default)
            log_callback: Optional function to receive progress events
            stop_event: Threading event to cancel generation

        Yields:
            NumPy array of audio samples at 24kHz sample rate

        The streaming flow:
        1. Load voice preset and prepare inputs
        2. Start generation thread with AudioStreamer callback
        3. AudioStreamer yields chunks → this generator yields to client
        4. Cleanup on completion (stop thread, release resources)
        """
        if not text.strip():
            return  # Empty input, no output
        text = text.replace("’", "'")  # Normalize apostrophe

        # Get voice preset and cached embeddings
        selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)

        def emit(event: str, **payload: Any) -> None:
            """Send log event to callback (if provided)."""
            if log_callback:
                try:
                    log_callback(event, **payload)
                except Exception as exc:
                    print(f"[log_callback] Error while emitting {event}: {exc}")

        # Determine inference steps to use
        steps_to_use = self.inference_steps
        if inference_steps is not None:
            try:
                parsed_steps = int(inference_steps)
                if parsed_steps > 0:
                    steps_to_use = parsed_steps
            except (TypeError, ValueError):
                pass  # Use default if parsing fails
        if self.model:
            self.model.set_ddpm_inference_steps(num_steps=steps_to_use)
        self.inference_steps = steps_to_use

        # Prepare model inputs
        inputs = self._prepare_inputs(text, prefilled_outputs)

        # Create AudioStreamer to handle audio chunk streaming
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list = []
        stop_signal = (
            stop_event or threading.Event()
        )  # Create stop event if not provided

        # Start generation in background thread
        thread = threading.Thread(
            target=self._run_generation,
            kwargs={
                "inputs": inputs,
                "audio_streamer": audio_streamer,
                "errors": errors,
                "cfg_scale": cfg_scale,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "refresh_negative": refresh_negative,
                "prefilled_outputs": prefilled_outputs,
                "stop_event": stop_signal,
            },
            daemon=True,  # Daemon threads don't block program exit
        )
        thread.start()

        generated_samples = 0

        try:
            stream = audio_streamer.get_stream(0)  # Get audio stream from streamer
            for audio_chunk in stream:
                # Convert chunk to numpy array with correct dtype
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                # Flatten to 1D if needed
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)

                # Normalize peak amplitude to 1.0 (prevent clipping)
                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak

                generated_samples += int(audio_chunk.size)
                emit(
                    "model_progress",
                    generated_sec=generated_samples / self.sample_rate,
                    chunk_sec=audio_chunk.size / self.sample_rate,
                )

                # Yield as float32 (will be converted to int16 for WebSocket)
                chunk_to_yield = audio_chunk.astype(np.float32, copy=False)

                yield chunk_to_yield
        finally:
            # Cleanup: stop generation, thread, and streamer
            stop_signal.set()
            audio_streamer.end()
            thread.join()
            if errors:
                emit("generation_error", message=str(errors[0]))
                raise errors[0]  # Re-raise any generation errors

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        """
        Convert audio chunk to PCM16 format for WebSocket transmission.

        Audio is internally processed as float32 in range [-1.0, 1.0].
        This method converts to 16-bit signed integers (PCM16) required by
        most audio systems and WebSocket clients.

        Args:
            chunk: NumPy array of float32 audio samples in range [-1.0, 1.0]

        Returns:
            Bytes of PCM16 (little-endian, 16-bit signed integers)

        Conversion process:
        1. Clip values to [-1.0, 1.0] to prevent overflow
        2. Scale to [-32767, 32767] range
        3. Convert to int16
        4. Return as bytes (little-endian)
        """
        chunk = np.clip(chunk, -1.0, 1.0)  # Prevent clipping artifacts
        pcm = (chunk * 32767.0).astype(np.int16)  # Scale and convert to int16
        return pcm.tobytes()  # Convert to bytes for transmission


# === FASTAPI LIFESPAN: Model initialization on startup ===
# FastAPI lifespan context manager: handles startup model loading and state setup
@asynccontextmanager  # Decorator for async context manager (runs on app startup/shutdown)
async def lifespan(
    app: FastAPI,
):  # Lifespan callback: initializes global state on app start
    model_path = os.environ.get(
        "MODEL_PATH"
    )  # Retrieve required model path from env vars
    if not model_path:  # Validate that MODEL_PATH environment variable is set
        raise RuntimeError("MODEL_PATH not set in environment")  # Fail fast if missing
    device = os.environ.get("MODEL_DEVICE", "cuda")  # Get target device, default "cuda"

    service = StreamingTTSService(
        model_path=model_path, device=device
    )  # Create TTS service instance
    service.load()  # Load processor, model, voices (heavy operation, done once at startup)

    app.state.tts_service = service  # Store service globally in FastAPI app.state
    app.state.model_path = model_path  # Cache model path for logging/info endpoints
    app.state.device = device  # Cache device info
    app.state.websocket_lock = (
        asyncio.Lock()
    )  # Async lock to serialize concurrent generations (single inference at a time)
    print("[startup] Model ready.")  # Confirm startup complete to console

    yield  # Yield to start serving requests (startup phase complete)
    # Implicit cleanup on shutdown (torch garbage collected)


app = FastAPI(lifespan=lifespan)


# Wrapper generator for easy access to TTS service from endpoints
def streaming_tts(
    text: str, **kwargs
) -> Iterator[np.ndarray]:  # Public generator that delegates to service.stream
    service: StreamingTTSService = (
        app.state.tts_service
    )  # Retrieve shared service instance
    yield from service.stream(
        text, **kwargs
    )  # Yield audio chunks from service generator (forwards all kwargs)


# === WEBSOCKET TTS STREAMING ENDPOINT ===
# Core WebSocket handler for real-time TTS: parses params, serializes access, streams PCM16 audio bytes
@app.websocket(
    "/stream"
)  # WebSocket route: clients connect here for streaming synthesis
async def websocket_stream(ws: WebSocket) -> None:  # Async WebSocket endpoint handler
    await ws.accept()  # Handshake: accept the WebSocket connection
    text = ws.query_params.get(
        "text", ""
    )  # Extract synthesis text from URL query params
    print(f"Client connected, text={text!r}")  # Console log: connection + input preview
    cfg_param = ws.query_params.get("cfg")  # CFG scale param (guidance strength)
    steps_param = ws.query_params.get("steps")  # Denoising steps override
    voice_param = ws.query_params.get("voice")  # Voice preset selector

    # Parse CFG scale with safe defaults
    try:
        cfg_scale = (
            float(cfg_param) if cfg_param is not None else 1.5
        )  # Convert string to float (default 1.5)
    except ValueError:  # Invalid float input
        cfg_scale = 1.5  # Reset to default
    if cfg_scale <= 0:  # Sanity check: avoid non-positive scales
        cfg_scale = 1.5  # Enforce minimum

    # Parse inference steps with safe defaults
    try:
        inference_steps = (
            int(steps_param) if steps_param is not None else None
        )  # Convert to int
        if inference_steps is not None and inference_steps <= 0:  # Reject non-positive
            inference_steps = None  # Fall back to service default
    except ValueError:  # Parse failure
        inference_steps = None  # Use default steps

    service: StreamingTTSService = app.state.tts_service  # Retrieve global TTS service
    lock: asyncio.Lock = (
        app.state.websocket_lock
    )  # Async lock for single-generation serialization

    if (
        lock.locked()
    ):  # Prevent concurrent generations: check if lock held by another request
        busy_message = {  # JSON message notifying client of server busy state
            "type": "log",  # Message type: log event
            "event": "backend_busy",  # Specific event for frontend handling
            "data": {
                "message": "Please wait for the other requests to complete."
            },  # Informative text
            "timestamp": get_timestamp(),  # Log timestamp
        }
        print("Please wait for the other requests to complete.")  # Server-side log
        try:
            await ws.send_text(
                json.dumps(busy_message)
            )  # Send busy notification to client
        except Exception:  # Client may have disconnected
            pass
        await ws.close(
            code=1013, reason="Service busy"
        )  # Close WS with standard "try later" code
        return  # Early exit: no processing

    acquired = False  # Flag to track if lock was acquired (for finally release)
    try:
        await lock.acquire()  # Wait for exclusive access (blocks if busy)
        acquired = True  # Mark success

        log_queue: "Queue[Dict[str, Any]]" = (
            Queue()
        )  # Thread-safe queue for buffering log messages (sync from thread → async WS)

        def enqueue_log(
            event: str, **data: Any
        ) -> None:  # Callback to queue log events from generation thread
            log_queue.put({"event": event, "data": data})  # Non-blocking put to queue

        async def flush_logs() -> (
            None
        ):  # Async helper: drain queue and send JSON logs to WS client
            while True:  # Loop until queue empty
                try:
                    entry = log_queue.get_nowait()  # Non-blocking dequeue
                except Empty:  # Queue empty
                    break
                message = {  # Build standardized log JSON
                    "type": "log",  # Fixed type
                    "event": entry.get("event"),  # Log event name
                    "data": entry.get("data", {}),  # Event payload
                    "timestamp": get_timestamp(),  # Current time
                }
                try:
                    await ws.send_text(
                        json.dumps(message)
                    )  # Send batched log to client
                except Exception:  # WS closed or error
                    break  # Stop flushing

        enqueue_log(  # Send initial request-received log
            "backend_request_received",  # Event: params logged
            text_length=len(text or ""),  # Input length
            cfg_scale=cfg_scale,  # Parsed CFG
            inference_steps=inference_steps,  # Parsed steps
            voice=voice_param,  # Voice param
        )

        stop_signal = threading.Event()  # Threading event to signal generation stop

        iterator = (
            streaming_tts(  # Start TTS generator (runs model in background thread)
                text,  # Input text
                cfg_scale=cfg_scale,
                inference_steps=inference_steps,
                voice_key=voice_param,
                log_callback=enqueue_log,  # Pass log callback
                stop_event=stop_signal,  # Pass stop signal
            )
        )
        sentinel = object()  # End-of-iterator marker
        first_ws_send_logged = False  # Flag for first audio chunk log

        await flush_logs()  # Flush initial logs before streaming

        try:
            while (
                ws.client_state == WebSocketState.CONNECTED
            ):  # Stream until client disconnects
                await flush_logs()  # Send pending logs
                chunk = await asyncio.to_thread(
                    next, iterator, sentinel
                )  # Offload sync next() to threadpool (generator blocks on model)
                if chunk is sentinel:  # Generator exhausted
                    break
                chunk = cast(np.ndarray, chunk)  # Type assert numpy array
                payload = service.chunk_to_pcm16(chunk)  # Convert float32 → PCM16 bytes
                await ws.send_bytes(payload)  # Stream raw audio bytes to client
                if not first_ws_send_logged:  # Log first chunk only once
                    first_ws_send_logged = True
                    enqueue_log("backend_first_chunk_sent")  # Progress event
                await flush_logs()  # Flush logs after chunk
        except WebSocketDisconnect:  # Graceful client disconnect
            print("Client disconnected (WebSocketDisconnect)")  # Log
            enqueue_log("client_disconnected")  # Notify via log
            stop_signal.set()  # Abort generation
        except Exception as e:  # Catch-all for generation/streaming errors
            print(f"Error in websocket stream: {e}")  # Log error
            traceback.print_exc()  # Full stack trace
            enqueue_log("backend_error", message=str(e))  # Client notification
            stop_signal.set()  # Stop generation
        finally:  # Always cleanup regardless of exit path
            stop_signal.set()  # Ensure generation stops
            enqueue_log("backend_stream_complete")  # Final log event
            await flush_logs()  # Send completion logs
            try:
                iterator_close = getattr(
                    iterator, "close", None
                )  # Optional generator close
                if callable(iterator_close):
                    iterator_close()  # Close if supported
            except Exception:  # Ignore close errors
                pass
            # Drain remaining queue (discard unsent logs)
            while not log_queue.empty():
                try:
                    log_queue.get_nowait()
                except Empty:
                    break
            try:
                if ws.client_state == WebSocketState.CONNECTED:  # Close if still open
                    await ws.close()
            except Exception as e:
                print(f"Error closing websocket: {e}")  # Log close failure
            print("WS handler exit")  # Debug log
    finally:  # Outer finally: always release lock
        if acquired:  # Only if we acquired it
            lock.release()  # Release lock for next request


@app.get("/")  # Root GET endpoint: serves the frontend UI
def index():  # Handler for serving static HTML page
    return FileResponse(
        BASE / "index.html"
    )  # Serve the web demo's index.html from local dir


@app.get("/config")  # GET endpoint: returns available voices for frontend dropdown
def get_config():  # Returns JSON config with voice list and default
    service: StreamingTTSService = app.state.tts_service  # Access shared service
    voices = sorted(
        service.voice_presets.keys()
    )  # Alphabetically sorted voice preset names
    return {  # JSON-serializable dict for frontend
        "voices": voices,  # List of all available voice keys
        "default_voice": service.default_voice_key,  # Currently selected default voice
    }
