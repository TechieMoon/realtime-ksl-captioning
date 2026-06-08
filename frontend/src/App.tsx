import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const FRAME_WIDTH = 640;
const FRAME_HEIGHT = 360;
const DEFAULT_FPS = 15;
const DEFAULT_VIRTUAL_CAMERA_DEVICE = "/dev/video20";

type ConnectionMode = "localhost" | "ip";

type RecordedFrame = {
  metadata: {
    frame_id: number;
    timestamp_ms: number;
    width: number;
    height: number;
    format: "jpeg";
    segment_id: string;
  };
  jpegBytes: ArrayBuffer;
};

function isEditableTarget(target: EventTarget | null) {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLSelectElement ||
    target instanceof HTMLTextAreaElement
  );
}

function compactCaptionText(text: string) {
  return text.replace(/\s+/g, " ").trim();
}

function appendWordToCaption(lines: string[], word: string) {
  const next = lines.length > 0 ? [...lines] : [""];
  const lastIndex = next.length - 1;
  next[lastIndex] = compactCaptionText(
    [next[lastIndex], word].filter(Boolean).join(" "),
  );
  return next.slice(-2);
}

function moveToNextSentence(lines: string[]) {
  const next = lines.length > 0 ? [...lines] : [""];
  if (next.length >= 2) {
    return [next[1], ""];
  }
  return [next[0], ""];
}

function visibleCaptionLines(lines: string[]) {
  return lines.map(compactCaptionText).filter(Boolean).slice(-2);
}

function buildFramePacket(metadata: RecordedFrame["metadata"], jpegBytes: ArrayBuffer) {
  const metadataBytes = new TextEncoder().encode(JSON.stringify(metadata));
  const packet = new Uint8Array(4 + metadataBytes.length + jpegBytes.byteLength);
  const view = new DataView(packet.buffer);
  view.setUint32(0, metadataBytes.length, false);
  packet.set(metadataBytes, 4);
  packet.set(new Uint8Array(jpegBytes), 4 + metadataBytes.length);
  return packet;
}

function drawContainedVideo(
  ctx: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  width: number,
  height: number,
) {
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, width, height);

  const sourceWidth = video.videoWidth || width;
  const sourceHeight = video.videoHeight || height;
  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  const x = (width - drawWidth) / 2;
  const y = (height - drawHeight) / 2;
  ctx.drawImage(video, x, y, drawWidth, drawHeight);
}

function wrapCanvasText(
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
) {
  const words = text.split(" ").filter(Boolean);
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (ctx.measureText(candidate).width <= maxWidth) {
      current = candidate;
      continue;
    }
    if (current) {
      lines.push(current);
    }
    if (ctx.measureText(word).width <= maxWidth) {
      current = word;
      continue;
    }

    let chunk = "";
    for (const char of Array.from(word)) {
      const chunkCandidate = `${chunk}${char}`;
      if (ctx.measureText(chunkCandidate).width <= maxWidth) {
        chunk = chunkCandidate;
      } else {
        if (chunk) lines.push(chunk);
        chunk = char;
      }
    }
    current = chunk;
  }

  if (current) lines.push(current);
  return lines;
}

function drawCaptionText(
  ctx: CanvasRenderingContext2D,
  captionLines: string[],
  width: number,
  height: number,
  requestedFontSize: number,
) {
  const lines = visibleCaptionLines(captionLines);
  if (lines.length === 0) return;

  const maxWidth = width * 0.88;
  let fontSize = Math.min(requestedFontSize, 44);
  ctx.font = `900 ${fontSize}px Inter, Arial, sans-serif`;
  let wrapped = lines.flatMap((line) => wrapCanvasText(ctx, line, maxWidth));
  wrapped = wrapped.slice(-4);
  if (wrapped.length === 0) return;

  fontSize = Math.min(fontSize, Math.floor((height * 0.3) / (wrapped.length * 1.25)));
  fontSize = Math.max(18, fontSize);
  const lineHeight = fontSize * 1.25;
  ctx.font = `900 ${fontSize}px Inter, Arial, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = Math.max(5, fontSize * 0.13);
  ctx.fillStyle = "#ffffff";

  const totalHeight = wrapped.length * lineHeight;
  const firstY = height - Math.max(38, height * 0.11) - totalHeight + lineHeight / 2;
  wrapped.forEach((line, index) => {
    const y = firstY + index * lineHeight;
    ctx.strokeText(line, width / 2, y, maxWidth);
    ctx.fillText(line, width / 2, y, maxWidth);
  });
}

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const virtualCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const virtualWsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recordTimerRef = useRef<number | null>(null);
  const virtualTimerRef = useRef<number | null>(null);
  const recordedFramesRef = useRef<RecordedFrame[]>([]);
  const currentSegmentIdRef = useRef<string>("");
  const frameIdRef = useRef(0);
  const virtualFrameIdRef = useRef(0);
  const virtualFrameSendingRef = useRef(false);
  const virtualCameraErrorRef = useRef("");
  const captionLinesRef = useRef<string[]>([""]);
  const fontSizeRef = useRef(34);

  const [captionLines, setCaptionLines] = useState<string[]>([""]);
  const [recognitionStatus, setRecognitionStatus] = useState("자막 대기");
  const [connected, setConnected] = useState(false);
  const [recording, setRecording] = useState(false);
  const [sending, setSending] = useState(false);
  const [latency, setLatency] = useState(0);
  const [recordedFrames, setRecordedFrames] = useState(0);
  const [fontSize, setFontSize] = useState(34);
  const [fps, setFps] = useState(DEFAULT_FPS);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [cameraStatus, setCameraStatus] = useState("카메라 대기");

  const [connectionMode, setConnectionMode] =
    useState<ConnectionMode>("localhost");
  const [serverHost, setServerHost] = useState("127.0.0.1");
  const [serverPort, setServerPort] = useState("8000");
  const [sessionId, setSessionId] = useState("demo-1");
  const [token, setToken] = useState("");
  const [serverStatus, setServerStatus] = useState("서버 미연결");
  const [virtualCameraDevice, setVirtualCameraDevice] = useState(
    DEFAULT_VIRTUAL_CAMERA_DEVICE,
  );
  const [virtualCameraConnected, setVirtualCameraConnected] = useState(false);
  const [virtualCameraStatus, setVirtualCameraStatus] =
    useState("가상카메라 미연결");

  const resolvedServerHost = useMemo(
    () => (connectionMode === "localhost" ? "localhost" : serverHost.trim()),
    [connectionMode, serverHost],
  );

  const resolvedServerPort = useMemo(
    () => serverPort.trim() || "8000",
    [serverPort],
  );

  const serverUrl = useMemo(() => {
    const params = new URLSearchParams({ session_id: sessionId.trim() });
    if (token.trim()) {
      params.set("token", token.trim());
    }
    return `ws://${resolvedServerHost || "localhost"}:${resolvedServerPort}/ws/captions?${params.toString()}`;
  }, [resolvedServerHost, resolvedServerPort, sessionId, token]);

  const virtualCameraUrl = useMemo(() => {
    const params = new URLSearchParams({
      session_id: `${sessionId.trim() || "demo-1"}-virtual-camera`,
    });
    if (token.trim()) {
      params.set("token", token.trim());
    }
    if (virtualCameraDevice.trim()) {
      params.set("device", virtualCameraDevice.trim());
    }
    return `ws://${resolvedServerHost || "localhost"}:${resolvedServerPort}/ws/virtual-camera?${params.toString()}`;
  }, [
    resolvedServerHost,
    resolvedServerPort,
    sessionId,
    token,
    virtualCameraDevice,
  ]);

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const mediaDevices = await navigator.mediaDevices.enumerateDevices();
    setDevices(mediaDevices.filter((device) => device.kind === "videoinput"));
  }, []);

  const stopRecordingTimer = useCallback(() => {
    if (recordTimerRef.current !== null) {
      window.clearInterval(recordTimerRef.current);
      recordTimerRef.current = null;
    }
  }, []);

  const stopVirtualCameraTimer = useCallback(() => {
    if (virtualTimerRef.current !== null) {
      window.clearInterval(virtualTimerRef.current);
      virtualTimerRef.current = null;
    }
  }, []);

  const stopWebcam = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  const startWebcam = useCallback(
    async (deviceId?: string) => {
      try {
        stopWebcam();
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: fps },
            ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
          },
          audio: false,
        });

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        const [videoTrack] = stream.getVideoTracks();
        setCameraStatus(videoTrack?.label || "카메라 연결됨");
        await refreshDevices();
      } catch (err) {
        console.error("webcam error:", err);
        setCameraStatus("카메라 오류");
      }
    },
    [fps, refreshDevices, stopWebcam],
  );

  const sendFramePacket = useCallback((frame: RecordedFrame) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    ws.send(buildFramePacket(frame.metadata, frame.jpegBytes));
  }, []);

  const sendVirtualCameraFrame = useCallback(async () => {
    const ws = virtualWsRef.current;
    const video = videoRef.current;
    const canvas = virtualCanvasRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !video || !canvas) return;
    if (virtualFrameSendingRef.current) return;
    if (video.readyState < video.HAVE_CURRENT_DATA) return;

    virtualFrameSendingRef.current = true;
    try {
      canvas.width = FRAME_WIDTH;
      canvas.height = FRAME_HEIGHT;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      drawContainedVideo(ctx, video, FRAME_WIDTH, FRAME_HEIGHT);
      drawCaptionText(
        ctx,
        captionLinesRef.current,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        fontSizeRef.current,
      );

      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.86);
      });
      if (!blob || ws.readyState !== WebSocket.OPEN) return;

      ws.send(
        buildFramePacket(
          {
            frame_id: virtualFrameIdRef.current++,
            timestamp_ms: Math.round(performance.now()),
            width: FRAME_WIDTH,
            height: FRAME_HEIGHT,
            format: "jpeg",
            segment_id: "virtual-camera",
          },
          await blob.arrayBuffer(),
        ),
      );
    } finally {
      virtualFrameSendingRef.current = false;
    }
  }, []);

  const captureFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const segmentId = currentSegmentIdRef.current;
    if (!video || !canvas || !segmentId) return;
    if (video.readyState < video.HAVE_CURRENT_DATA) return;

    canvas.width = FRAME_WIDTH;
    canvas.height = FRAME_HEIGHT;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(resolve, "image/jpeg", 0.82);
    });
    if (!blob || currentSegmentIdRef.current !== segmentId) return;

    recordedFramesRef.current.push({
      metadata: {
        frame_id: frameIdRef.current++,
        timestamp_ms: Math.round(performance.now()),
        width: FRAME_WIDTH,
        height: FRAME_HEIGHT,
        format: "jpeg",
        segment_id: segmentId,
      },
      jpegBytes: await blob.arrayBuffer(),
    });
    setRecordedFrames(recordedFramesRef.current.length);
  }, []);

  const startRecording = useCallback(() => {
    if (!connected || sending) return;
    recordedFramesRef.current = [];
    const segmentId = `segment-${Date.now()}`;
    currentSegmentIdRef.current = segmentId;
    setRecordedFrames(0);
    setRecording(true);
    setRecognitionStatus("녹화 중");
    void captureFrame();
    recordTimerRef.current = window.setInterval(
      () => void captureFrame(),
      Math.round(1000 / fps),
    );
  }, [captureFrame, connected, fps, sending]);

  const stopAndSendRecording = useCallback(() => {
    const ws = wsRef.current;
    const segmentId = currentSegmentIdRef.current;
    stopRecordingTimer();
    setRecording(false);
    currentSegmentIdRef.current = "";

    if (!ws || ws.readyState !== WebSocket.OPEN || !segmentId) return;
    const frames = [...recordedFramesRef.current];
    if (frames.length === 0) {
      setRecognitionStatus("프레임 없음");
      return;
    }

    setSending(true);
    setRecognitionStatus("전송 중");
    ws.send(JSON.stringify({ type: "segment_start", segment_id: segmentId }));
    frames.forEach(sendFramePacket);
    ws.send(JSON.stringify({ type: "segment_end", segment_id: segmentId }));
  }, [sendFramePacket, stopRecordingTimer]);

  const toggleRecording = useCallback(() => {
    if (recording) {
      stopAndSendRecording();
    } else {
      startRecording();
    }
  }, [recording, startRecording, stopAndSendRecording]);

  const connectServer = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(serverUrl);
    ws.binaryType = "arraybuffer";
    setServerStatus("서버 연결 중");

    ws.onopen = () => {
      setConnected(true);
      setServerStatus("서버 연결됨");
      ws.send(
        JSON.stringify({
          type: "start",
          width: FRAME_WIDTH,
          height: FRAME_HEIGHT,
          fps,
          format: "jpeg",
          client_name: "desktop-client",
        }),
      );
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "caption") {
          const word = compactCaptionText(data.text || "");
          if (word) {
            setCaptionLines((current) => appendWordToCaption(current, word));
            setRecognitionStatus(`인식: ${word}`);
          } else {
            setRecognitionStatus("결과 없음");
          }
          setSending(false);
          if (data.latency_ms) {
            setLatency(Math.round(data.latency_ms));
          }
        }
        if (data.type === "status") {
          setServerStatus(data.status);
        }
        if (data.type === "error") {
          setRecognitionStatus(data.message || data.code);
          setServerStatus(data.code);
          setSending(false);
        }
      } catch (err) {
        console.error(err);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      setSending(false);
      setServerStatus("서버 미연결");
      stopRecordingTimer();
      setRecording(false);
      currentSegmentIdRef.current = "";
    };

    ws.onerror = (err) => {
      console.error(err);
      setServerStatus("서버 오류");
    };

    wsRef.current = ws;
  }, [fps, serverUrl, stopRecordingTimer]);

  const disconnectServer = useCallback(() => {
    wsRef.current?.close();
  }, []);

  const connectVirtualCamera = useCallback(() => {
    if (virtualWsRef.current) {
      virtualWsRef.current.close();
    }

    virtualCameraErrorRef.current = "";
    const ws = new WebSocket(virtualCameraUrl);
    ws.binaryType = "arraybuffer";
    setVirtualCameraStatus("가상카메라 연결 중");

    ws.onopen = () => {
      setVirtualCameraStatus("가상카메라 서버 연결됨");
      virtualFrameIdRef.current = 0;
      ws.send(
        JSON.stringify({
          type: "start",
          width: FRAME_WIDTH,
          height: FRAME_HEIGHT,
          fps,
          format: "jpeg",
          client_name: "virtual-camera-client",
        }),
      );
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "status") {
          if (data.status === "virtual_camera_started") {
            setVirtualCameraConnected(true);
            setVirtualCameraStatus("가상카메라 시작됨");
            virtualFrameIdRef.current = 0;
            stopVirtualCameraTimer();
            void sendVirtualCameraFrame();
            virtualTimerRef.current = window.setInterval(
              () => void sendVirtualCameraFrame(),
              Math.round(1000 / fps),
            );
            return;
          }
          if (data.status === "virtual_frame_streaming") {
            setVirtualCameraStatus("가상카메라 송출 중");
            return;
          }
          if (data.status === "connected") {
            setVirtualCameraStatus("가상카메라 서버 연결됨");
            return;
          }
          setVirtualCameraStatus(data.status);
        }
        if (data.type === "error") {
          const message = data.message || data.code || "가상카메라 오류";
          virtualCameraErrorRef.current = message;
          setVirtualCameraStatus(message);
          setVirtualCameraConnected(false);
          stopVirtualCameraTimer();
          ws.close();
        }
      } catch (err) {
        console.error(err);
      }
    };

    ws.onclose = () => {
      stopVirtualCameraTimer();
      setVirtualCameraConnected(false);
      if (virtualWsRef.current === ws) {
        virtualWsRef.current = null;
      }
      if (!virtualCameraErrorRef.current) {
        setVirtualCameraStatus("가상카메라 미연결");
      }
    };

    ws.onerror = (err) => {
      console.error(err);
      setVirtualCameraStatus("가상카메라 오류");
    };

    virtualWsRef.current = ws;
  }, [fps, sendVirtualCameraFrame, stopVirtualCameraTimer, virtualCameraUrl]);

  const disconnectVirtualCamera = useCallback(() => {
    virtualCameraErrorRef.current = "";
    const ws = virtualWsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
    ws?.close();
  }, []);

  const startNewSentence = useCallback(() => {
    setCaptionLines((current) => moveToNextSentence(current));
  }, []);

  const clearCaptions = useCallback(() => {
    setCaptionLines([""]);
    setRecognitionStatus("자막 대기");
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void startWebcam(selectedDeviceId);
  }, [selectedDeviceId, startWebcam]);

  useEffect(() => {
    captionLinesRef.current = captionLines;
  }, [captionLines]);

  useEffect(() => {
    fontSizeRef.current = fontSize;
  }, [fontSize]);

  useEffect(() => {
    return () => {
      stopRecordingTimer();
      stopVirtualCameraTimer();
      wsRef.current?.close();
      virtualWsRef.current?.close();
      stopWebcam();
    };
  }, [stopRecordingTimer, stopVirtualCameraTimer, stopWebcam]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (isEditableTarget(event.target)) return;
      if (event.code === "Space") {
        event.preventDefault();
        toggleRecording();
      }
      if (event.code === "Enter") {
        event.preventDefault();
        startNewSentence();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [startNewSentence, toggleRecording]);

  const displayCaptionLines = visibleCaptionLines(captionLines);

  return (
    <div className="app">
      <header className="topbar">
        <div className={`status ${connected ? "connected" : "disconnected"}`}>
          {connected ? "Connected" : "Disconnected"}
        </div>
        <div>Latency: {latency}ms</div>
        <div>FPS: {fps}</div>
        <div>Frames: {recordedFrames}</div>
        <div>Virtual: {virtualCameraConnected ? "On" : "Off"}</div>
      </header>

      <main className="main-layout">
        <section className="preview-section">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="video-preview"
          />

          <div className="recording-badge" data-active={recording}>
            {recording ? "REC" : sending ? "SEND" : "READY"}
          </div>

          <div
            className="caption-overlay"
            style={{
              fontSize: `${fontSize}px`,
            }}
          >
            {displayCaptionLines.map((line, index) => (
              <span className="caption-line" key={`${index}-${line}`}>
                {line}
              </span>
            ))}
          </div>
        </section>

        <aside className="control-panel">
          <section className="control-section">
            <h2>Camera</h2>

            <label htmlFor="camera">Webcam</label>
            <select
              id="camera"
              value={selectedDeviceId}
              onChange={(event) => setSelectedDeviceId(event.target.value)}
            >
              <option value="">Default camera</option>
              {devices.map((device, index) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || `Camera ${index + 1}`}
                </option>
              ))}
            </select>

            <div className="field-note">{cameraStatus}</div>
          </section>

          <section className="control-section">
            <h2>Server</h2>

            <label htmlFor="connection-mode">Host Mode</label>
            <select
              id="connection-mode"
              value={connectionMode}
              onChange={(event) =>
                setConnectionMode(event.target.value as ConnectionMode)
              }
            >
              <option value="localhost">localhost</option>
              <option value="ip">IP address</option>
            </select>

            <label htmlFor="server-host">Host / IP</label>
            <input
              id="server-host"
              value={serverHost}
              disabled={connectionMode === "localhost"}
              onChange={(event) => setServerHost(event.target.value)}
            />

            <label htmlFor="server-port">Port</label>
            <input
              id="server-port"
              value={serverPort}
              inputMode="numeric"
              onChange={(event) => setServerPort(event.target.value)}
            />

            <label htmlFor="session-id">Session ID</label>
            <input
              id="session-id"
              value={sessionId}
              onChange={(event) => setSessionId(event.target.value)}
            />

            <label htmlFor="token">Token</label>
            <input
              id="token"
              value={token}
              onChange={(event) => setToken(event.target.value)}
            />

            <div className="button-row">
              <button onClick={connectServer} disabled={connected}>
                Connect
              </button>
              <button onClick={disconnectServer} disabled={!connected}>
                Disconnect
              </button>
            </div>

            <div className="field-note">{serverStatus}</div>
          </section>

          <section className="control-section">
            <h2>Virtual Camera</h2>

            <label htmlFor="virtual-camera-device">Device</label>
            <input
              id="virtual-camera-device"
              value={virtualCameraDevice}
              onChange={(event) => setVirtualCameraDevice(event.target.value)}
            />

            <div className="button-row">
              <button
                onClick={connectVirtualCamera}
                disabled={virtualCameraConnected}
              >
                Start
              </button>
              <button
                onClick={disconnectVirtualCamera}
                disabled={!virtualCameraConnected}
              >
                Stop
              </button>
            </div>

            <div className="field-note">{virtualCameraStatus}</div>
          </section>

          <section className="control-section">
            <h2>Capture</h2>

            <label htmlFor="fps">FPS: {fps}</label>
            <input
              id="fps"
              type="range"
              min="5"
              max="20"
              value={fps}
              onChange={(event) => setFps(Number(event.target.value))}
            />

            <button
              className="record-button"
              onClick={toggleRecording}
              disabled={!connected || sending}
              data-recording={recording}
            >
              {recording ? "Stop & Send" : "Record Word"}
            </button>
          </section>

          <section className="control-section">
            <h2>Overlay</h2>

            <label htmlFor="font-size">Font Size: {fontSize}px</label>
            <input
              id="font-size"
              type="range"
              min="20"
              max="80"
              value={fontSize}
              onChange={(event) => setFontSize(Number(event.target.value))}
            />

            <div className="button-row">
              <button onClick={startNewSentence}>New Line</button>
              <button onClick={clearCaptions}>Clear</button>
            </div>

            <button onClick={() => setFontSize(34)}>Reset Overlay</button>

            <div className="caption-buffer" aria-live="polite">
              {displayCaptionLines.map((line, index) => (
                <div key={`${index}-${line}`}>{line}</div>
              ))}
            </div>

            <div className="field-note">{recognitionStatus}</div>
          </section>
        </aside>
      </main>

      <canvas ref={canvasRef} className="hidden-canvas" />
      <canvas ref={virtualCanvasRef} className="hidden-canvas" />
    </div>
  );
}

export default App;
