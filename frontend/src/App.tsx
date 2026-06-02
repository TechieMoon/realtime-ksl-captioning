import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const FRAME_WIDTH = 640;
const FRAME_HEIGHT = 360;
const DEFAULT_FPS = 15;

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

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recordTimerRef = useRef<number | null>(null);
  const recordedFramesRef = useRef<RecordedFrame[]>([]);
  const currentSegmentIdRef = useRef<string>("");
  const frameIdRef = useRef(0);

  const [caption, setCaption] = useState("연결 대기 중");
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

  const serverUrl = useMemo(() => {
    const host =
      connectionMode === "localhost" ? "localhost" : serverHost.trim();
    const params = new URLSearchParams({ session_id: sessionId.trim() });
    if (token.trim()) {
      params.set("token", token.trim());
    }
    return `ws://${host || "localhost"}:${serverPort.trim() || "8000"}/ws/captions?${params.toString()}`;
  }, [connectionMode, serverHost, serverPort, sessionId, token]);

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

    const metadataBytes = new TextEncoder().encode(
      JSON.stringify(frame.metadata),
    );
    const packet = new Uint8Array(
      4 + metadataBytes.length + frame.jpegBytes.byteLength,
    );
    const view = new DataView(packet.buffer);
    view.setUint32(0, metadataBytes.length, false);
    packet.set(metadataBytes, 4);
    packet.set(new Uint8Array(frame.jpegBytes), 4 + metadataBytes.length);
    ws.send(packet);
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
    setCaption("녹화 중");
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
      setCaption("프레임 없음");
      return;
    }

    setSending(true);
    setCaption("전송 중");
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
          setCaption(data.text || "결과 없음");
          setSending(false);
          if (data.latency_ms) {
            setLatency(Math.round(data.latency_ms));
          }
        }
        if (data.type === "status") {
          setServerStatus(data.status);
        }
        if (data.type === "error") {
          setCaption(data.message || data.code);
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

  useEffect(() => {
    void startWebcam(selectedDeviceId);
  }, [selectedDeviceId, startWebcam]);

  useEffect(() => {
    return () => {
      stopRecordingTimer();
      wsRef.current?.close();
      stopWebcam();
    };
  }, [stopRecordingTimer, stopWebcam]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target;
      const isEditing =
        target instanceof HTMLInputElement ||
        target instanceof HTMLSelectElement ||
        target instanceof HTMLTextAreaElement;
      if (event.code !== "Space" || isEditing) return;
      event.preventDefault();
      toggleRecording();
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [toggleRecording]);

  return (
    <div className="app">
      <header className="topbar">
        <div className={`status ${connected ? "connected" : "disconnected"}`}>
          {connected ? "Connected" : "Disconnected"}
        </div>
        <div>Latency: {latency}ms</div>
        <div>FPS: {fps}</div>
        <div>Frames: {recordedFrames}</div>
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
            {caption}
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

            <button onClick={() => setFontSize(34)}>Reset Overlay</button>
          </section>
        </aside>
      </main>

      <canvas ref={canvasRef} className="hidden-canvas" />
    </div>
  );
}

export default App;
