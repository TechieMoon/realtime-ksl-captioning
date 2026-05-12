import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const frameIdRef = useRef(0);
  const intervalRef = useRef<number | null>(null);

  const [caption, setCaption] = useState("연결 대기 중...");
  const [connected, setConnected] = useState(false);

  const [latency, setLatency] = useState(0);
  const [fps] = useState(10);

  const [serverUrl, setServerUrl] = useState(
    "ws://localhost:8000/ws/captions?session_id=demo-1",
  );

  const [sessionId, setSessionId] = useState("demo-1");
  const [token, setToken] = useState("");

  useEffect(() => {
    startWebcam();

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }

      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  async function startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 1280,
          height: 720,
          frameRate: 10,
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("webcam error:", err);
    }
  }

  function connectServer() {
    if (wsRef.current) {
      wsRef.current.close();
    }

    let url = serverUrl;

    if (token.trim()) {
      url += `&token=${token}`;
    }

    const ws = new WebSocket(url);

    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log("connected");

      setConnected(true);

      ws.send(
        JSON.stringify({
          type: "start",
          width: 640,
          height: 360,
          fps: 10,
          format: "jpeg",
          client_name: "desktop-client",
        }),
      );

      intervalRef.current = window.setInterval(() => {
        captureAndSendFrame();
      }, 100);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        console.log(data);

        if (data.type === "caption") {
          setCaption(data.text);

          if (data.latency_ms) {
            setLatency(Math.round(data.latency_ms));
          }
        }
      } catch (err) {
        console.error(err);
      }
    };

    ws.onclose = () => {
      console.log("disconnected");

      setConnected(false);

      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };

    ws.onerror = (err) => {
      console.error(err);
    };

    wsRef.current = ws;
  }

  async function captureAndSendFrame() {
    if (!videoRef.current) return;

    if (!canvasRef.current) return;

    if (!wsRef.current) return;

    if (wsRef.current.readyState !== WebSocket.OPEN) return;

    const canvas = canvasRef.current;

    const ctx = canvas.getContext("2d");

    if (!ctx) return;

    canvas.width = 640;
    canvas.height = 360;

    ctx.drawImage(videoRef.current, 0, 0, 640, 360);

    canvas.toBlob(
      async (blob) => {
        if (!blob) return;

        const jpegBytes = await blob.arrayBuffer();

        sendFramePacket(jpegBytes);
      },
      "image/jpeg",
      0.8,
    );
  }

  function sendFramePacket(jpegBytes: ArrayBuffer) {
    if (!wsRef.current) return;

    const metadata = {
      frame_id: frameIdRef.current++,
      timestamp_ms: performance.now(),
      width: 640,
      height: 360,
      format: "jpeg",
    };

    const metadataBytes = new TextEncoder().encode(JSON.stringify(metadata));

    const packet = new Uint8Array(
      4 + metadataBytes.length + jpegBytes.byteLength,
    );

    const view = new DataView(packet.buffer);

    // big-endian
    view.setUint32(0, metadataBytes.length, false);

    packet.set(metadataBytes, 4);

    packet.set(new Uint8Array(jpegBytes), 4 + metadataBytes.length);

    wsRef.current.send(packet);
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className={`status ${connected ? "connected" : "disconnected"}`}>
          {connected ? "Connected" : "Disconnected"}
        </div>

        <div>Latency: {latency}ms</div>

        <div>FPS: {fps}</div>
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

          <div className="caption-overlay">{caption}</div>
        </section>

        <aside className="control-panel">
          <h2>Server</h2>

          <label>Server URL</label>

          <input
            value={serverUrl}
            onChange={(e) => setServerUrl(e.target.value)}
          />

          <label>Session ID</label>

          <input
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
          />

          <label>Token</label>

          <input value={token} onChange={(e) => setToken(e.target.value)} />

          <button onClick={connectServer}>Connect</button>

          <hr />

          <h2>Overlay</h2>

          <label>Font Size</label>

          <input type="range" min="20" max="60" />

          <label>Opacity</label>

          <input type="range" min="0" max="100" />

          <hr />

          <button>Clean Preview Mode</button>
        </aside>
      </main>

      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}

export default App;
