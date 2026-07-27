/**
 * CameraStream.tsx
 * WebRTC WHEP 即時串流 + canvas 骨架/偵測框疊加
 * - video: cam_in (原始乾淨畫面)
 * - canvas: 從 SSE 接收 YOLO 結果，畫 bbox + 17 點骨架
 * - 無二次推理，推理引擎每 4 幀順手推一次結果
 */
import { useEffect, useRef, useState } from 'react';

const WHEP_URL = 'http://localhost:8889/cam_in/whep';
const DETECT_SSE = '/api/events/live-detection/stream';

// COCO 17 關鍵點骨骼連線定義
const SKELETON: [number, number][] = [
  [0,1],[0,2],[1,3],[2,4],           // 頭部
  [5,6],                              // 肩膀
  [5,7],[7,9],[6,8],[8,10],           // 手臂
  [5,11],[6,12],[11,12],              // 軀幹
  [11,13],[13,15],[12,14],[14,16],    // 腿部
];

interface Person {
  bbox: [number, number, number, number]; // 像素座標 x1 y1 x2 y2
  conf: number;
  kps: [number, number][];               // 正規化 0-1
}

interface Props {
  cameraLabel?: string;
}

export default function CameraStream({ cameraLabel = 'AI 即時監控' }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const rafRef = useRef<number>(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const personsRef = useRef<Person[]>([]);   // 最新一幀偵測結果
  const sseRef = useRef<EventSource | null>(null);

  const [status, setStatus] = useState('正在初始化…');
  const [fps, setFps] = useState('--');
  const [hudVisible, setHudVisible] = useState(true);

  const lastUpdateRef = useRef<number>(0);

  // ── SSE 訂閱偵測結果 ──────────────────────────────────
  useEffect(() => {
    const es = new EventSource(DETECT_SSE);
    sseRef.current = es;
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        personsRef.current = data.persons ?? [];
        lastUpdateRef.current = Date.now();
      } catch {}
    };
    es.onerror = () => {};   // 安靜重連，不影響 UI
    return () => es.close();
  }, []);

  // ── canvas 繪製迴圈 ────────────────────────────────────
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext('2d')!;

    let frameCount = 0;
    let lastFpsTime = performance.now();

    function draw() {
      if (!video || video.paused || video.ended) { rafRef.current = requestAnimationFrame(draw); return; }

      // 同步 canvas 尺寸與顯示尺寸
      if (canvas!.width !== video.clientWidth || canvas!.height !== video.clientHeight) {
        canvas!.width = video.clientWidth;
        canvas!.height = video.clientHeight;
      }
      ctx.clearRect(0, 0, canvas!.width, canvas!.height);

      // 若超過 1200ms 沒收到新影格座標（人物離開畫面），自動清空殘留畫框與骨架
      if (Date.now() - lastUpdateRef.current > 1200) {
        personsRef.current = [];
      }

      const W = canvas!.width;
      const H = canvas!.height;
      const persons = personsRef.current;

      persons.forEach((p) => {
        // ─ bbox（傳入的是 0.0 ~ 1.0 相對座標）
        const [nx1, ny1, nx2, ny2] = p.bbox;
        const rx1 = nx1 * W, ry1 = ny1 * H;
        const rx2 = nx2 * W, ry2 = ny2 * H;

        // 偵測框
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 2;
        ctx.strokeRect(rx1, ry1, rx2 - rx1, ry2 - ry1);

        // 信心度標籤
        ctx.fillStyle = 'rgba(0,0,0,0.65)';
        ctx.fillRect(rx1, ry1 - 18, 80, 18);
        ctx.fillStyle = '#00ff88';
        ctx.font = 'bold 12px monospace';
        ctx.fillText(`person ${p.conf.toFixed(2)}`, rx1 + 4, ry1 - 4);

        // 骨架關鍵點 & 連線
        if (p.kps && p.kps.length >= 17) {
          // 連線（青色）
          ctx.strokeStyle = 'rgba(0,229,255,0.85)';
          ctx.lineWidth = 2;
          SKELETON.forEach(([a, b]) => {
            const ka = p.kps[a], kb = p.kps[b];
            if (!ka || !kb || (ka[0] === 0 && ka[1] === 0) || (kb[0] === 0 && kb[1] === 0)) return;
            ctx.beginPath();
            ctx.moveTo(ka[0] * W, ka[1] * H);
            ctx.lineTo(kb[0] * W, kb[1] * H);
            ctx.stroke();
          });

          // 關鍵點（黃色）
          p.kps.forEach(([kx, ky]) => {
            if (kx === 0 && ky === 0) return;
            ctx.beginPath();
            ctx.arc(kx * W, ky * H, 3, 0, Math.PI * 2);
            ctx.fillStyle = '#ffdd00';
            ctx.fill();
          });
        }
      });

      // FPS 計數
      frameCount++;
      const now = performance.now();
      if (now - lastFpsTime >= 1000) {
        setFps(((frameCount * 1000) / (now - lastFpsTime)).toFixed(1));
        frameCount = 0;
        lastFpsTime = now;
      }

      rafRef.current = requestAnimationFrame(draw);
    }

    video.addEventListener('playing', () => {
      frameCount = 0; lastFpsTime = performance.now();
      draw();
    });
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  // ── WebRTC WHEP 連線 ───────────────────────────────────
  async function connect() {
    const video = videoRef.current;
    if (!video) return;
    if (pcRef.current) { pcRef.current.close(); pcRef.current = null; }
    video.srcObject = null;
    setStatus('正在連接 MediaMTX…');

    try {
      const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
      pcRef.current = pc;
      pc.addTransceiver('video', { direction: 'recvonly' });
      pc.ontrack = (e) => {
        const stream = e.streams[0] ?? new MediaStream([e.track]);
        video.srcObject = stream;
        video.play().catch(() => {});
      };
      pc.onconnectionstatechange = () => {
        const s = pc.connectionState;
        if (s === 'connected') setStatus('串流中');
        else if (s === 'disconnected' || s === 'failed') { setStatus('斷線，5 秒後重連…'); scheduleReconnect(); }
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await new Promise<void>((resolve) => {
        if (pc.iceGatheringState === 'complete') { resolve(); return; }
        pc.addEventListener('icegatheringstatechange', function h() {
          if (pc.iceGatheringState === 'complete') { pc.removeEventListener('icegatheringstatechange', h); resolve(); }
        });
      });

      const res = await fetch(WHEP_URL, { method: 'POST', headers: { 'Content-Type': 'application/sdp' }, body: pc.localDescription!.sdp });
      if (!res.ok) throw new Error(`WHEP ${res.status}`);
      await pc.setRemoteDescription({ type: 'answer', sdp: await res.text() });
      setStatus('載入影像中…');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(`連線失敗：${msg}`);
      scheduleReconnect();
    }
  }

  function scheduleReconnect() {
    if (reconnectTimer.current) return;
    reconnectTimer.current = setTimeout(() => { reconnectTimer.current = null; connect(); }, 5000);
  }

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      cancelAnimationFrame(rafRef.current);
      pcRef.current?.close();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="relative w-full overflow-hidden rounded-2xl bg-black border border-[var(--border)]" style={{ aspectRatio: '16/9' }}>
      {/* 攝影機標籤 */}
      <span className="absolute left-3 top-3 z-20 rounded-md border border-[var(--border)] bg-[var(--bg-surface)]/80 px-2 py-0.5 text-xs text-[var(--text-primary)] backdrop-blur">
        {cameraLabel}
      </span>

      {/* FPS */}
      <span className="absolute right-3 top-3 z-20 rounded-md bg-[rgba(0,180,216,0.85)] px-2 py-0.5 text-xs font-mono font-bold text-white">
        Web FPS: {fps}
      </span>

      {/* WebRTC HUD */}
      {hudVisible && (
        <div className="absolute left-3 top-9 z-20 rounded-md border border-[rgba(0,255,128,0.35)] bg-[rgba(0,0,0,0.72)] px-3 py-2 font-mono text-[10px] leading-snug text-gray-300 backdrop-blur">
          <div className="font-bold text-[#00ff88]">[連線] WebRTC 即時監控</div>
          <div className="text-[#00e5ff]">cam_in  WHEP</div>
          <div>YOLO 骨架 <span className="text-[#00e5ff]">每 4 幀</span></div>
        </div>
      )}

      {/* 狀態列 */}
      <div className="absolute bottom-0 left-0 right-0 z-20 flex items-center justify-between bg-[rgba(0,0,0,0.5)] px-3 py-1 text-xs text-gray-300">
        <span>{status}</span>
        <div className="flex gap-2">
          <button className="rounded bg-[rgba(255,255,255,0.12)] px-2 py-0.5 text-[10px] hover:bg-[rgba(255,255,255,0.22)] transition-colors" onClick={() => setHudVisible(v => !v)}>HUD</button>
          <button className="rounded bg-[rgba(255,255,255,0.12)] px-2 py-0.5 text-[10px] hover:bg-[rgba(255,255,255,0.22)] transition-colors" onClick={connect}>重連</button>
        </div>
      </div>

      {/* 影片 */}
      <video ref={videoRef} autoPlay muted playsInline className="absolute inset-0 h-full w-full object-cover" />

      {/* canvas 疊加層：骨架 + 偵測框 */}
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full pointer-events-none" />
    </div>
  );
}
