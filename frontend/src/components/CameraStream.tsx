/**
 * CameraStream.tsx
 * WebRTC WHEP 即時串流 + canvas 骨架/偵測框疊加
 * - video: cam_in (原始乾淨畫面)
 * - canvas: 從 SSE 接收 YOLO 結果，畫 bbox + 17 點骨架
 * - 無二次推理，推理引擎每 4 幀順手推一次結果
 */
import { useEffect, useRef, useState } from 'react';

const WHEP_URL = 'http://localhost:8889/cam_in/whep';



// 直接連線至後端 Port 8000 的即時骨架 SSE 推播通道
const DETECT_SSE = 'http://localhost:8000/events/live-detection/stream';

// COCO 17 關鍵點骨骼連線定義


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
  kps?: [number, number][];              // 正規化 0-1
  keypoints?: [number, number][];        // 雙重別名支援
  is_fall?: boolean;
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
  const aiFpsRef = useRef<number>(0);

  const [status, setStatus] = useState('正在初始化…');
  const [fps, setFps] = useState('--');
  const [hudVisible, setHudVisible] = useState(true);
  const [showSkeleton, setShowSkeleton] = useState(true);


  const lastUpdateRef = useRef<number>(0);

  // ── SSE 訂閱偵測結果 ──────────────────────────────────
  useEffect(() => {
    const es = new EventSource(DETECT_SSE);
    sseRef.current = es;
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        const parsedPersons = data['1'] || data['Room_301_Bed'] || data.persons || (Object.values(data).find(v => Array.isArray(v))) || [];
        personsRef.current = Array.isArray(parsedPersons) ? parsedPersons : [];
        if (data.backend_fps !== undefined) {
          aiFpsRef.current = data.backend_fps;
        }
        lastUpdateRef.current = Date.now();
      } catch (err) {
        console.error("[CameraStream SSE Parse Error]", err);
      }
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
      // Refs 會在元件卸載時清空；使用這兩個已確認非 null 的區域變數，
      // 也避免 TypeScript 在 animation callback 中失去型別收窄。
      const videoElement = video!;
      const canvasElement = canvas!;

      // 同步 canvas 尺寸與顯示尺寸
      if (canvasElement.width !== videoElement.clientWidth || canvasElement.height !== videoElement.clientHeight) {
        canvasElement.width = videoElement.clientWidth;
        canvasElement.height = videoElement.clientHeight;
      }
      ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);

      // 移除 video.paused / clientWidth 檢查，確保只要有 SSE 姿態資料傳入，Canvas 永遠繪製
      if (!canvasElement.clientWidth || !canvasElement.clientHeight) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      // 若超過 1200ms 沒收到新影格座標（人物離開畫面），自動清空殘留畫框與骨架
      if (Date.now() - lastUpdateRef.current > 1200) {
        personsRef.current = [];
      }

      const W = canvasElement.width;
      const H = canvasElement.height;
      const persons = personsRef.current;

      persons.forEach((p) => {
        // ─ bbox（傳入的是 0.0 ~ 1.0 相對座標）
        const [nx1, ny1, nx2, ny2] = p.bbox;
        const rx1 = nx1 * W, ry1 = ny1 * H;
        const rx2 = nx2 * W, ry2 = ny2 * H;
        const bw = rx2 - rx1;
        const bh = ry2 - ry1;

        // 偵測框顏色：跌倒為警告亮紅色 (#ff334b)，無跌倒為健康亮綠色 (#00ff88)
        const boxColor = p.is_fall ? '#ff334b' : '#00ff88';

        // 1. 緊湊型標籤 (Slim Label Tag) - 貼合文字寬度，置於框外頂部避免遮擋頭部
        const labelText = `person ${p.conf.toFixed(2)}`;
        ctx.font = 'bold 13px Arial, sans-serif';
        const textMetrics = ctx.measureText(labelText);
        const labelWidth = textMetrics.width + 10;
        const labelHeight = 20;
        
        // 優先置於畫框上方，若靠頂部邊界才放框內
        const labelY = ry1 - labelHeight >= 0 ? ry1 - labelHeight : ry1;

        // 實心彩色背景標籤
        ctx.fillStyle = boxColor;
        ctx.fillRect(rx1, labelY, labelWidth, labelHeight);

        // 黑色粗體文字
        ctx.fillStyle = '#000000';
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, rx1 + 5, labelY + labelHeight / 2);

        // 2. 外框 Bounding Box (2px 清晰極簡線寬，不干擾視野)
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = 2;
        ctx.strokeRect(rx1, ry1, bw, bh);

        // 4. 骨架關鍵點 & 連線 (100% 強效渲染)
        const kpts = p.kps || p.keypoints;
        if (showSkeleton && kpts && kpts.length >= 17) {
          const getPos = (pt: any): [number, number] => {
            if (!pt) return [0, 0];
            if (Array.isArray(pt)) return [pt[0], pt[1]];
            if (typeof pt === 'object') return [pt.x ?? 0, pt.y ?? 0];
            return [0, 0];
          };

          ctx.strokeStyle = '#00e5ff';
          ctx.lineWidth = 3;
          SKELETON.forEach(([a, b]) => {
            const [ax, ay] = getPos(kpts[a]);
            const [bx, by] = getPos(kpts[b]);
            if ((ax === 0 && ay === 0) || (bx === 0 && by === 0)) return;
            ctx.beginPath();
            ctx.moveTo(ax * W, ay * H);
            ctx.lineTo(bx * W, by * H);
            ctx.stroke();
          });

          kpts.forEach((pt) => {
            const [kx, ky] = getPos(pt);
            if (kx === 0 && ky === 0) return;
            ctx.beginPath();
            ctx.arc(kx * W, ky * H, 4, 0, Math.PI * 2);
            ctx.fillStyle = '#ffdd00';
            ctx.fill();
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 1;
            ctx.stroke();
          });
        }
      });

      // 繪製 AI Inference FPS 於右下角 (仿造老師截圖風格)
      const aiFps = aiFpsRef.current;
      if (aiFps > 0) {
        const fpsText = `FPS: ${aiFps.toFixed(2)}`;
        ctx.font = 'bold 15px Arial';
        const textMetrics = ctx.measureText(fpsText);
        const textWidth = textMetrics.width;
        const padX = 8;
        const padY = 6;
        
        const boxW = textWidth + padX * 2;
        const boxH = 15 + padY * 2;
        const boxX = W - boxW - 10;
        const boxY = H - boxH - 10;
        
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(boxX, boxY, boxW, boxH);
        
        ctx.fillStyle = '#ffffff';
        ctx.textBaseline = 'middle';
        ctx.fillText(fpsText, boxX + padX, boxY + boxH / 2);
      }

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

    // 立即啟動，讓延遲建立的 WebRTC 串流也能被繪製。
    draw();
    return () => cancelAnimationFrame(rafRef.current);
  }, [showSkeleton]);

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
    <div 
      className={`relative w-full overflow-hidden rounded-2xl bg-black transition-all duration-300 ${
        personsRef.current.some(p => p.is_fall) 
          ? 'border-2 border-[#ff334b] shadow-[0_0_25px_rgba(255,51,75,0.5)] animate-pulse' 
          : 'border-2 border-[#00ff88]/60 shadow-[0_0_20px_rgba(0,255,136,0.2)]'
      }`} 
      style={{ aspectRatio: '16/9' }}
    >
      {/* 攝影機名稱標籤 (右上/左上組合) */}
      <div className="absolute left-3 top-3 z-20 flex items-center gap-2">
        <span className="flex items-center gap-1.5 rounded-lg border border-[rgba(0,255,136,0.3)] bg-black/75 px-3 py-1 text-xs font-semibold text-[#00ff88] backdrop-blur-md shadow-lg">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#00ff88] opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-[#00ff88]"></span>
          </span>
          {cameraLabel}
        </span>
      </div>

      {/* 右上角 Web FPS 標籤 */}
      <div className="absolute right-3 top-3 z-20 flex items-center gap-2">
        <span className="rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 px-2.5 py-1 text-xs font-mono font-bold text-white shadow-md backdrop-blur-md border border-cyan-400/40">
          Web FPS: {fps}
        </span>
      </div>

      {/* 左上角 WebRTC 調試 HUD 面板 */}
      {hudVisible && (
        <div className="absolute left-3 top-12 z-20 rounded-xl border border-cyan-500/30 bg-black/80 p-3 font-mono text-[11px] leading-relaxed text-gray-200 backdrop-blur-md shadow-2xl">
          <div className="flex items-center gap-1.5 font-bold text-[#00ff88] border-b border-cyan-500/20 pb-1 mb-1">
            <span className="inline-block w-1.5 h-1.5 bg-[#00ff88] rounded-full"></span>
            [傳輸] WebRTC 即時監控
          </div>
          <div className="text-cyan-300">cam_in WHEP (原始高清鏡頭)</div>

          <div className="text-gray-400">YOLO 骨架 <span className="text-cyan-400 font-semibold">每 4 幀</span></div>
          <div className="text-[#00ff88] mt-1 pt-1 border-t border-cyan-500/20 flex items-center gap-1">
            <span className="text-gray-400">[bbox]</span> 畫面時間軸對齊
          </div>
        </div>
      )}

      {/* 底部高科技狀態條 */}
      <div className="absolute bottom-0 left-0 right-0 z-20 flex items-center justify-between bg-gradient-to-t from-black/90 via-black/60 to-transparent px-4 py-2 text-xs backdrop-blur-[2px]">
        <div className="flex items-center gap-2 font-mono text-gray-300">
          <span className={`inline-block h-2 w-2 rounded-full ${status.includes('串流中') ? 'bg-[#00ff88] shadow-[0_0_8px_#00ff88]' : 'bg-yellow-400 animate-pulse'}`}></span>
          <span>{status}</span>
        </div>
        <div className="flex gap-2">
          <button 
            className="rounded-lg border border-purple-500/30 bg-purple-950/40 px-2.5 py-1 text-[11px] font-medium text-purple-300 hover:bg-purple-500/20 hover:text-white transition-all shadow-sm active:scale-95" 
            onClick={() => setShowSkeleton(v => !v)}
          >
            {showSkeleton ? '隱藏骨架' : '顯示骨架'}
          </button>
          <button 
            className="rounded-lg border border-cyan-500/30 bg-cyan-950/40 px-2.5 py-1 text-[11px] font-medium text-cyan-300 hover:bg-cyan-500/20 hover:text-white transition-all shadow-sm active:scale-95" 
            onClick={() => setHudVisible(v => !v)}
          >
            {hudVisible ? '隱藏 HUD' : '顯示 HUD'}
          </button>
          <button 
            className="rounded-lg border border-emerald-500/30 bg-emerald-950/40 px-2.5 py-1 text-[11px] font-medium text-emerald-300 hover:bg-emerald-500/20 hover:text-white transition-all shadow-sm active:scale-95" 
            onClick={connect}
          >
            重新連線
          </button>
        </div>
      </div>

      {/* 影片串流 */}
      <video ref={videoRef} autoPlay muted playsInline className="absolute inset-0 h-full w-full object-cover" />

      {/* canvas 疊加層：骨架 + 偵測框 */}
      <canvas ref={canvasRef} className="absolute inset-0 z-10 h-full w-full pointer-events-none" />
    </div>
  );
}
