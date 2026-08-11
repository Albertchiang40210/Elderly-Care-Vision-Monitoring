

export function SystemMonitor() {
  return (
    <div className="flex flex-col h-[calc(100vh-80px)] space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500 fill-mode-both">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
          設備效能監控
          <span className="ml-3 inline-flex items-center rounded-full border border-[#00c3ff]/30 bg-[#00c3ff]/10 px-2.5 py-0.5 text-xs font-medium text-[#00c3ff]">
            即時硬體狀態
          </span>
        </h1>
        <p className="mt-2 text-sm text-gray-400">
          透過 Netdata 即時監控邊緣裝置的 CPU、GPU、記憶體與網路資源，確保 AI 跌倒偵測模型穩定運行。
        </p>
      </div>

      <div className="flex-1 w-full rounded-xl overflow-hidden border border-gray-800 bg-[var(--bg-surface-1)] shadow-xl relative">
        {/* 使用 iframe 嵌入完整的 Netdata 原生儀表板 */}
        <iframe
          src="http://localhost:19999"
          title="Netdata System Monitor"
          className="w-full h-full border-none"
          allowFullScreen
        />
        
        {/* 如果需要可以加一個簡單的遮罩避免初次載入的白光，不過 Netdata 預設也是暗色系的 */}
      </div>
    </div>
  );
}

export default SystemMonitor;
