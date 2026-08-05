import React, { useState, useEffect } from 'react';
import CameraStream from '../components/CameraStream';
import { getCameras } from '../api/cameras';
import type { Camera } from '../types';

export default function HazardDetection() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await getCameras();
        setCameras(data);
      } catch (err) {
        console.error('Failed to load cameras:', err);
      } finally {
        setIsLoading(false);
      }
    }
    load();
  }, []);

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <div className="text-xl text-gray-400 font-mono animate-pulse">載入中...</div>
      </div>
    );
  }

  const selectedCamera = cameras.find((c) => c.id.toString() === selectedCameraId);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 fill-mode-both">
      {/* 標題與操作區 */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
            環境安全巡檢
            <span className="ml-3 inline-flex items-center rounded-full border border-[#00c3ff]/30 bg-[#00c3ff]/10 px-2.5 py-0.5 text-xs font-medium text-[#00c3ff]">
              即時監控
            </span>
          </h1>
          <p className="mt-2 text-sm text-gray-400">
            基於 RT-DETR 架構即時辨識走道病床、輪椅等環境物品，避免動線阻塞。
          </p>
        </div>
        
        {/* 手機版下拉選單 */}
        <div className="sm:hidden w-full relative">
          <select
            value={selectedCameraId}
            onChange={(e) => setSelectedCameraId(e.target.value)}
            className="w-full appearance-none rounded-xl border border-gray-800 bg-[var(--bg-surface-1)] py-3 pl-4 pr-10 text-sm font-medium text-gray-200 outline-none ring-1 ring-inset ring-transparent transition-all focus:border-cyan-500 focus:bg-gray-900/50 focus:ring-cyan-500/20"
          >
            <option value="all">所有鏡頭 (四宮格)</option>
            {cameras.map((camera) => (
              <option key={camera.id} value={camera.id}>
                {camera.zone}（{camera.name}）
              </option>
            ))}
          </select>
          <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-4">
            <svg className="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      </div>

      {/* 桌面版鏡頭選擇列 */}
      <div className="hidden sm:block">
        <div className="flex gap-2 p-1 bg-gray-900/50 rounded-xl border border-gray-800/60 overflow-x-auto no-scrollbar">
          <button
            onClick={() => setSelectedCameraId('all')}
            className={`whitespace-nowrap rounded-lg px-4 py-2 text-sm font-medium transition-all duration-200 ${
              selectedCameraId === 'all'
                ? 'bg-cyan-500 text-white shadow-md shadow-cyan-500/20'
                : 'text-gray-400 hover:bg-gray-800/80 hover:text-gray-200'
            }`}
          >
            {selectedCameraId === 'all' && <span className="mr-2 opacity-80">✓</span>}
            所有鏡頭 (四宮格)
          </button>
          {cameras.map((camera) => (
            <button
              key={camera.id}
              onClick={() => setSelectedCameraId(camera.id.toString())}
              className={`whitespace-nowrap rounded-lg px-4 py-2 text-sm font-medium transition-all duration-200 ${
                selectedCameraId === camera.id.toString()
                  ? 'bg-cyan-500 text-white shadow-md shadow-cyan-500/20'
                  : 'text-gray-400 hover:bg-gray-800/80 hover:text-gray-200'
              }`}
            >
              {selectedCameraId === camera.id.toString() && <span className="mr-2 opacity-80">✓</span>}
              {camera.zone} ({camera.name})
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-6">
        {/* 影像顯示區塊 */}
        <div className="col-span-1 min-h-[400px]">
          {selectedCameraId === 'all' || !selectedCamera ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <CameraStream streamId="cam_0" cameraLabel="301 病房 - 床位 A" mode="detr" />
              <CameraStream streamId="cam_1" cameraLabel="301 病房 - 床位 B" mode="detr" />
              <CameraStream streamId="cam_2" cameraLabel="走廊監視器 - 北側" mode="detr" />
              <CameraStream streamId="cam_3" cameraLabel="交誼廳 - 主視角" mode="detr" />
            </div>
          ) : (
            <div className="w-full">
              <CameraStream 
                streamId={`cam_${selectedCamera.id - 1}`}
                cameraLabel={`${selectedCamera.zone} - ${selectedCamera.name}`} 
                mode="detr"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
