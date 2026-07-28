import { useEffect, useMemo, useState } from 'react';
import { getCameras, updateCameraName } from '../api/cameras';
import { CameraRow } from '../components/CameraRow';
import { CameraDetailModal } from '../components/CameraDetailModal';
import { CctvGridTile } from '../components/CctvGridTile';
import { ChevronDownIcon } from '../components/icons';
import { useEvents } from '../hooks/eventsContext';
import { groupCamerasByZone } from '../utils/groupCamerasByZone';
import { getDetectingCameraIds } from '../utils/cameraActivity';
import type { Camera } from '../types';

const ALL_ZONES_VALUE = 'all';
const ALL_ZONES_LABEL = '全部區域';

function ColumnHeader({ label, hideOnMobile }: { label: string; hideOnMobile?: boolean }) {
  return (
    <th className={`px-4 py-3 font-medium ${hideOnMobile ? 'hidden sm:table-cell' : ''}`}>
      <span className="inline-flex items-center gap-1">
        {label}
        <ChevronDownIcon className="h-3 w-3" aria-hidden="true" />
      </span>
    </th>
  );
}

export function Monitoring() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [zoneFilter, setZoneFilter] = useState<string>(ALL_ZONES_VALUE);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid'); // 預設九宮格監控牆
  const [selectedCameraId, setSelectedCameraId] = useState<number | null>(null);
  const { events } = useEvents();

  useEffect(() => {
    getCameras().then(setCameras);
  }, []);

  const zoneGroups = groupCamerasByZone(cameras);
  const detectingIds = getDetectingCameraIds(cameras, events);
  const selectedCamera = cameras.find((c) => c.id === selectedCameraId) ?? null;

  const visibleCameras = useMemo(
    () => (zoneFilter === ALL_ZONES_VALUE ? cameras : cameras.filter((c) => c.zone === zoneFilter)),
    [cameras, zoneFilter],
  );

  // 補足 9 格九宮格 (3x3) Tiles
  const nineGridTiles = useMemo(() => {
    const tiles = new Array(9).fill(null);
    for (let i = 0; i < Math.min(9, visibleCameras.length); i++) {
      tiles[i] = visibleCameras[i];
    }
    return tiles;
  }, [visibleCameras]);

  function handleUpdateCameraName(id: number, name: string) {
    setCameras((prev) => prev.map((c) => (c.id === id ? { ...c, name } : c)));
    void updateCameraName(id, name);
  }

  return (
    <div className="flex w-full flex-1 flex-col gap-4">
      {/* 標題欄與檢視模式切換 */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] pb-3">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold text-[var(--text-primary)]">即時監控中心</h1>
          <span className="rounded-full bg-emerald-950 border border-emerald-500/30 px-2.5 py-0.5 text-xs font-medium text-emerald-400">
            3x3 九宮格電視牆
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* 切換九宮格 vs 列表 */}
          <div className="flex items-center rounded-lg border border-[var(--border)] bg-[var(--bg-surface-2)] p-1 text-xs">
            <button
              type="button"
              onClick={() => setViewMode('grid')}
              className={`rounded-md px-3 py-1 font-medium transition-colors ${
                viewMode === 'grid'
                  ? 'bg-[var(--brand)] text-white shadow'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              九宮格 (3x3)
            </button>
            <button
              type="button"
              onClick={() => setViewMode('list')}
              className={`rounded-md px-3 py-1 font-medium transition-colors ${
                viewMode === 'list'
                  ? 'bg-[var(--brand)] text-white shadow'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              列表檢視
            </button>
          </div>

          {/* 區域篩選選單 */}
          <div className="flex items-center gap-1">
            <span aria-hidden="true" className="text-xs text-[var(--text-muted)]">▾</span>
            <select
              value={zoneFilter}
              onChange={(e) => setZoneFilter(e.target.value)}
              className="rounded-md border border-[var(--border)] bg-[var(--bg-surface)] px-2.5 py-1 text-sm text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--brand)]"
            >
              <option value={ALL_ZONES_VALUE}>{ALL_ZONES_LABEL}</option>
              {zoneGroups.map((group) => (
                <option key={group.zone} value={group.zone}>{group.zone}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {visibleCameras.length === 0 ? (
        <p className="text-sm text-[var(--text-muted)] py-8 text-center">目前沒有可顯示的攝影機</p>
      ) : viewMode === 'grid' ? (
        /* 3x3 九宮格電視牆 (9-Grid Layout) */
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 w-full bg-slate-900/60 p-3 rounded-xl border border-slate-800 shadow-2xl">
          {nineGridTiles.map((cam, idx) => (
            <CctvGridTile
              key={cam ? cam.id : `empty-tile-${idx}`}
              camera={cam ?? undefined}
              tileIndex={idx}
              isDetecting={cam ? detectingIds.has(cam.id) : false}
              onSelect={(c) => setSelectedCameraId(c.id)}
            />
          ))}
        </div>
      ) : (
        /* 傳統列表檢視 */
        <div className="overflow-x-auto rounded-xl border border-[var(--border)]">
          <table className="w-full text-left text-sm">
            <thead className="bg-[var(--bg-surface-2)] text-[var(--text-secondary)]">
              <tr>
                <ColumnHeader label="名稱" />
                <ColumnHeader label="區域" hideOnMobile />
                <ColumnHeader label="狀態" />
                <th className="px-4 py-3" aria-hidden="true" />
              </tr>
            </thead>
            <tbody>
              {visibleCameras.map((camera) => (
                <CameraRow
                  key={camera.id}
                  camera={camera}
                  isDetecting={detectingIds.has(camera.id)}
                  onSelect={(c) => setSelectedCameraId(c.id)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* 滿版/大視窗焦點監控 Modal */}
      {selectedCamera && (
        <CameraDetailModal
          camera={selectedCamera}
          isDetecting={detectingIds.has(selectedCamera.id)}
          onClose={() => setSelectedCameraId(null)}
          onNameChange={handleUpdateCameraName}
        />
      )}
    </div>
  );
}

export default Monitoring;
