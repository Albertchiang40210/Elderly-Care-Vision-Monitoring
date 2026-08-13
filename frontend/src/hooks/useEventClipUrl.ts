import { useEffect, useState } from 'react';
import { getEventMedia } from '../api/events';

interface ClipResult {
  id: string | undefined;
  clipUrl: string | null;
  snapshotUrl: string | null;
  error: boolean;
}

// 共用 hook：換發事件的限時影片網址，供事件詳情頁／全螢幕警示／誤報確認彈窗共用。
// eventId 換了就重抓；網址約 1 小時失效，故不快取、每次進場重新呼叫。
// loading 狀態靠比對「上次抓回結果所屬的 id」與「目前的 eventId」推得，
// 避免在 effect 內同步呼叫 setState（該作法會觸發 cascading render，ESLint 會擋下來）。
export function useEventClipUrl(eventId: string | undefined) {
  const [result, setResult] = useState<ClipResult>({
    id: undefined,
    clipUrl: null,
    snapshotUrl: null,
    error: false,
  });

  useEffect(() => {
    if (!eventId) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    function fetchMedia() {
      getEventMedia(eventId!)
        .then(async (media) => {
          if (cancelled) return;

          if (!media.clip_url) {
            // S3 上傳中，每 2 秒輪詢直到影片就緒。
            if (!cancelled) {
              setResult((prev) => ({ ...prev, id: eventId, snapshotUrl: media.snapshot_url }));
            }
            timer = setTimeout(fetchMedia, 2000);
            return;
          }

          // 地端推論在事件建立後才會繼續收集／合成後 5 秒影片；此時後端已
          // 回傳本機 clip_path，但檔案還沒產生。先確認檔案可讀，避免 video 收到
          // 一次 404 後就永久顯示「案件片段影像」。
          try {
            const response = await fetch(media.clip_url, { method: 'HEAD', cache: 'no-store' });
            if (!response.ok) throw new Error(`影片尚未就緒 (${response.status})`);
          } catch {
            if (!cancelled) {
              setResult((prev) => ({ ...prev, id: eventId, snapshotUrl: media.snapshot_url }));
              timer = setTimeout(fetchMedia, 2000);
            }
            return;
          }

          if (!cancelled) {
            setResult({ id: eventId, clipUrl: media.clip_url, snapshotUrl: media.snapshot_url, error: false });
          }
        })
        .catch(() => {
          if (cancelled) return;
          setResult({ id: eventId, clipUrl: null, snapshotUrl: null, error: true });
        });
    }

    // 重置為載入中，避免切換事件時看到上一個事件的畫面
    setResult({ id: undefined, clipUrl: null, snapshotUrl: null, error: false });
    fetchMedia();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [eventId]);

  if (!eventId) {
    return { clipUrl: null, snapshotUrl: null, loading: false, error: false };
  }

  // 只要 clipUrl 還沒拿到，就視為 loading，但 snapshotUrl 可以提早顯示
  const loading = result.id !== eventId || result.clipUrl === null;
  return {
    clipUrl: loading ? null : result.clipUrl,
    snapshotUrl: result.snapshotUrl, // 即使 loading 也能回傳快照
    loading,
    error: loading ? false : result.error,
  };
}
