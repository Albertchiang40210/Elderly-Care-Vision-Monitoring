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
        .then((media) => {
          if (cancelled) return;
          if (media.clip_url) {
            setResult({ id: eventId, clipUrl: media.clip_url, snapshotUrl: media.snapshot_url, error: false });
          } else {
            // S3 上傳中，每 2 秒輪詢直到影片就緒
            setResult((prev) => ({ ...prev, id: eventId, snapshotUrl: media.snapshot_url, error: false }));
            timer = setTimeout(fetchMedia, 2000);
          }
        })
        .catch(() => {
          if (cancelled) return;
          setResult({ id: eventId, clipUrl: null, snapshotUrl: null, error: true });
        });
    }

    fetchMedia();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [eventId]);

  if (!eventId) {
    return { clipUrl: null, snapshotUrl: null, loading: false, error: false };
  }

  const loading = result.id !== eventId;
  return {
    clipUrl: loading ? null : result.clipUrl,
    snapshotUrl: loading ? null : result.snapshotUrl,
    loading,
    error: loading ? false : result.error,
  };
}
