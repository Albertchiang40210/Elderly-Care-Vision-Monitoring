import { useMemo } from 'react';
import { useEvents } from '../hooks/eventsContext';

export function DataAnalysis() {
  const { events } = useEvents();

  const kpiSummary = useMemo(() => {
    const totalHandled = events.filter(e => e.status === 'resolved' || e.verdict === 'false_alarm').length;
    const falseAlarms = events.filter(e => e.verdict === 'false_alarm').length;
    const pendingEvents = events.filter(e => e.status === 'pending' || e.status === 'acknowledged').length;
    
    const far = totalHandled > 0 ? Math.round((falseAlarms / totalHandled) * 1000) / 10 : 0;

    return {
      pending_events: pendingEvents,
      false_positive_rate: far,
      hnp_count: totalHandled,
      hnp_threshold: 0,
    };
  }, [events]);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-[var(--text-primary)]">數據分析 (Data Analysis)</h1>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
        {/* 誤報率卡片 */}
        <div className="flex min-h-[210px] flex-col rounded-2xl border border-[var(--border)] bg-[var(--bg-surface-2)] p-6 shadow-sm">
          <div className="flex flex-[7] flex-col justify-center">
            <p className="text-base text-[var(--text-muted)]">系統誤報率 (FAR)</p>
            <p className="mt-1 text-6xl font-semibold leading-none text-[var(--brand)] drop-shadow-sm">
              {kpiSummary?.false_positive_rate ?? 0}<span className="text-3xl text-[var(--text-muted)]">%</span>
            </p>
          </div>
          <div className="flex flex-[3] items-end justify-between">
            <p className="text-sm text-[var(--text-muted)]">
              基於日常動作 (Hard Negatives) 測試<br/>
              HNP 測試數量: {kpiSummary?.hnp_count ?? 0}
            </p>
          </div>
        </div>

        {/* 待處理事件卡片 */}
        <div className="flex min-h-[210px] flex-col rounded-2xl border border-[var(--border)] bg-[var(--bg-surface-2)] p-6 shadow-sm">
          <div className="flex flex-[7] flex-col justify-center">
            <p className="text-base text-[var(--text-muted)]">未結案事件數量</p>
            <p className="mt-1 text-6xl font-semibold leading-none text-[var(--danger)] drop-shadow-sm">
              {kpiSummary?.pending_events ?? 0}
            </p>
          </div>
          <div className="flex flex-[3] items-end justify-between">
            <p className="text-sm text-[var(--text-muted)]">
              所有尚在追蹤中的警報<br/>包含初步通報與續報階段
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}
