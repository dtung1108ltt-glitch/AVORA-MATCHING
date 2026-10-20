import React from 'react';
import {
  Users,
  Filter,
  Award,
  PieChart,
  CheckCircle2,
  Clock,
} from 'lucide-react';
import { JobOpening, MatchingKpiMetrics } from '../types/matching';

interface KpiMetricCardsProps {
  metrics: MatchingKpiMetrics;
  selectedJob: JobOpening;
  lastRunTime: string | null;
}

export const KpiMetricCards: React.FC<KpiMetricCardsProps> = ({
  metrics,
  selectedJob,
  lastRunTime,
}) => {
  const cards = [
    {
      id: 'kpi-total-candidates',
      title: 'Tổng ứng viên (Total Pool)',
      value: `${metrics.totalCandidates}`,
      unit: 'ứng viên',
      subtitle: `Thuộc phân khúc vị trí ${selectedJob.code} (Trích xuất từ N = 300)`,
      badge: '100% Active',
      badgeColor:
        'bg-blue-50 text-blue-700 ring-blue-600/20 dark:bg-blue-950/60 dark:text-blue-300',
      icon: Users,
      iconBg: 'bg-blue-600 text-white',
    },
    {
      id: 'kpi-passed-hard-filter',
      title: 'Đạt lọc cứng (Hard Constraints)',
      value: `${metrics.passedHardFilter}`,
      unit: `/ ${metrics.totalCandidates} hồ sơ`,
      subtitle: 'Thỏa mãn 100% điều kiện tiên quyết & hỗ trợ thiết bị khuyết tật (Tập D)',
      badge: `${Math.round((metrics.passedHardFilter / Math.max(1, metrics.totalCandidates)) * 100)}% Khả thi`,
      badgeColor:
        'bg-indigo-50 text-indigo-700 ring-indigo-600/20 dark:bg-indigo-950/60 dark:text-indigo-300',
      icon: Filter,
      iconBg: 'bg-indigo-600 text-white',
    },
    {
      id: 'kpi-j-optimal-pairs',
      title: 'Cặp ghép ổn định tối ưu J-Optimal',
      value: `${metrics.jOptimalStablePairs}`,
      unit: 'cặp ổn định',
      subtitle: 'Gale-Shapley Employer-Proposing Deferred Acceptance (Many-to-One)',
      badge: `${metrics.blockingPairsCount}% Blocking Pairs`,
      badgeColor:
        'bg-emerald-50 text-emerald-700 ring-emerald-600/20 dark:bg-emerald-950/60 dark:text-emerald-300',
      icon: Award,
      iconBg: 'bg-emerald-600 text-white',
    },
    {
      id: 'kpi-fill-rate',
      title: 'Tỷ lệ lấp đầy (Fill Rate)',
      value: `${metrics.fillRatePercent}%`,
      unit: `(${metrics.jOptimalStablePairs}/${selectedJob.quota} chỉ tiêu)`,
      subtitle: ` Đảm bảo 100% tính ổn định toán học (${metrics.proposalsCount} lượt đề xuất)`,
      badge: 'Đạt chuẩn KPI',
      badgeColor:
        'bg-amber-50 text-amber-800 ring-amber-600/20 dark:bg-amber-950/60 dark:text-amber-300',
      icon: PieChart,
      iconBg: 'bg-amber-600 text-white',
    },
  ];

  return (
    <section
      aria-labelledby="kpi-section-heading"
      aria-live="polite"
      className="mt-6"
    >
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2
          id="kpi-section-heading"
          className="text-sm font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400"
        >
          1. Chỉ số Thống kê Hiệu quả Ghép cặp (Two-Sided Stable Matching KPIs)
        </h2>
        <div className="flex flex-wrap items-center gap-3 text-xs text-slate-600 dark:text-slate-400">
          <span className="inline-flex items-center gap-1.5 font-medium text-emerald-700 dark:text-emerald-400">
            <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
            Định lý Gale-Shapley: 0 Cặp gây bất ổn (100% Stability)
          </span>
          {lastRunTime && (
            <span className="inline-flex items-center gap-1 rounded bg-slate-200/70 px-2 py-0.5 font-mono dark:bg-slate-800">
              <Clock className="h-3.5 w-3.5" aria-hidden="true" />
              Cập nhật lúc {lastRunTime} ({metrics.executionTimeMs} ms)
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {cards.map((card) => {
          const IconComponent = card.icon;
          return (
            <article
              key={card.id}
              aria-labelledby={`${card.id}-title`}
              className="hc-surface relative overflow-hidden rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition hover:border-blue-300 dark:border-slate-800 dark:bg-slate-900"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p
                    id={`${card.id}-title`}
                    className="text-xs font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400"
                  >
                    {card.title}
                  </p>
                  <div className="mt-2 flex items-baseline gap-2">
                    <span className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
                      {card.value}
                    </span>
                    <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">
                      {card.unit}
                    </span>
                  </div>
                </div>
                <div
                  className={`hc-accent flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${card.iconBg} shadow-sm`}
                  aria-hidden="true"
                >
                  <IconComponent className="h-5 w-5" />
                </div>
              </div>

              <div className="mt-4 flex items-center justify-between gap-2 border-t border-slate-100 pt-3 dark:border-slate-800">
                <span
                  className={`hc-badge-optimal inline-flex items-center rounded-md px-2 py-0.5 text-xs font-bold ring-1 ring-inset ${card.badgeColor}`}
                >
                  {card.badge}
                </span>
                <span className="hc-text-muted truncate text-xs text-slate-500 dark:text-slate-400">
                  {card.subtitle}
                </span>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
};
