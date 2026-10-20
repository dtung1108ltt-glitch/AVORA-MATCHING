import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { BarChart3 } from 'lucide-react';
import { CandidateMatchItem } from '../types/matching';

interface MarketAnalyticsPanelProps {
  candidates: CandidateMatchItem[];
}

export const MarketAnalyticsPanel: React.FC<MarketAnalyticsPanelProps> = ({
  candidates,
}) => {
  const chartData = candidates.slice(0, 6).map((c) => ({
    id: c.id,
    name: `${c.fullName.split(' ').slice(-2).join(' ')} (${c.id})`,
    Score_J: c.scoreJ,
    Score_C: c.scoreC,
  }));

  return (
    <section
      aria-labelledby="analytics-chart-heading"
      className="hc-surface mt-6 rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900"
    >
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2.5">
          <div
            className="hc-accent flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600 text-white"
            aria-hidden="true"
          >
            <BarChart3 className="h-4 w-4" />
          </div>
          <div>
            <h2
              id="analytics-chart-heading"
              className="text-sm font-bold text-slate-900 dark:text-white"
            >
              Biểu đồ Đối sánh Điểm Ưu tiên Hai phía Score_J(j, c) &amp; Score_C(c, j)
            </h2>
            <p className="hc-text-muted text-xs text-slate-600 dark:text-slate-400">
              Trực quan hóa sự cân bằng lợi ích giữa Nhà tuyển dụng (GDU Tech) và Ứng viên trong thuật toán Gale-Shapley J-Optimal.
            </p>
          </div>
        </div>
      </div>

      <div
        className="mt-4 h-64 w-full"
        role="img"
        aria-label="Biểu đồ cột so sánh điểm phù hợp phía nhà tuyển dụng Score_J và điểm hài lòng phía ứng viên Score_C"
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis domain={[60, 100]} unit="%" tick={{ fontSize: 11 }} />
            <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar
              dataKey="Score_J"
              name="Điểm NTD Score_J(j,c) (%)"
              fill="#10b981"
              radius={[6, 6, 0, 0]}
            />
            <Bar
              dataKey="Score_C"
              name="Điểm Ứng viên Score_C(c,j) (%)"
              fill="#2563eb"
              radius={[6, 6, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
};
