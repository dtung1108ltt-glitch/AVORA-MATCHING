import React from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import {
  X,
  Sparkles,
  ShieldCheck,
  CheckCircle2,
  Accessibility,
  Scale,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { CandidateMatchItem, JobOpening } from '../types/matching';

interface ExplainableAiModalProps {
  candidate: CandidateMatchItem | null;
  job: JobOpening;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const ExplainableAiModal: React.FC<ExplainableAiModalProps> = ({
  candidate,
  job,
  open,
  onOpenChange,
}) => {
  if (!candidate) return null;

  const eChartData = [
    {
      name: 'Kỹ năng (35%)',
      score: Math.round(candidate.eBreakdown.skill * 100),
      weight: '0.35',
    },
    {
      name: 'Hình thức & Giờ (20%)',
      score: Math.round(candidate.eBreakdown.work_mode_hours * 100),
      weight: '0.20',
    },
    {
      name: 'Tiếp cận (20%)',
      score: Math.round(candidate.eBreakdown.nice_to_have_access * 100),
      weight: '0.20',
    },
    {
      name: 'Lương (10%)',
      score: Math.round(candidate.eBreakdown.salary * 100),
      weight: '0.10',
    },
    {
      name: 'Kinh nghiệm (10%)',
      score: Math.round(candidate.eBreakdown.experience * 100),
      weight: '0.10',
    },
    {
      name: 'Khu vực (5%)',
      score: Math.round(candidate.eBreakdown.location * 100),
      weight: '0.05',
    },
  ];

  const cChartData = [
    {
      label: 'Thu nhập kỳ vọng (Trọng số 0.50)',
      val: Math.round(candidate.cBreakdown.income * 100),
    },
    {
      label: 'Khoảng cách / Remote (Trọng số 0.30)',
      val: Math.round(candidate.cBreakdown.distance_or_remote * 100),
    },
    {
      label: 'Định hướng nghề nghiệp (Trọng số 0.20)',
      val: Math.round(candidate.cBreakdown.career_path * 100),
    },
  ];

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-slate-950/60 backdrop-blur-sm" />
        <Dialog.Content
          aria-describedby="xai-modal-description"
          className="hc-surface fixed left-1/2 top-1/2 z-50 max-h-[90vh] w-[95vw] max-w-3xl -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl dark:border-slate-800 dark:bg-slate-900"
        >
          <div className="flex items-start justify-between gap-4 border-b border-slate-200 pb-4 dark:border-slate-800">
            <div className="flex items-center gap-3">
              <div
                className="hc-accent flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-white"
                aria-hidden="true"
              >
                <Sparkles className="h-5 w-5" />
              </div>
              <div>
                <Dialog.Title className="text-lg font-bold text-slate-900 dark:text-white">
                  Explainable AI (XAI) — Phân rã Toán học Ghép cặp Hai phía
                </Dialog.Title>
                <Dialog.Description
                  id="xai-modal-description"
                  className="hc-text-muted text-xs text-slate-600 dark:text-slate-400"
                >
                  Ứng viên: <strong>{candidate.fullName}</strong> ({candidate.id}) ↔ Vị trí:{' '}
                  <strong>{job.code}</strong>
                </Dialog.Description>
              </div>
            </div>

            <Dialog.Close asChild>
              <button
                type="button"
                aria-label="Đóng hộp thoại giải thích AI"
                className="hc-surface rounded-lg border border-slate-200 p-2 text-slate-500 transition hover:bg-slate-100 hover:text-slate-800 dark:border-slate-700 dark:hover:bg-slate-800 dark:hover:text-white"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </Dialog.Close>
          </div>

          {/* Lý do giải thích tự động (Mục 6.1 Bước 7) */}
          <div className="mt-4 rounded-xl border border-blue-200 bg-blue-50/70 p-4 dark:border-blue-900/50 dark:bg-blue-950/40">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-blue-800 dark:text-blue-300">
              <ShieldCheck className="h-4 w-4" aria-hidden="true" />
              Giải thích Minh bạch từ Engine (src/core/scoring.py):
            </div>
            <p className="mt-1.5 text-sm leading-relaxed text-slate-800 dark:text-slate-200">
              {candidate.explanation}
            </p>
          </div>

          {/* Tóm tắt điểm 2 chiều */}
          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-slate-200 p-3.5 dark:border-slate-800">
              <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                Điểm Nhà tuyển dụng Score_J(j, c)
              </span>
              <div className="mt-1 font-mono text-2xl font-extrabold text-emerald-600 dark:text-emerald-400">
                {candidate.scoreJ.toFixed(1)}%
              </div>
              <span className="text-[11px] text-slate-500">Hạng ưu tiên tại Job: #{candidate.employerRank}</span>
            </div>

            <div className="rounded-xl border border-slate-200 p-3.5 dark:border-slate-800">
              <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                Điểm Ứng viên Score_C(c, j)
              </span>
              <div className="mt-1 font-mono text-2xl font-extrabold text-blue-600 dark:text-blue-400">
                {candidate.scoreC.toFixed(1)}%
              </div>
              <span className="text-[11px] text-slate-500">Hạng ưu tiên của ƯV: #{candidate.candidateRank}</span>
            </div>

            <div className="rounded-xl border border-slate-200 p-3.5 dark:border-slate-800">
              <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                Trạng thái Gale-Shapley
              </span>
              <div className="mt-1 text-sm font-extrabold text-slate-900 dark:text-white">
                {candidate.stabilityLabel}
              </div>
              <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-600">
                <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
                0 Blocking Pairs
              </span>
            </div>
          </div>

          {/* Biểu đồ Phân rã 6 Trọng số Điểm E (Mục 6.2) */}
          <div className="mt-5 rounded-xl border border-slate-200 p-4 dark:border-slate-800">
            <h3 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
              <Scale className="h-4 w-4 text-blue-600" aria-hidden="true" />
              Phân rã 6 Tiêu chí Điểm E — Nhà tuyển dụng Đánh giá (Mục 6.2)
            </h3>
            <div className="mt-3 h-52 w-full" role="img" aria-label="Biểu đồ cột phân rã 6 trọng số điểm nhà tuyển dụng">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={eChartData} layout="vertical" margin={{ left: 20, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis type="number" domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" width={135} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(value: number) => [`${value}%`, 'Điểm thành phần']} />
                  <Bar dataKey="score" fill="#2563eb" radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Phân rã Điểm C & Điều kiện tiếp cận */}
          <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                Phân rã Điểm C (Sở thích Ứng viên: 0.5 / 0.3 / 0.2)
              </h4>
              <div className="mt-3 space-y-2.5">
                {cChartData.map((item) => (
                  <div key={item.label}>
                    <div className="flex justify-between text-xs font-semibold">
                      <span>{item.label}</span>
                      <span className="font-mono text-blue-600 dark:text-blue-400">{item.val}%</span>
                    </div>
                    <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
                      <div
                        className="h-full rounded-full bg-blue-600"
                        style={{ width: `${item.val}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
              <h4 className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                <Accessibility className="h-4 w-4 text-purple-600" aria-hidden="true" />
                Đối khớp Điều kiện Tiếp cận (Accessibility Match)
              </h4>
              <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                Nhóm khuyết tật: <strong>{candidate.disabilityGroup}</strong>
              </p>
              <ul className="mt-2.5 space-y-1.5 text-xs">
                {candidate.essentialAccessNeeds.map((need) => (
                  <li key={need} className="flex items-center gap-2 font-medium text-emerald-700 dark:text-emerald-400">
                    <CheckCircle2 className="h-4 w-4 shrink-0" aria-hidden="true" />
                    <span>Thiết yếu (Đạt 100%): {need}</span>
                  </li>
                ))}
                {candidate.niceToHaveAccessNeeds.map((nice) => (
                  <li key={nice} className="flex items-center gap-2 text-slate-600 dark:text-slate-300">
                    <CheckCircle2 className="h-4 w-4 shrink-0 text-blue-500" aria-hidden="true" />
                    <span>Khuyến khích (Soft +20%): {nice}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="mt-6 flex justify-end">
            <Dialog.Close asChild>
              <button
                type="button"
                className="hc-accent rounded-lg bg-slate-900 px-4 py-2 text-xs font-bold text-white transition hover:bg-slate-800 dark:bg-blue-600 dark:hover:bg-blue-700"
              >
                Đóng bảng giải thích (Esc)
              </button>
            </Dialog.Close>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
