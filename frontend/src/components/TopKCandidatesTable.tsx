import React from 'react';
import {
  Search,
  Sparkles,
  CalendarCheck,
  ShieldCheck,
  CheckCircle2,
  Accessibility,
  MapPin,
  Briefcase,
} from 'lucide-react';
import { CandidateMatchItem, JobOpening } from '../types/matching';

interface TopKCandidatesTableProps {
  candidates: CandidateMatchItem[];
  selectedJob: JobOpening;
  searchKeyword: string;
  onSearchChange: (keyword: string) => void;
  onOpenExplainableAi: (candidate: CandidateMatchItem) => void;
  onOpenInviteModal: (candidate: CandidateMatchItem) => void;
  isLoading: boolean;
}

export const TopKCandidatesTable: React.FC<TopKCandidatesTableProps> = ({
  candidates,
  selectedJob,
  searchKeyword,
  onSearchChange,
  onOpenExplainableAi,
  onOpenInviteModal,
  isLoading,
}) => {
  return (
    <section
      aria-labelledby="candidates-table-heading"
      aria-busy={isLoading}
      className="hc-surface mt-6 rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900"
    >
      {/* Header của Bảng + Ô tìm kiếm nhanh */}
      <div className="flex flex-col gap-4 border-b border-slate-200 p-5 dark:border-slate-800 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2
              id="candidates-table-heading"
              className="text-base font-bold text-slate-900 dark:text-white"
            >
              3. Bảng Danh sách Ứng viên Gợi ý (Top-K Matching Candidates Table)
            </h2>
            <span className="hc-badge-optimal inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-bold text-emerald-700 ring-1 ring-inset ring-emerald-600/20 dark:bg-emerald-950/60 dark:text-emerald-300">
              <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
              Ma trận Điểm 2 Phía Score_J(j, c) &amp; Gale-Shapley
            </span>
          </div>
          <p className="hc-text-muted mt-1 text-xs text-slate-600 dark:text-slate-400">
            Kết quả phân bổ vị trí{' '}
            <strong className="font-mono text-blue-700 dark:text-blue-400">
              {selectedJob.code}
            </strong>{' '}
            ({selectedJob.title}) — Đang hiển thị <strong>{candidates.length}</strong> ứng viên phù hợp nhất.
          </p>
        </div>

        {/* Ô tìm kiếm ứng viên theo tên, mã ID hoặc kỹ năng */}
        <div className="relative w-full lg:w-80">
          <label htmlFor="candidate-search-input" className="sr-only">
            Tìm kiếm ứng viên theo tên, mã CAND hoặc kỹ năng
          </label>
          <Search
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
            aria-hidden="true"
          />
          <input
            id="candidate-search-input"
            type="search"
            value={searchKeyword}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Tìm tên, CAND_301, Java, Rust, Qdrant..."
            aria-label="Tìm kiếm ứng viên theo tên, mã định danh hoặc kỹ năng"
            className="hc-surface w-full rounded-lg border border-slate-300 bg-slate-50 py-2 pl-9 pr-3 text-xs font-medium text-slate-900 placeholder:text-slate-400 focus:border-blue-600 dark:border-slate-700 dark:bg-slate-800 dark:text-white"
          />
        </div>
      </div>

      {/* Vùng thông báo Screen Reader (ARIA Live Region) */}
      <div className="sr-only" aria-live="polite" role="status">
        {isLoading
          ? 'Đang tính toán ma trận điểm hai phía và chạy thuật toán Gale-Shapley...'
          : `Bảng kết quả đã cập nhật gồm ${candidates.length} ứng viên cho vị trí ${selectedJob.code}.`}
      </div>

      {/* Bảng dữ liệu chuẩn WCAG */}
      <div className="overflow-x-auto">
        <table
          aria-label={`Bảng danh sách ứng viên gợi ý Top-K cho vị trí ${selectedJob.code}`}
          className="w-full border-collapse text-left text-sm"
        >
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50/90 text-xs font-bold uppercase tracking-wider text-slate-600 dark:border-slate-800 dark:bg-slate-800/70 dark:text-slate-300">
              <th scope="col" className="px-4 py-3.5">
                STT &amp; ID Ứng viên
              </th>
              <th scope="col" className="px-4 py-3.5">
                Điểm phù hợp Score_J(j,c)
              </th>
              <th scope="col" className="px-4 py-3.5">
                Trạng thái Ổn định
              </th>
              <th scope="col" className="px-4 py-3.5">
                Kỹ năng nổi bật &amp; Tiếp cận
              </th>
              <th scope="col" className="px-4 py-3.5 text-right">
                Hành động nhanh
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
            {candidates.length === 0 ? (
              <tr>
                <td
                  colSpan={5}
                  className="px-6 py-12 text-center text-sm text-slate-500 dark:text-slate-400"
                >
                  Không có ứng viên nào thỏa mãn bộ lọc điều kiện cứng hiện tại. Hãy nới lỏng mức lương trần hoặc bật thêm thiết bị hỗ trợ tiếp cận.
                </td>
              </tr>
            ) : (
              candidates.map((cand, index) => {
                const isJOptimal = cand.stabilityStatus === 'J-Optimal';
                return (
                  <tr
                    key={cand.id}
                    className="group transition hover:bg-blue-50/40 dark:hover:bg-slate-800/50"
                  >
                    {/* Cột 1: STT & ID Ứng viên */}
                    <td className="px-4 py-4 align-top">
                      <div className="flex items-start gap-3">
                        <span
                          aria-label={`Số thứ tự ${index + 1}`}
                          className="hc-accent flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-100 font-mono text-xs font-extrabold text-slate-700 dark:bg-slate-800 dark:text-slate-200"
                        >
                          {index + 1}
                        </span>
                        <div>
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="font-bold text-slate-900 dark:text-white">
                              {cand.fullName}
                            </span>
                            <span className="rounded bg-blue-100 px-1.5 py-0.5 font-mono text-xs font-extrabold text-blue-800 dark:bg-blue-950 dark:text-blue-300">
                              {cand.id}
                            </span>
                            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2 py-0.5 text-[11px] font-bold text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300">
                              <span
                                className="h-1.5 w-1.5 rounded-full bg-emerald-600"
                                aria-hidden="true"
                              />
                              {cand.status}
                            </span>
                          </div>
                          <p className="mt-1 text-xs font-medium text-slate-600 dark:text-slate-300">
                            {cand.roleTitle}
                          </p>
                          <div className="hc-text-muted mt-1.5 flex flex-wrap items-center gap-3 text-[11px] text-slate-500 dark:text-slate-400">
                            <span className="inline-flex items-center gap-1">
                              <Accessibility className="h-3.5 w-3.5 text-purple-600 dark:text-purple-400" aria-hidden="true" />
                              {cand.disabilityGroup}
                            </span>
                            <span className="inline-flex items-center gap-1">
                              <Briefcase className="h-3.5 w-3.5" aria-hidden="true" />
                              {cand.experienceYears} năm KN
                            </span>
                            <span className="inline-flex items-center gap-1">
                              <MapPin className="h-3.5 w-3.5" aria-hidden="true" />
                              {cand.location} ({cand.expectedSalaryMillion}Tr)
                            </span>
                          </div>
                        </div>
                      </div>
                    </td>

                    {/* Cột 2: Điểm phù hợp Score_J(j, c) */}
                    <td className="px-4 py-4 align-top">
                      <div className="flex flex-col items-start gap-1.5">
                        <span
                          aria-label={`Điểm phù hợp nhà tuyển dụng ${cand.scoreJ.toFixed(1)} phần trăm`}
                          className="hc-badge-optimal inline-flex items-center gap-1.5 rounded-lg bg-emerald-100 px-2.5 py-1 font-mono text-sm font-extrabold text-emerald-800 ring-1 ring-inset ring-emerald-600/30 dark:bg-emerald-950/90 dark:text-emerald-300"
                        >
                          <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" aria-hidden="true" />
                          {cand.scoreJ.toFixed(1)}%
                        </span>
                        <div className="w-28 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700" aria-hidden="true">
                          <div
                            className="h-1.5 rounded-full bg-emerald-600"
                            style={{ width: `${Math.min(100, cand.scoreJ)}%` }}
                          />
                        </div>
                        <span className="hc-text-muted font-mono text-[11px] text-slate-500 dark:text-slate-400">
                          Score_C: {cand.scoreC.toFixed(1)}% | Surplus: {cand.surplusScore.toFixed(2)}
                        </span>
                      </div>
                    </td>

                    {/* Cột 3: Trạng thái ổn định */}
                    <td className="px-4 py-4 align-top">
                      <div className="flex flex-col items-start gap-1.5">
                        <span
                          className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-bold ring-1 ring-inset ${
                            isJOptimal
                              ? 'hc-badge-optimal bg-emerald-50 text-emerald-800 ring-emerald-600/30 dark:bg-emerald-950/70 dark:text-emerald-300'
                              : 'bg-amber-50 text-amber-800 ring-amber-600/30 dark:bg-amber-950/70 dark:text-amber-300'
                          }`}
                        >
                          <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
                          {cand.stabilityLabel}
                        </span>
                        <span className="hc-text-muted font-mono text-[11px] text-slate-500 dark:text-slate-400">
                          Hạng NTD: #{cand.employerRank} | Hạng ƯV: #{cand.candidateRank}
                        </span>
                      </div>
                    </td>

                    {/* Cột 4: Kỹ năng nổi bật (Tags) */}
                    <td className="px-4 py-4 align-top">
                      <div className="flex max-w-xs flex-wrap gap-1.5" aria-label={`Kỹ năng của ${cand.fullName}`}>
                        {cand.skills.map((skill) => (
                          <span
                            key={skill}
                            className="hc-surface inline-flex items-center rounded-md border border-slate-200 bg-slate-100 px-2 py-0.5 font-mono text-xs font-semibold text-slate-800 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
                          >
                            {skill}
                          </span>
                        ))}
                      </div>
                      <div className="mt-2 flex flex-wrap gap-1">
                        {cand.essentialAccessNeeds.map((need) => (
                          <span
                            key={need}
                            className="inline-flex items-center rounded bg-purple-50 px-1.5 py-0.5 text-[10px] font-semibold text-purple-700 ring-1 ring-inset ring-purple-600/20 dark:bg-purple-950/50 dark:text-purple-300"
                          >
                            Hỗ trợ: {need}
                          </span>
                        ))}
                      </div>
                    </td>

                    {/* Cột 5: Hành động nhanh */}
                    <td className="px-4 py-4 align-top text-right">
                      <div className="flex flex-col items-end gap-2 sm:flex-row sm:justify-end">
                        <button
                          type="button"
                          onClick={() => onOpenInviteModal(cand)}
                          aria-label={`Mời phỏng vấn ứng viên ${cand.fullName} (${cand.id})`}
                          className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-bold shadow-sm transition ${
                            cand.invitedToInterview
                              ? 'bg-emerald-600 text-white hover:bg-emerald-700'
                              : 'hc-accent bg-blue-600 text-white hover:bg-blue-700'
                          }`}
                        >
                          <CalendarCheck className="h-3.5 w-3.5" aria-hidden="true" />
                          <span>
                            {cand.invitedToInterview
                              ? '[ Đã gửi lời mời ]'
                              : '[ Mời phỏng vấn ]'}
                          </span>
                        </button>

                        <button
                          type="button"
                          onClick={() => onOpenExplainableAi(cand)}
                          aria-label={`Xem chi tiết và giải thích AI cho ứng viên ${cand.fullName} (${cand.id})`}
                          className="hc-surface inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-bold text-slate-700 shadow-sm transition hover:border-blue-500 hover:text-blue-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:border-blue-400"
                        >
                          <Sparkles className="h-3.5 w-3.5 text-blue-600 dark:text-blue-400" aria-hidden="true" />
                          <span>[ Xem chi tiết / Explainable AI ]</span>
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
};
