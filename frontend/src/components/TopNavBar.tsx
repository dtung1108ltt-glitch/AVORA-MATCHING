import React from 'react';
import {
  Building2,
  Contrast,
  Moon,
  ShieldCheck,
  Sun,
  UserCheck,
  Briefcase,
  Database,
} from 'lucide-react';
import { AccountId, EmployerAccount, JobOpening } from '../types/matching';
import { MOCK_ACCOUNTS } from '../data/mockMarketData';

interface TopNavBarProps {
  currentAccount: EmployerAccount;
  onSelectAccount: (accountId: AccountId) => void;
  jobs: JobOpening[];
  selectedJob: JobOpening;
  onSelectJob: (job: JobOpening) => void;
  highContrast: boolean;
  onToggleHighContrast: () => void;
  darkMode: boolean;
  onToggleDarkMode: () => void;
}

export const TopNavBar: React.FC<TopNavBarProps> = ({
  currentAccount,
  onSelectAccount,
  jobs,
  selectedJob,
  onSelectJob,
  highContrast,
  onToggleHighContrast,
  darkMode,
  onToggleDarkMode,
}) => {
  return (
    <header
      role="banner"
      aria-label="Thanh điều hướng Quản trị và Chọn vị trí tuyển dụng AVORA-MATCHING"
      className="hc-surface sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur dark:border-slate-800 dark:bg-slate-900/95"
    >
      <div className="mx-auto max-w-7xl px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          {/* Thương hiệu & Thông tin Đơn vị / Thị trường */}
          <div className="flex items-start gap-3.5">
            <div
              className="hc-accent flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white shadow-sm"
              aria-hidden="true"
            >
              <Building2 className="h-6 w-6" />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="inline-flex items-center gap-1 rounded-md bg-blue-50 px-2 py-0.5 text-xs font-bold uppercase tracking-wider text-blue-700 ring-1 ring-inset ring-blue-600/20 dark:bg-blue-950/70 dark:text-blue-300">
                  <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
                  AVORA-MATCHING v2.2
                </span>
                <span className="inline-flex items-center gap-1 rounded-md bg-emerald-50 px-2 py-0.5 text-xs font-semibold text-emerald-700 ring-1 ring-inset ring-emerald-600/20 dark:bg-emerald-950/70 dark:text-emerald-300">
                  <Database className="h-3 w-3" aria-hidden="true" />
                  Thị trường: N = {currentAccount.totalMarketCandidates} Ứng viên | M ={' '}
                  {currentAccount.totalMarketJobs} Công việc
                </span>
              </div>
              <h1 className="mt-1 text-lg font-bold tracking-tight text-slate-900 dark:text-white sm:text-xl">
                {currentAccount.companyName}
              </h1>
              <p className="hc-text-muted text-xs text-slate-600 dark:text-slate-400">
                {currentAccount.marketSummary}
              </p>
            </div>
          </div>

          {/* Khu vực Điều khiển: Bộ chọn Job, Chuyển đổi Tài khoản & Trợ năng (High Contrast / Dark Mode) */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Bộ chọn Vị trí tuyển dụng (JD Selector) */}
            <div className="flex flex-col">
              <label
                htmlFor="job-opening-selector"
                className="mb-1 flex items-center gap-1 text-xs font-semibold text-slate-700 dark:text-slate-300"
              >
                <Briefcase className="h-3.5 w-3.5 text-blue-600 dark:text-blue-400" aria-hidden="true" />
                Vị trí tuyển dụng đang mở (Job ID):
              </label>
              <select
                id="job-opening-selector"
                aria-label="Chọn vị trí tuyển dụng đang mở để lọc và chạy ghép cặp"
                value={selectedJob.code}
                onChange={(e) => {
                  const found = jobs.find((j) => j.code === e.target.value);
                  if (found) onSelectJob(found);
                }}
                className="hc-surface rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-sm font-semibold text-slate-900 shadow-sm transition hover:border-blue-500 focus:border-blue-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              >
                {jobs.map((job) => (
                  <option key={job.code} value={job.code}>
                    [{job.code}] {job.title.slice(0, 42)}... (Quota: {job.quota})
                  </option>
                ))}
              </select>
            </div>

            {/* Bộ chọn Tài khoản Kiểm thử (Admin / Employer Account Switcher) */}
            <div className="flex flex-col">
              <label
                htmlFor="account-role-selector"
                className="mb-1 flex items-center gap-1 text-xs font-semibold text-slate-700 dark:text-slate-300"
              >
                <UserCheck className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" aria-hidden="true" />
                Tài khoản phiên làm việc:
              </label>
              <select
                id="account-role-selector"
                aria-label="Chuyển đổi tài khoản kiểm thử Admin hoặc Nhà tuyển dụng"
                value={currentAccount.id}
                onChange={(e) => onSelectAccount(e.target.value as AccountId)}
                className="hc-surface rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 font-mono text-xs font-bold text-slate-900 shadow-sm transition hover:border-blue-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              >
                {Object.values(MOCK_ACCOUNTS).map((acc) => (
                  <option key={acc.id} value={acc.id}>
                    {acc.id} ({acc.id === 'ADMIN_AVORA_ROOT_01' ? 'Root Admin' : 'Employer GDU'})
                  </option>
                ))}
              </select>
            </div>

            {/* Nút Bật/Tắt Chế độ Tương phản cao (WCAG High Contrast Toggle) */}
            <div className="flex items-center gap-2 pt-4">
              <button
                type="button"
                onClick={onToggleHighContrast}
                aria-pressed={highContrast}
                aria-label={
                  highContrast
                    ? 'Tắt chế độ tương phản cao (WCAG AAA)'
                    : 'Bật chế độ tương phản cao dành cho người thị lực yếu (WCAG AAA)'
                }
                title="Phím tắt: Alt + C | Bật/Tắt chế độ tương phản cao chuẩn WCAG AAA"
                className={`inline-flex items-center gap-2 rounded-lg border px-3.5 py-2 text-xs font-bold transition ${
                  highContrast
                    ? 'hc-accent border-white bg-yellow-400 text-black'
                    : 'border-slate-300 bg-white text-slate-800 hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700'
                }`}
              >
                <Contrast className="h-4 w-4" aria-hidden="true" />
                <span>Tương phản cao: {highContrast ? 'BẬT' : 'TẮT'}</span>
              </button>

              {/* Nút Bật/Tắt Dark Mode */}
              <button
                type="button"
                onClick={onToggleDarkMode}
                aria-pressed={darkMode}
                aria-label={darkMode ? 'Chuyển sang giao diện Sáng' : 'Chuyển sang giao diện Tối (Dark Mode)'}
                title="Chuyển đổi giao diện Sáng / Tối"
                className="hc-surface inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white p-2 text-slate-700 transition hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
              >
                {darkMode ? (
                  <Sun className="h-4 w-4 text-amber-400" aria-hidden="true" />
                ) : (
                  <Moon className="h-4 w-4 text-slate-700" aria-hidden="true" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
