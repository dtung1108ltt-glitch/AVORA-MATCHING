import React, { useEffect, useMemo, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { CheckCircle2, Keyboard } from 'lucide-react';
import {
  AccountId,
  CandidateMatchItem,
  HardConstraintsFilterState,
  JobOpening,
  MatchingKpiMetrics,
} from './types/matching';
import {
  INITIAL_FILTERS,
  INITIAL_KPI_METRICS,
  MOCK_ACCOUNTS,
  MOCK_CANDIDATES,
  MOCK_JOBS,
} from './data/mockMarketData';
import { runEmployerProposingMatching } from './services/matchingApi';
import { TopNavBar } from './components/TopNavBar';
import { KpiMetricCards } from './components/KpiMetricCards';
import { HardConstraintsFilterBar } from './components/HardConstraintsFilterBar';
import { TopKCandidatesTable } from './components/TopKCandidatesTable';
import { ExplainableAiModal } from './components/ExplainableAiModal';
import { InterviewInviteModal } from './components/InterviewInviteModal';
import { MarketAnalyticsPanel } from './components/MarketAnalyticsPanel';

export const App: React.FC = () => {
  // Trạng thái tài khoản Admin / Employer giả lập (ADMIN_AVORA_ROOT_01 hoặc EMP_GDU_TECH_01)
  const [accountId, setAccountId] = useState<AccountId>('ADMIN_AVORA_ROOT_01');
  const currentAccount = MOCK_ACCOUNTS[accountId];

  // Trạng thái vị trí tuyển dụng đang mở (JD_JAVA_DEV_01 hoặc JD_REACT_DEV_02)
  const [selectedJob, setSelectedJob] = useState<JobOpening>(MOCK_JOBS[0]);

  // Trạng thái bộ lọc cứng (Hard Constraints)
  const [filters, setFilters] = useState<HardConstraintsFilterState>(INITIAL_FILTERS);

  // Danh sách ứng viên và chỉ số KPI
  const [candidatesPool, setCandidatesPool] = useState<CandidateMatchItem[]>(MOCK_CANDIDATES);
  const [kpiMetrics, setKpiMetrics] = useState<MatchingKpiMetrics>(INITIAL_KPI_METRICS);
  const [lastRunTime, setLastRunTime] = useState<string | null>('23:15:00');
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // Trạng thái Trợ năng (Accessibility): High Contrast Mode & Dark Mode
  const [highContrast, setHighContrast] = useState<boolean>(false);
  const [darkMode, setDarkMode] = useState<boolean>(false);

  // Trạng thái Modal Explainable AI & Mời phỏng vấn
  const [selectedXaiCandidate, setSelectedXaiCandidate] = useState<CandidateMatchItem | null>(null);
  const [selectedInviteCandidate, setSelectedInviteCandidate] =
    useState<CandidateMatchItem | null>(null);

  // Đồng bộ class CSS trên thẻ <html> cho High Contrast & Dark Mode
  useEffect(() => {
    const root = document.documentElement;
    if (highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }
    if (darkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [highContrast, darkMode]);

  // Mutation gọi POST /api/v1/matching/employer-proposing qua TanStack Query
  const matchingMutation = useMutation({
    mutationFn: () =>
      runEmployerProposingMatching(selectedJob, filters, candidatesPool),
    onSuccess: (result) => {
      setKpiMetrics(result.kpi);
      setLastRunTime(result.executedAt);
      setToastMessage(
        `Đã thực thi Gale-Shapley J-Optimal (POST /api/v1/matching/employer-proposing) thành công trong ${result.kpi.executionTimeMs} ms — 0% Blocking Pairs!`
      );
      setTimeout(() => setToastMessage(null), 5000);
    },
  });

  // Hỗ trợ phím tắt toàn cục cho người dùng bàn phím (Alt+M chạy thuật toán, Alt+C bật tương phản cao)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.altKey && e.key.toLowerCase() === 'm') {
        e.preventDefault();
        matchingMutation.mutate();
      } else if (e.altKey && e.key.toLowerCase() === 'c') {
        e.preventDefault();
        setHighContrast((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [matchingMutation]);

  // Lọc danh sách ứng viên hiển thị theo thời gian thực dựa trên bộ lọc cứng + từ khóa
  const visibleCandidates = useMemo(() => {
    return candidatesPool
      .filter((cand) => {
        const passSalary = cand.expectedSalaryMillion <= filters.maxSalaryMillion;
        const passExp = cand.experienceYears >= filters.minExperienceYears;
        const passLocation =
          filters.locations.length === 0 ||
          filters.locations.includes(cand.location) ||
          cand.workModePreference === 'remote';
        const passAccess = cand.essentialAccessNeeds.every((need) =>
          filters.accessibilitySupports.includes(need)
        );
        const passJOptimal =
          !filters.onlyJOptimal || cand.stabilityStatus === 'J-Optimal';

        const kw = filters.searchKeyword.trim().toLowerCase();
        const passKeyword =
          !kw ||
          cand.fullName.toLowerCase().includes(kw) ||
          cand.id.toLowerCase().includes(kw) ||
          cand.roleTitle.toLowerCase().includes(kw) ||
          cand.disabilityGroup.toLowerCase().includes(kw) ||
          cand.skills.some((s) => s.toLowerCase().includes(kw));

        return (
          passSalary &&
          passExp &&
          passLocation &&
          passAccess &&
          passJOptimal &&
          passKeyword
        );
      })
      .sort((a, b) => b.scoreJ - a.scoreJ);
  }, [candidatesPool, filters]);

  const handleConfirmInvite = (candidateId: string) => {
    setCandidatesPool((prev) =>
      prev.map((c) =>
        c.id === candidateId
          ? { ...c, invitedToInterview: true, status: 'Interviewing' }
          : c
      )
    );
    setToastMessage(`Đã gửi lời mời phỏng vấn có hỗ trợ tiếp cận tới ứng viên ${candidateId}.`);
    setTimeout(() => setToastMessage(null), 4000);
  };

  return (
    <div className="min-h-screen bg-slate-50 pb-14 text-slate-900 transition-colors dark:bg-slate-950 dark:text-slate-100">
      {/* Liên kết Bỏ qua điều hướng (Skip Link) chuẩn WCAG 2.4.1 khi nhấn phím Tab đầu tiên */}
      <a
        href="#main-dashboard-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-blue-600 focus:px-4 focus:py-2.5 focus:text-sm focus:font-bold focus:text-white focus:shadow-lg"
      >
        Bỏ qua thanh điều hướng để đến thẳng Nội dung chính Dashboard
      </a>

      {/* KHU VỰC 1A: Thanh Điều hướng (Top Bar) */}
      <TopNavBar
        currentAccount={currentAccount}
        onSelectAccount={setAccountId}
        jobs={MOCK_JOBS}
        selectedJob={selectedJob}
        onSelectJob={setSelectedJob}
        highContrast={highContrast}
        onToggleHighContrast={() => setHighContrast((prev) => !prev)}
        darkMode={darkMode}
        onToggleDarkMode={() => setDarkMode((prev) => !prev)}
      />

      {/* Thông báo Trạng thái (ARIA Live Toast) */}
      {toastMessage && (
        <div
          role="alert"
          aria-live="assertive"
          className="mx-auto mt-4 max-w-7xl px-4 sm:px-6 lg:px-8"
        >
          <div className="hc-badge-optimal flex items-center justify-between gap-3 rounded-xl border border-emerald-300 bg-emerald-50 px-4 py-3 text-xs font-bold text-emerald-900 shadow-sm dark:border-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-200">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" aria-hidden="true" />
              <span>{toastMessage}</span>
            </div>
            <button
              type="button"
              onClick={() => setToastMessage(null)}
              className="underline hover:no-underline"
            >
              Đóng
            </button>
          </div>
        </div>
      )}

      {/* Nội dung Chính của Dashboard */}
      <main
        id="main-dashboard-content"
        tabIndex={-1}
        className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8"
      >
        {/* Thanh hướng dẫn nhanh Điều hướng Bàn phím (Keyboard Navigation Guide) */}
        <div
          role="note"
          aria-label="Hướng dẫn điều hướng bằng bàn phím"
          className="hc-surface mt-4 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-xs text-slate-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-400"
        >
          <div className="flex flex-wrap items-center gap-2">
            <Keyboard className="h-4 w-4 text-blue-600 dark:text-blue-400" aria-hidden="true" />
            <span className="font-semibold text-slate-800 dark:text-slate-200">
              Trợ năng Bàn phím (WCAG 2.1 AA/AAA):
            </span>
            <span>
              Dùng <kbd className="rounded border bg-slate-100 px-1.5 py-0.5 font-mono dark:bg-slate-800">Tab</kbd> /{' '}
              <kbd className="rounded border bg-slate-100 px-1.5 py-0.5 font-mono dark:bg-slate-800">Shift+Tab</kbd> để di chuyển,{' '}
              <kbd className="rounded border bg-slate-100 px-1.5 py-0.5 font-mono dark:bg-slate-800">Enter</kbd> /{' '}
              <kbd className="rounded border bg-slate-100 px-1.5 py-0.5 font-mono dark:bg-slate-800">Space</kbd> để chọn,{' '}
              <kbd className="rounded border bg-slate-100 px-1.5 py-0.5 font-mono dark:bg-slate-800">Alt+M</kbd> chạy Matching,{' '}
              <kbd className="rounded border bg-slate-100 px-1.5 py-0.5 font-mono dark:bg-slate-800">Alt+C</kbd> bật/tắt Tương phản cao.
            </span>
          </div>
          <span className="font-mono text-[11px] font-semibold text-blue-700 dark:text-blue-400">
            Active Role: {currentAccount.id}
          </span>
        </div>

        {/* KHU VỰC 1B: 4 Thẻ Thống kê KPI (Metric Cards) */}
        <KpiMetricCards
          metrics={kpiMetrics}
          selectedJob={selectedJob}
          lastRunTime={lastRunTime}
        />

        {/* KHU VỰC 2: Bộ lọc Sơ tuyển & Tiêu chí Lọc cứng (Hard Constraints Filter Bar) */}
        <HardConstraintsFilterBar
          filters={filters}
          onChangeFilters={setFilters}
          onRunMatching={() => matchingMutation.mutate()}
          isRunning={matchingMutation.isPending}
          backendSource={kpiMetrics.source}
        />

        {/* KHU VỰC 3: Bảng danh sách Ứng viên Gợi ý (Top-K Matching Candidates Table) */}
        <TopKCandidatesTable
          candidates={visibleCandidates}
          selectedJob={selectedJob}
          searchKeyword={filters.searchKeyword}
          onSearchChange={(kw) => setFilters((prev) => ({ ...prev, searchKeyword: kw }))}
          onOpenExplainableAi={(cand) => setSelectedXaiCandidate(cand)}
          onOpenInviteModal={(cand) => setSelectedInviteCandidate(cand)}
          isLoading={matchingMutation.isPending}
        />

        {/* Khu vực Biểu đồ Trực quan hóa Điểm 2 Phía (Recharts) */}
        <MarketAnalyticsPanel candidates={visibleCandidates} />
      </main>

      {/* Modal Explainable AI (XAI) */}
      <ExplainableAiModal
        candidate={selectedXaiCandidate}
        job={selectedJob}
        open={Boolean(selectedXaiCandidate)}
        onOpenChange={(open) => {
          if (!open) setSelectedXaiCandidate(null);
        }}
      />

      {/* Modal Mời Phỏng vấn Tiếp cận */}
      <InterviewInviteModal
        candidate={selectedInviteCandidate}
        job={selectedJob}
        open={Boolean(selectedInviteCandidate)}
        onOpenChange={(open) => {
          if (!open) setSelectedInviteCandidate(null);
        }}
        onConfirmInvite={handleConfirmInvite}
      />
    </div>
  );
};

export default App;
