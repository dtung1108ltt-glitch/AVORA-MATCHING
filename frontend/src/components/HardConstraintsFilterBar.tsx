import React from 'react';
import * as Slider from '@radix-ui/react-slider';
import {
  SlidersHorizontal,
  Play,
  RotateCcw,
  Accessibility,
  MapPin,
  Briefcase,
  DollarSign,
  Loader2,
  Check,
} from 'lucide-react';
import { HardConstraintsFilterState } from '../types/matching';
import {
  ACCESSIBILITY_SUPPORT_OPTIONS,
  INITIAL_FILTERS,
  LOCATION_FILTER_OPTIONS,
} from '../data/mockMarketData';

interface HardConstraintsFilterBarProps {
  filters: HardConstraintsFilterState;
  onChangeFilters: (next: HardConstraintsFilterState) => void;
  onRunMatching: () => void;
  isRunning: boolean;
  backendSource: 'fastapi-backend' | 'local-gale-shapley';
}

export const HardConstraintsFilterBar: React.FC<HardConstraintsFilterBarProps> = ({
  filters,
  onChangeFilters,
  onRunMatching,
  isRunning,
  backendSource,
}) => {
  const toggleAccessibilityOption = (optionId: string) => {
    const exists = filters.accessibilitySupports.includes(optionId);
    const updated = exists
      ? filters.accessibilitySupports.filter((item) => item !== optionId)
      : [...filters.accessibilitySupports, optionId];
    onChangeFilters({
      ...filters,
      accessibilitySupports: updated,
    });
  };

  const toggleLocationOption = (loc: string) => {
    const exists = filters.locations.includes(loc);
    const updated = exists
      ? filters.locations.filter((item) => item !== loc)
      : [...filters.locations, loc];
    onChangeFilters({
      ...filters,
      locations: updated,
    });
  };

  return (
    <section
      aria-labelledby="hard-constraints-heading"
      className="hc-surface mt-6 rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900"
    >
      <div className="flex flex-col gap-3 border-b border-slate-100 pb-4 dark:border-slate-800 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2.5">
          <div
            className="hc-accent flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600 text-white"
            aria-hidden="true"
          >
            <SlidersHorizontal className="h-4 w-4" />
          </div>
          <div>
            <h2
              id="hard-constraints-heading"
              className="text-base font-bold text-slate-900 dark:text-white"
            >
              2. Bộ lọc Sơ tuyển &amp; Tiêu chí Lọc cứng (Hard Constraints Domain Filter D)
            </h2>
            <p className="hc-text-muted text-xs text-slate-600 dark:text-slate-400">
              Hàm chỉ báo toán học <code className="font-mono">I_D(c, j) ∈ &#123;0, 1&#125;</code>: Loại bỏ ngay từ đầu các cặp vi phạm ngân sách lương, số năm kinh nghiệm hoặc thiếu thiết bị hỗ trợ tiếp cận thiết yếu.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => onChangeFilters(INITIAL_FILTERS)}
            aria-label="Khôi phục bộ lọc điều kiện tiên quyết về mặc định"
            className="hc-surface inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            Đặt lại bộ lọc
          </button>

          <button
            type="button"
            onClick={onRunMatching}
            disabled={isRunning}
            aria-busy={isRunning}
            aria-label="Chạy thuật toán Matching Backend gửi request POST tới API /api/v1/matching/employer-proposing"
            title="Phím tắt: Alt + M | Gửi POST request tới /api/v1/matching/employer-proposing"
            className="hc-accent inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-bold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-70"
          >
            {isRunning ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                <span>Đang chạy Gale-Shapley...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-current" aria-hidden="true" />
                <span>[ Chạy thuật toán Matching Backend ]</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Trạng thái Endpoint & Screen Reader Live Announcement */}
      <div
        role="status"
        aria-live="polite"
        className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg bg-slate-50 px-3.5 py-2 text-xs text-slate-700 dark:bg-slate-800/60 dark:text-slate-300"
      >
        <div className="flex items-center gap-2">
          <span className="font-mono font-bold text-blue-700 dark:text-blue-400">
            POST /api/v1/matching/employer-proposing
          </span>
          <span className="hidden text-slate-400 sm:inline">|</span>
          <span>
            Chế độ: <strong>Employer-Proposing Deferred Acceptance (J-Optimal)</strong>
          </span>
        </div>
        <span className="font-medium">
          Nguồn thực thi:{' '}
          <strong className="text-emerald-700 dark:text-emerald-400">
            {backendSource === 'fastapi-backend'
              ? 'FastAPI Backend (:8000 Live)'
              : 'Gale-Shapley Engine (:8000 Ready / Client Sync)'}
          </strong>
        </span>
      </div>

      {/* Lưới 3 Nhóm Bộ lọc tương tác */}
      <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Cột 1: Mức lương trần (Radix Slider) & Kinh nghiệm tối thiểu (Dropdown) */}
        <div className="space-y-5 lg:col-span-4">
          {/* Salary Slider */}
          <div className="rounded-lg border border-slate-200 p-4 dark:border-slate-800">
            <div className="flex items-center justify-between">
              <label
                id="salary-slider-label"
                htmlFor="salary-slider-thumb"
                className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-slate-700 dark:text-slate-200"
              >
                <DollarSign className="h-4 w-4 text-emerald-600" aria-hidden="true" />
                Ngân sách lương tối đa:
              </label>
              <span className="rounded-md bg-emerald-50 px-2.5 py-1 font-mono text-sm font-extrabold text-emerald-700 ring-1 ring-inset ring-emerald-600/20 dark:bg-emerald-950/60 dark:text-emerald-300">
                {filters.maxSalaryMillion} Triệu VNĐ
              </span>
            </div>

            <Slider.Root
              className="relative mt-4 flex h-5 w-full touch-none select-none items-center"
              value={[filters.maxSalaryMillion]}
              min={15}
              max={60}
              step={1}
              aria-labelledby="salary-slider-label"
              onValueChange={(vals) =>
                onChangeFilters({
                  ...filters,
                  maxSalaryMillion: vals[0] ?? 45,
                })
              }
            >
              <Slider.Track className="relative h-2 grow rounded-full bg-slate-200 dark:bg-slate-700">
                <Slider.Range className="hc-accent absolute h-full rounded-full bg-blue-600" />
              </Slider.Track>
              <Slider.Thumb
                id="salary-slider-thumb"
                aria-label={`Ngân sách lương tối đa ${filters.maxSalaryMillion} triệu đồng`}
                className="block h-5 w-5 cursor-pointer rounded-full border-2 border-blue-600 bg-white shadow transition hover:scale-110 focus:outline-none"
              />
            </Slider.Root>

            <div className="mt-2 flex justify-between text-[11px] font-medium text-slate-500 dark:text-slate-400">
              <span>15 Triệu</span>
              <span>30 Triệu</span>
              <span>45 Triệu</span>
              <span>60 Triệu</span>
            </div>
          </div>

          {/* Minimum Experience Dropdown */}
          <div className="rounded-lg border border-slate-200 p-4 dark:border-slate-800">
            <label
              htmlFor="min-experience-select"
              className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-slate-700 dark:text-slate-200"
            >
              <Briefcase className="h-4 w-4 text-blue-600" aria-hidden="true" />
              Kinh nghiệm tối thiểu (Years of Experience):
            </label>
            <select
              id="min-experience-select"
              aria-label="Chọn số năm kinh nghiệm tối thiểu bắt buộc"
              value={filters.minExperienceYears}
              onChange={(e) =>
                onChangeFilters({
                  ...filters,
                  minExperienceYears: Number(e.target.value),
                })
              }
              className="hc-surface mt-2.5 w-full rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-sm font-semibold text-slate-900 shadow-sm transition hover:border-blue-500 dark:border-slate-700 dark:bg-slate-800 dark:text-white"
            >
              <option value={0}>Không yêu cầu tối thiểu (Tất cả Fresher / Senior)</option>
              <option value={1}>Từ 1 năm kinh nghiệm trở lên (&ge; 1 năm)</option>
              <option value={2}>Từ 2 năm kinh nghiệm trở lên (&ge; 2 năm)</option>
              <option value={3}>Từ 3 năm kinh nghiệm trở lên (&ge; 3 năm - Mid/Senior)</option>
              <option value={4}>Từ 4 năm kinh nghiệm trở lên (&ge; 4 năm - Senior/Lead)</option>
            </select>
          </div>
        </div>

        {/* Cột 2: Địa điểm & Hình thức làm việc (Checkboxes) */}
        <div className="rounded-lg border border-slate-200 p-4 dark:border-slate-800 lg:col-span-3">
          <fieldset>
            <legend className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-slate-700 dark:text-slate-200">
              <MapPin className="h-4 w-4 text-indigo-600" aria-hidden="true" />
              Địa điểm &amp; Hình thức làm việc:
            </legend>
            <p className="hc-text-muted mt-1 text-xs text-slate-500 dark:text-slate-400">
              Chọn các khu vực làm việc hoặc cho phép Remote toàn thời gian:
            </p>
            <div className="mt-3 space-y-2.5">
              {LOCATION_FILTER_OPTIONS.map((loc) => {
                const checked = filters.locations.includes(loc);
                const checkboxId = `loc-check-${loc.replace(/\s+/g, '-')}`;
                return (
                  <label
                    key={loc}
                    htmlFor={checkboxId}
                    className={`flex cursor-pointer items-center gap-3 rounded-lg border p-2.5 text-xs font-semibold transition ${
                      checked
                        ? 'border-blue-500 bg-blue-50/60 text-slate-900 dark:border-blue-500/70 dark:bg-blue-950/30 dark:text-white'
                        : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300'
                    }`}
                  >
                    <input
                      id={checkboxId}
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleLocationOption(loc)}
                      className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span>{loc}</span>
                  </label>
                );
              })}

              <label
                htmlFor="only-j-optimal-toggle"
                className="mt-3 flex cursor-pointer items-center gap-3 rounded-lg border border-emerald-200 bg-emerald-50/50 p-2.5 text-xs font-bold text-emerald-900 dark:border-emerald-900/50 dark:bg-emerald-950/30 dark:text-emerald-300"
              >
                <input
                  id="only-j-optimal-toggle"
                  type="checkbox"
                  checked={filters.onlyJOptimal}
                  onChange={(e) =>
                    onChangeFilters({
                      ...filters,
                      onlyJOptimal: e.target.checked,
                    })
                  }
                  className="h-4 w-4 rounded border-emerald-400 text-emerald-600 focus:ring-emerald-500"
                />
                <span>Chỉ hiển thị cặp Ổn định tối ưu (J-Optimal)</span>
              </label>
            </div>
          </fieldset>
        </div>

        {/* Cột 3: Hỗ trợ thiết bị & Điều kiện tiếp cận cho Người khuyết tật (Checkboxes) */}
        <div className="rounded-lg border border-slate-200 p-4 dark:border-slate-800 lg:col-span-5">
          <fieldset>
            <legend className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-slate-700 dark:text-slate-200">
              <Accessibility className="h-4 w-4 text-purple-600" aria-hidden="true" />
              Hỗ trợ Thiết bị &amp; Điều kiện Tiếp cận Người khuyết tật (Essential Access):
            </legend>
            <p className="hc-text-muted mt-1 text-xs text-slate-500 dark:text-slate-400">
              Đánh dấu các trang thiết bị mà công ty GDU Tech cung cấp để khớp với nhu cầu tiếp cận thiết yếu của ứng viên:
            </p>
            <div className="mt-3 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
              {ACCESSIBILITY_SUPPORT_OPTIONS.map((opt) => {
                const checked = filters.accessibilitySupports.includes(opt.id);
                const inputId = `access-opt-${opt.id.replace(/[^a-zA-Z0-9]/g, '-')}`;
                return (
                  <label
                    key={opt.id}
                    htmlFor={inputId}
                    className={`flex cursor-pointer items-start gap-2.5 rounded-lg border p-3 text-xs transition ${
                      checked
                        ? 'border-blue-500 bg-blue-50/60 text-slate-900 dark:border-blue-500/70 dark:bg-blue-950/30 dark:text-white'
                        : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300'
                    }`}
                  >
                    <input
                      id={inputId}
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleAccessibilityOption(opt.id)}
                      className="mt-0.5 h-4 w-4 shrink-0 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                    />
                    <div className="leading-snug">
                      <div className="font-bold">{opt.label}</div>
                      <span className="mt-1 inline-flex items-center gap-1 rounded bg-slate-200/80 px-1.5 py-0.5 text-[10px] font-semibold text-slate-700 dark:bg-slate-800 dark:text-slate-300">
                        <Check className="h-2.5 w-2.5" aria-hidden="true" />
                        Nhóm: {opt.group}
                      </span>
                    </div>
                  </label>
                );
              })}
            </div>
          </fieldset>
        </div>
      </div>
    </section>
  );
};
