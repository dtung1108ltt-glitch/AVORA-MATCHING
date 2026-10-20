import React, { useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { CalendarCheck, X, Accessibility, Send } from 'lucide-react';
import { CandidateMatchItem, JobOpening } from '../types/matching';

interface InterviewInviteModalProps {
  candidate: CandidateMatchItem | null;
  job: JobOpening;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirmInvite: (candidateId: string, notes: string) => void;
}

export const InterviewInviteModal: React.FC<InterviewInviteModalProps> = ({
  candidate,
  job,
  open,
  onOpenChange,
  onConfirmInvite,
}) => {
  const [interviewFormat, setInterviewFormat] = useState('Online qua Google Meet (Có phụ đề tự động & trình đọc màn hình)');
  const [notes, setNotes] = useState(
    'Công ty Cổ phần Công nghệ GDU Tech trân trọng mời bạn tham dự buổi phỏng vấn trao đổi kỹ thuật với đầy đủ hỗ trợ tiếp cận.'
  );

  if (!candidate) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onConfirmInvite(candidate.id, notes);
    onOpenChange(false);
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-slate-950/60 backdrop-blur-sm" />
        <Dialog.Content
          aria-describedby="invite-modal-desc"
          className="hc-surface fixed left-1/2 top-1/2 z-50 w-[95vw] max-w-lg -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl dark:border-slate-800 dark:bg-slate-900"
        >
          <div className="flex items-start justify-between gap-4 border-b border-slate-200 pb-4 dark:border-slate-800">
            <div className="flex items-center gap-3">
              <div
                className="hc-accent flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-600 text-white"
                aria-hidden="true"
              >
                <CalendarCheck className="h-5 w-5" />
              </div>
              <div>
                <Dialog.Title className="text-base font-bold text-slate-900 dark:text-white">
                  Gửi Lời mời Phỏng vấn Tiếp cận (Accessible Interview)
                </Dialog.Title>
                <Dialog.Description
                  id="invite-modal-desc"
                  className="hc-text-muted text-xs text-slate-600 dark:text-slate-400"
                >
                  Ứng viên: <strong>{candidate.fullName}</strong> ({candidate.id}) — Điểm phù hợp:{' '}
                  <strong>{candidate.scoreJ.toFixed(1)}%</strong>
                </Dialog.Description>
              </div>
            </div>

            <Dialog.Close asChild>
              <button
                type="button"
                aria-label="Đóng cửa sổ mời phỏng vấn"
                className="hc-surface rounded-lg border border-slate-200 p-2 text-slate-500 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </Dialog.Close>
          </div>

          <form onSubmit={handleSubmit} className="mt-4 space-y-4">
            <div className="rounded-lg border border-purple-200 bg-purple-50/60 p-3 text-xs dark:border-purple-900/50 dark:bg-purple-950/30">
              <div className="flex items-center gap-1.5 font-bold text-purple-900 dark:text-purple-300">
                <Accessibility className="h-4 w-4" aria-hidden="true" />
                Hỗ trợ tiếp cận tự động kích hoạt cho nhóm: {candidate.disabilityGroup}
              </div>
              <p className="mt-1 text-slate-700 dark:text-slate-300">
                Thiết bị đã cam kết: {candidate.essentialAccessNeeds.join(', ')}.
              </p>
            </div>

            <div>
              <label
                htmlFor="interview-format-select"
                className="block text-xs font-bold text-slate-700 dark:text-slate-200"
              >
                Hình thức phỏng vấn &amp; Công cụ trợ năng:
              </label>
              <select
                id="interview-format-select"
                value={interviewFormat}
                onChange={(e) => setInterviewFormat(e.target.value)}
                className="hc-surface mt-1.5 w-full rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-900 dark:border-slate-700 dark:bg-slate-800 dark:text-white"
              >
                <option>Online qua Google Meet (Có phụ đề tự động &amp; trình đọc màn hình)</option>
                <option>Phỏng vấn qua văn bản / Chat kỹ thuật (Không yêu cầu gọi thoại)</option>
                <option>Trực tiếp tại GDU Tech (Có lối đi xe lăn &amp; phiên dịch ngôn ngữ ký hiệu)</option>
              </select>
            </div>

            <div>
              <label
                htmlFor="interview-notes-textarea"
                className="block text-xs font-bold text-slate-700 dark:text-slate-200"
              >
                Thông điệp gửi ứng viên ({job.code}):
              </label>
              <textarea
                id="interview-notes-textarea"
                rows={3}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                className="hc-surface mt-1.5 w-full rounded-lg border border-slate-300 bg-slate-50 p-2.5 text-xs text-slate-900 dark:border-slate-700 dark:bg-slate-800 dark:text-white"
              />
            </div>

            <div className="flex items-center justify-end gap-2 pt-2">
              <Dialog.Close asChild>
                <button
                  type="button"
                  className="hc-surface rounded-lg border border-slate-300 px-3.5 py-2 text-xs font-bold text-slate-700 dark:border-slate-700 dark:text-slate-300"
                >
                  Hủy bỏ
                </button>
              </Dialog.Close>
              <button
                type="submit"
                className="hc-accent inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-4 py-2 text-xs font-bold text-white shadow-sm hover:bg-blue-700"
              >
                <Send className="h-3.5 w-3.5" aria-hidden="true" />
                Xác nhận gửi lời mời
              </button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
