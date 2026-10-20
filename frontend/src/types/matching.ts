export type AccountId = 'ADMIN_AVORA_ROOT_01' | 'EMP_GDU_TECH_01';

export interface EmployerAccount {
  id: AccountId;
  roleLabel: string;
  companyName: string;
  marketSummary: string;
  totalMarketCandidates: number; // N = 300
  totalMarketJobs: number;       // M = 275
  department: string;
}

export interface JobOpening {
  id: string;
  code: string; // e.g., 'JD_JAVA_DEV_01', 'JD_REACT_DEV_02'
  title: string;
  department: string;
  quota: number;
  filledSlots: number;
  salaryMinMillion: number;
  salaryMaxMillion: number;
  minExperienceYears: number;
  location: string;
  workMode: 'remote' | 'hybrid' | 'on-site';
  workHours: 'full-time' | 'part-time' | 'flexible';
  skillsRequired: string[];
  accessSupportProvided: string[];
  allowedDisabilityGroups: string[];
  totalApplicantsPool: number;     // e.g., 48
  passedHardConstraints: number;   // e.g., 32
  jOptimalMatchedCount: number;    // e.g., 12
  fillRatePercent: number;         // e.g., 75
}

export interface ScoreBreakdownE {
  skill: number;               // Trọng số 0.35
  work_mode_hours: number;     // Trọng số 0.20
  nice_to_have_access: number; // Trọng số 0.20
  salary: number;              // Trọng số 0.10
  experience: number;          // Trọng số 0.10
  location: number;            // Trọng số 0.05
}

export interface ScoreBreakdownC {
  income: number;              // Ưu tiên 1 (0.50)
  distance_or_remote: number;  // Ưu tiên 2 (0.30)
  career_path: number;         // Ưu tiên 3 (0.20)
}

export interface CandidateMatchItem {
  id: string;                  // VD: 'CAND_301', 'CAND_001', 'CAND_002', 'CAND_003'
  fullName: string;            // VD: 'Đỗ Anh Tùng'
  roleTitle: string;           // VD: 'Fullstack & Matching Systems Engineer'
  status: 'Active' | 'Interviewing' | 'Offered';
  disabilityGroup: string;     // Nhóm khuyết tật từ scripts/data/candidates.csv
  scoreJ: number;              // Điểm phù hợp phía NTD Score_J(j, c) thang 0..100 (VD: 96.5)
  scoreC: number;              // Điểm hài lòng phía Ứng viên Score_C(c, j) thang 0..100 (VD: 94.0)
  surplusScore: number;        // Điểm thặng dư chuẩn hóa hai phía
  stabilityStatus: 'J-Optimal' | 'Top-K Backup' | 'Filtered-Out';
  stabilityLabel: string;      // VD: 'Ổn định tối ưu (J-Optimal)'
  employerRank: number;        // Thứ hạng trong danh sách ưu tiên của Job
  candidateRank: number;       // Thứ hạng của Job trong danh sách ưu tiên của Ứng viên
  skills: string[];            // Tags: Java, Spring Boot, React, Qdrant...
  experienceYears: number;
  expectedSalaryMillion: number;
  location: string;
  workModePreference: 'remote' | 'hybrid' | 'on-site';
  essentialAccessNeeds: string[];
  niceToHaveAccessNeeds: string[];
  explanation: string;
  eBreakdown: ScoreBreakdownE;
  cBreakdown: ScoreBreakdownC;
  matchedJobId: string;
  invitedToInterview?: boolean;
}

export interface HardConstraintsFilterState {
  maxSalaryMillion: number;        // Salary Slider (Triệu VNĐ)
  minExperienceYears: number;      // Dropdown kinh nghiệm tối thiểu
  locations: string[];             // Checkboxes địa điểm / hình thức
  accessibilitySupports: string[]; // Checkboxes thiết bị & hỗ trợ người khuyết tật
  searchKeyword: string;
  onlyJOptimal: boolean;
}

export interface MatchingKpiMetrics {
  totalCandidates: number;      // 48
  passedHardFilter: number;     // 32
  jOptimalStablePairs: number;  // 12
  fillRatePercent: number;      // 75
  blockingPairsCount: number;   // 0
  executionTimeMs: number;
  proposalsCount: number;
  source: 'fastapi-backend' | 'local-gale-shapley';
}

export interface FastApiStableMatchingResponse {
  status: string;
  optimal: string;
  matches: Record<string, string>;
  matched_pairs: Array<{
    candidate_id: string;
    job_id: string;
    c_score: number;
    e_score: number;
    candidate_rank?: number;
    job_rank?: number;
    explanation?: string;
    e_breakdown?: Record<string, number>;
    c_breakdown?: Record<string, number>;
  }>;
  job_assignments: Record<string, string[]>;
  unmatched_candidates: string[];
  unmatched_jobs: string[];
  metrics: {
    total_candidates: number;
    total_jobs: number;
    total_quota: number;
    matched_pairs_count: number;
    blocking_pairs_count: number;
    is_stable: boolean;
    fill_rate: number;
    proposals_count: number;
  };
  execution_time_ms: number;
}
