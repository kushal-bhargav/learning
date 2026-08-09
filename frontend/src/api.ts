export type HumanAction = 'accept' | 'edit' | 'regenerate' | 'delegate';
export type StageStatus = 'pending' | 'completed' | 'error';
export type Actor = 'agent' | 'human';

export type StageName =
  | 'recipient_profiling'
  | 'relationship_analysis'
  | 'gift_intent_reasoning'
  | 'multi_agent_planning'
  | 'recommendation'
  | 'creative_generation'
  | 'greeting_story'
  | 'delivery_planner';

export interface PersonaSummary {
  persona_id: string;
  label: string;
  synthetic: boolean;
  occasions: Array<Record<string, unknown> & { id: string; name?: string; budget_hint?: string }>;
}

export interface StageLogEntry {
  stage: StageName;
  proposed_by: Actor;
  output: Record<string, unknown>;
  human_action: HumanAction | null;
  human_edit: Record<string, unknown> | null;
  confidence?: number | null;
  rationale?: string | null;
  timestamp: string;
  status: StageStatus;
}

export interface LedgerItem {
  stage: StageName;
  actor: Actor;
  action: HumanAction | 'pending' | 'completed' | 'error';
  status: StageStatus;
  timestamp: string;
  rationale?: string | null;
}

export interface LedgerSummary {
  session_id: string;
  timeline: LedgerItem[];
  counts: Record<string, number>;
  authorship: 'ai' | 'human' | 'hybrid' | string;
  stage_count: number;
  completed: boolean;
}

export interface GiftSessionResponse {
  session_id: string;
  giver_id: string;
  recipient_id: string;
  occasion_id: string;
  stage_log: StageLogEntry[];
  next_stage: StageName | null;
  ledger: LedgerSummary;
}

export interface LiveProfilePayload {
  giver_name: string;
  recipient_name: string;
  relationship_type: string;
  closeness_score: number;
  occasion_name: string;
  occasion_date: string;
  budget_hint: string;
  formality: string;
  preferences: string[];
  memories: string[];
}
export interface FeedbackResponse {
  session_id: string;
  reward: number;
  action: {
    recommendation_category: string;
    agency_bucket: string;
    style_archetype: string;
  };
  bandit_counts: Record<string, number>;
}

const explicitApiBase = import.meta.env.VITE_API_BASE;
const localBackend = ['localhost', '127.0.0.1'].includes(window.location.hostname)
  ? 'http://127.0.0.1:8000'
  : '';
const API_BASE = (explicitApiBase ?? localBackend).replace(/\/$/, '');

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true', ...(options.headers ?? {}) },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      const bodyDetail = body.detail ?? detail;
      detail = typeof bodyDetail === 'string' ? bodyDetail : JSON.stringify(bodyDetail);
    } catch {
      // Keep the HTTP text fallback.
    }
    throw new Error(String(detail));
  }
  return response.json() as Promise<T>;
}

export const api = {
  baseUrl: API_BASE,
  artifactUrl(path: string): string {
    if (path.startsWith('/')) return path;
    return `${API_BASE}/artifacts/${path.split('/').map(encodeURIComponent).join('/')}`;
  },
  personas(): Promise<PersonaSummary[]> {
    return request('/personas');
  },
  createSession(payload: {
    persona_id: string;
    occasion_id?: string | null;
    budget_hint?: string | null;
    agency_slider?: number | null;
    seed?: number;
    custom_profile?: LiveProfilePayload | null;
  }): Promise<GiftSessionResponse> {
    return request('/sessions', { method: 'POST', body: JSON.stringify(payload) });
  },
  getSession(sessionId: string): Promise<GiftSessionResponse> {
    return request(`/sessions/${sessionId}`);
  },
  propose(sessionId: string, stage: StageName, overrides: Record<string, unknown> = {}): Promise<GiftSessionResponse> {
    return request(`/sessions/${sessionId}/stages/${stage}/propose`, {
      method: 'POST',
      body: JSON.stringify({ overrides }),
    });
  },
  accept(sessionId: string, stage: StageName): Promise<GiftSessionResponse> {
    return request(`/sessions/${sessionId}/stages/${stage}/accept`, { method: 'POST', body: '{}' });
  },
  edit(sessionId: string, stage: StageName, humanEdit: Record<string, unknown>): Promise<GiftSessionResponse> {
    return request(`/sessions/${sessionId}/stages/${stage}/edit`, {
      method: 'POST',
      body: JSON.stringify({ human_edit: humanEdit }),
    });
  },
  regenerate(sessionId: string, stage: StageName, overrides: Record<string, unknown> = {}): Promise<GiftSessionResponse> {
    return request(`/sessions/${sessionId}/stages/${stage}/regenerate`, {
      method: 'POST',
      body: JSON.stringify({ overrides }),
    });
  },
  delegate(sessionId: string, stage: StageName): Promise<GiftSessionResponse> {
    return request(`/sessions/${sessionId}/stages/${stage}/delegate`, { method: 'POST', body: '{}' });
  },
  submitFeedback(sessionId: string, payload: {
    rating: number;
    authorship?: string;
    open_text?: string;
    measures?: Record<string, number>;
  }): Promise<FeedbackResponse> {
    return request(`/sessions/${sessionId}/feedback`, { method: 'POST', body: JSON.stringify(payload) });
  },
};









