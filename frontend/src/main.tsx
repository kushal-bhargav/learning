import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { api, GiftSessionResponse, HumanAction, PersonaSummary, StageLogEntry, StageName } from './api';
import { replaySession } from './replaySession';
import './styles.css';

type Screen = 'setup' | 'console' | 'ledger' | 'feedback' | 'replay';

const REPLAY_ONLY = new URLSearchParams(window.location.search).has('replay') || window.location.hash === '#replay';

const STAGES: Array<{ id: StageName; title: string; system: string; promise: string }> = [
  { id: 'recipient_profiling', title: 'Recipient Profiling', system: 'Ollama · Instructor', promise: 'What do we know about the recipient?' },
  { id: 'relationship_analysis', title: 'Relationship Analysis', system: 'Ollama · smolagents', promise: 'How should the gift sound and feel?' },
  { id: 'gift_intent_reasoning', title: 'Gift Intent Reasoning', system: 'Hybrid intent layer', promise: 'What is the gift trying to accomplish?' },
  { id: 'multi_agent_planning', title: 'Multi-Agent Planning', system: 'Bounded planner', promise: 'How should the agents solve this request?' },
  { id: 'recommendation', title: 'Gift Recommendation', system: 'Ollama · smolagents', promise: 'Which gift direction should we pursue?' },
  { id: 'creative_generation', title: 'Creative Generation', system: 'MemoryGAN live inference', promise: 'How much creative agency should the model have?' },
  { id: 'greeting_story', title: 'Greeting + Story', system: 'Ollama chat', promise: 'What message travels with the gift?' },
  { id: 'delivery_planner', title: 'Delivery Planner', system: 'Simulated logistics', promise: 'How would this be delivered in the demo?' },
];

const ACTION_LABELS: Record<string, string> = {
  accept: 'Accepted',
  edit: 'Edited',
  regenerate: 'Regenerated',
  delegate: 'Delegated',
  pending: 'Pending',
  completed: 'Completed',
  error: 'Error',
};

function App() {
  const [personas, setPersonas] = useState<PersonaSummary[]>([]);
  const [session, setSession] = useState<GiftSessionResponse | null>(null);
  const [screen, setScreen] = useState<Screen>(REPLAY_ONLY ? 'replay' : 'setup');
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedbackResult, setFeedbackResult] = useState<string | null>(null);

  useEffect(() => {
    if (screen === 'replay') return;
    api.personas().then(setPersonas).catch((err) => setError(err.message));
  }, [screen]);

  async function run<T>(label: string, task: () => Promise<T>): Promise<T | null> {
    setBusy(label);
    setError(null);
    try {
      return await task();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return null;
    } finally {
      setBusy(null);
    }
  }

  async function createSession(payload: Parameters<typeof api.createSession>[0]) {
    const next = await run('Preparing a new gift session…', () => api.createSession(payload));
    if (next) {
      setSession(next);
      setScreen('console');
      setFeedbackResult(null);
    }
  }

  async function updateSession(label: string, task: () => Promise<GiftSessionResponse>) {
    const next = await run(label, task);
    if (next) setSession(next);
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Gift Memory · Agency Console</p>
          <h1>Co-create a gift with a visible hand on the wheel.</h1>
          <p className="hero-copy">
            A warm little cockpit for negotiating authorship: every agent proposal shows its why, every human action is logged, and the creative slider stays tactile.
          </p>
        </div>
        <div className="session-pill">
          <span>{screen === 'replay' ? 'Replay mode' : session ? 'Live session' : 'Ready'}</span>
          <strong>{screen === 'replay' ? replaySession.session_id : session?.session_id ?? 'Choose a persona'}</strong>
        </div>
      </header>

      <nav className="tabs" aria-label="Agency Console screens">
        {(['setup', 'console', 'ledger', 'feedback', 'replay'] as const).map((item) => (
          <button key={item} className={screen === item ? 'active' : ''} onClick={() => setScreen(item)} disabled={!['setup', 'replay'].includes(item) && !session}>
            {item === 'setup' ? 'Session Setup' : item === 'console' ? 'Agency Console' : item === 'ledger' ? 'Agency Ledger' : item === 'feedback' ? 'Feedback' : 'Replay Mode'}
          </button>
        ))}
      </nav>

      {error && <div className="notice error">{error}</div>}
      {busy && <div className="notice busy">{busy}</div>}

      {screen === 'setup' && <SessionSetup personas={personas} onCreate={createSession} />}
      {screen === 'console' && session && <Console session={session} onUpdate={updateSession} goLedger={() => setScreen('ledger')} />}
      {screen === 'ledger' && session && <Ledger session={session} goFeedback={() => setScreen('feedback')} />}
      {screen === 'replay' && <ReplayMode />}
      {screen === 'feedback' && session && (
        <Feedback
          session={session}
          result={feedbackResult}
          onSubmit={async (payload) => {
            const response = await run('Submitting feedback to the bandit…', () => api.submitFeedback(session.session_id, payload));
            if (response) setFeedbackResult(`Reward ${response.reward.toFixed(3)} · ${response.action.agency_bucket} agency arm updated`);
          }}
        />
      )}
    </div>
  );
}

function SessionSetup({ personas, onCreate }: { personas: PersonaSummary[]; onCreate: (payload: Parameters<typeof api.createSession>[0]) => void }) {
  const templates = personas.filter((persona) => persona.persona_id !== 'custom-live');
  const [templateId, setTemplateId] = useState('');
  const [giverName, setGiverName] = useState('');
  const [recipientName, setRecipientName] = useState('');
  const [relationshipType, setRelationshipType] = useState('friend');
  const [closeness, setCloseness] = useState(3);
  const [occasionName, setOccasionName] = useState('Birthday');
  const [occasionDate, setOccasionDate] = useState('2026-12-18');
  const [formality, setFormality] = useState('casual');
  const [budget, setBudget] = useState('USD 60-100');
  const [preferences, setPreferences] = useState('');
  const [memories, setMemories] = useState('');
  const [agency, setAgency] = useState(0.5);
  const [seed, setSeed] = useState(2026);

  function applyTemplate(personaId: string) {
    setTemplateId(personaId);
    const selected = templates.find((persona) => persona.persona_id === personaId);
    const occasion = selected?.occasions[0];
    if (selected) {
      setRecipientName(selected.label);
      setOccasionName(String(occasion?.name ?? occasionName));
      setOccasionDate(String(occasion?.date ?? occasionDate));
      setBudget(String(occasion?.budget_hint ?? budget));
      setFormality(String(occasion?.formality ?? formality));
    }
  }

  function submit() {
    onCreate({
      persona_id: 'custom-live',
      agency_slider: agency,
      seed,
      custom_profile: {
        giver_name: giverName || 'Gift giver',
        recipient_name: recipientName || 'Gift recipient',
        relationship_type: relationshipType,
        closeness_score: closeness,
        occasion_name: occasionName || 'Gift occasion',
        occasion_date: occasionDate || '2026-12-18',
        budget_hint: budget || 'Flexible',
        formality,
        preferences: preferences.split(',').map((item) => item.trim()).filter(Boolean),
        memories: memories.split('\n').map((item) => item.trim()).filter(Boolean),
      },
    });
  }

  return (
    <main className="setup-grid">
      <section className="warm-card setup-card">
        <p className="eyebrow">1 · Session Setup</p>
        <h2>Describe the gift moment.</h2>
        <p className="muted">Enter the real context you want the agents to reason over. The app keeps the run local to this backend session and shows every AI proposal before it can affect the ledger.</p>

        {templates.length > 0 && (
          <label>
            Optional template prefill
            <select value={templateId} onChange={(event) => applyTemplate(event.target.value)}>
              <option value="">Start from blank</option>
              {templates.map((persona) => (
                <option key={persona.persona_id} value={persona.persona_id}>{persona.label}</option>
              ))}
            </select>
          </label>
        )}

        <div className="form-pair">
          <label>
            Your name
            <input value={giverName} onChange={(event) => setGiverName(event.target.value)} placeholder="Asha" />
          </label>
          <label>
            Recipient name
            <input value={recipientName} onChange={(event) => setRecipientName(event.target.value)} placeholder="Mira" />
          </label>
        </div>

        <div className="form-pair">
          <label>
            Relationship
            <select value={relationshipType} onChange={(event) => setRelationshipType(event.target.value)}>
              <option value="partner">Partner</option>
              <option value="parent-child">Parent / child</option>
              <option value="sibling">Sibling</option>
              <option value="friend">Friend</option>
              <option value="colleague">Colleague</option>
              <option value="extended-family">Extended family</option>
              <option value="mentor">Mentor</option>
              <option value="other">Other</option>
            </select>
          </label>
          <label>
            Closeness · {closeness.toFixed(1)} / 5
            <input type="range" min="1" max="5" step="0.5" value={closeness} onChange={(event) => setCloseness(Number(event.target.value))} />
          </label>
        </div>

        <div className="form-pair">
          <label>
            Occasion
            <input value={occasionName} onChange={(event) => setOccasionName(event.target.value)} placeholder="Birthday, graduation, promotion" />
          </label>
          <label>
            Occasion date
            <input type="date" value={occasionDate} onChange={(event) => setOccasionDate(event.target.value)} />
          </label>
        </div>

        <div className="form-pair">
          <label>
            Budget hint
            <input value={budget} onChange={(event) => setBudget(event.target.value)} placeholder="USD 60-100" />
          </label>
          <label>
            Formality
            <select value={formality} onChange={(event) => setFormality(event.target.value)}>
              <option value="casual">Casual</option>
              <option value="semi-formal">Semi-formal</option>
              <option value="professional">Professional</option>
              <option value="ceremonial">Ceremonial</option>
            </select>
          </label>
        </div>

        <label>
          Recipient preferences, comma-separated
          <input value={preferences} onChange={(event) => setPreferences(event.target.value)} placeholder="ceramics, quiet mornings, green, handwritten notes" />
        </label>

        <label>
          Memory notes, one per line
          <textarea value={memories} onChange={(event) => setMemories(event.target.value)} rows={5} placeholder={'They once got lost finding a tiny tea shop.\nThey always send photos of interesting doors.'} />
        </label>

        <label>
          Default agency · {agency.toFixed(2)}
          <input type="range" min="0" max="1" step="0.01" value={agency} onChange={(event) => setAgency(Number(event.target.value))} />
        </label>

        <label>
          Deterministic seed
          <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
        </label>

        <button className="primary" onClick={submit}>
          Begin Agency Console
        </button>
      </section>

      <aside className="memory-board live-board">
        <div className="photo-note">Your shared moments</div>
        <div className="text-note">Preferences, tone, budget</div>
        <div className="photo-note yellow">Live agent reasoning</div>
        <p>The console turns your entered context into profile, relationship, recommendation, image, message, and feedback stages.</p>
      </aside>
    </main>
  );
}
function Console({ session, onUpdate, goLedger }: { session: GiftSessionResponse; onUpdate: (label: string, task: () => Promise<GiftSessionResponse>) => void; goLedger: () => void }) {
  const lastEntry = session.stage_log.length ? session.stage_log[session.stage_log.length - 1] : null;
  const pending = lastEntry?.status === 'pending' ? lastEntry : null;
  const nextStage = session.next_stage;

  return (
    <main className="console-layout">
      <section className="stage-column">
        {STAGES.map((stage) => {
          const latest = latestForStage(session.stage_log, stage.id);
          const isPending = pending?.stage === stage.id;
          const canPropose = !pending && nextStage === stage.id;
          return (
            <StageCard
              key={stage.id}
              stage={stage}
              entry={latest}
              isPending={isPending}
              canPropose={canPropose}
              sessionId={session.session_id}
              onUpdate={onUpdate}
            />
          );
        })}
      </section>
      <aside className="side-panel warm-card">
        <p className="eyebrow">Current authorship</p>
        <LedgerBar session={session} />
        <p className="muted">Green = accept, amber = edit, blue = regenerate, purple = delegate.</p>
        <dl>
          <div><dt>Next visible step</dt><dd>{pending ? `Review ${labelFor(pending.stage)}` : nextStage ? `Ask ${labelFor(nextStage)} to propose` : 'Session complete'}</dd></div>
          <div><dt>Ledger entries</dt><dd>{session.stage_log.length}</dd></div>
        </dl>
        <button className="secondary" onClick={goLedger}>Open Ledger View</button>
      </aside>
    </main>
  );
}

function StageCard({
  stage,
  entry,
  isPending,
  canPropose,
  sessionId,
  onUpdate,
}: {
  stage: (typeof STAGES)[number];
  entry: StageLogEntry | null;
  isPending: boolean;
  canPropose: boolean;
  sessionId: string;
  onUpdate: (label: string, task: () => Promise<GiftSessionResponse>) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState('{}');

  function submitEdit() {
    try {
      const parsed = JSON.parse(editText) as Record<string, unknown>;
      onUpdate('Saving your edit…', () => api.edit(sessionId, stage.id, parsed));
      setEditing(false);
    } catch {
      alert('Edit must be valid JSON, e.g. {"tone":"more playful"}');
    }
  }

  return (
    <article className={`stage-card ${isPending ? 'pending' : ''} ${entry?.status === 'completed' ? 'complete' : ''}`}>
      <div className="stage-head">
        <div>
          <p className="eyebrow">{stage.title} · {stage.system}</p>
          <h3>{stage.promise}</h3>
        </div>
        <span className={`stage-status ${entry?.human_action ?? entry?.status ?? 'waiting'}`}>{entry ? ACTION_LABELS[entry.human_action ?? entry.status] ?? entry.status : 'Waiting'}</span>
      </div>

      {entry ? <Proposal entry={entry} /> : <EmptyProposal />}

      {stage.id === 'creative_generation' && entry && isPending && (
        <AgencySlider sessionId={sessionId} entry={entry} onUpdate={onUpdate} />
      )}

      {canPropose && (
        <button className="primary" onClick={() => onUpdate(`Asking ${stage.title} for a proposal…`, () => api.propose(sessionId, stage.id))}>
          Generate proposal
        </button>
      )}

      {isPending && (
        <div className="actions">
          <button onClick={() => onUpdate('Accepting this proposal…', () => api.accept(sessionId, stage.id))}>Accept</button>
          <button onClick={() => setEditing((value) => !value)}>Edit</button>
          <button onClick={() => onUpdate('Regenerating proposal…', () => api.regenerate(sessionId, stage.id))}>Regenerate</button>
          <button className="delegate" onClick={() => onUpdate('Delegating the remaining stages to AI…', () => api.delegate(sessionId, stage.id))}>Delegate rest to AI</button>
        </div>
      )}

      {editing && (
        <div className="edit-box">
          <label>
            Human edit patch (JSON)
            <textarea value={editText} onChange={(event) => setEditText(event.target.value)} rows={5} />
          </label>
          <button className="primary" onClick={submitEdit}>Commit edit</button>
        </div>
      )}
    </article>
  );
}

function Proposal({ entry }: { entry: StageLogEntry }) {
  return (
    <div className="proposal">
      {entry.rationale && <p className="why"><span>Why</span>{entry.rationale}</p>}
      <OutputView entry={entry} />
      {entry.human_edit && <pre className="human-edit">Human edit: {JSON.stringify(entry.human_edit, null, 2)}</pre>}
    </div>
  );
}

function EmptyProposal() {
  return <div className="empty-proposal">No proposal yet. The next stage waits for a visible action.</div>;
}

function OutputView({ entry }: { entry: StageLogEntry }) {
  const output = entry.output;
  if (entry.stage === 'creative_generation' && typeof output.artifact_path === 'string') {
    return (
      <div className="image-output">
        <img src={api.artifactUrl(output.artifact_path)} alt="Generated gift artifact" />
        <div>
          <strong>Agency {Number(output.agency_slider ?? 0).toFixed(2)}</strong>
          <span>{String(output.width)}×{String(output.height)} · {String(output.media_type ?? 'image')}</span>
        </div>
      </div>
    );
  }
  if (entry.stage === 'recommendation' && Array.isArray(output.recommendations)) {
    return (
      <ol className="recommendations">
        {(output.recommendations as Array<Record<string, unknown>>).map((item) => (
          <li key={String(item.rank)}>
            <strong>{String(item.category)}</strong>
            <p>{String(item.concept)}</p>
            <small>{String(item.budget_fit)}</small>
          </li>
        ))}
      </ol>
    );
  }
  if (entry.stage === 'greeting_story' && typeof output.message === 'string') {
    return <blockquote className="message">{output.message}</blockquote>;
  }
  return <pre>{JSON.stringify(output, null, 2)}</pre>;
}

function AgencySlider({ sessionId, entry, onUpdate }: { sessionId: string; entry: StageLogEntry; onUpdate: (label: string, task: () => Promise<GiftSessionResponse>) => void }) {
  const [value, setValue] = useState(Number(entry.output.agency_slider ?? 0.5));
  const [isDragging, setIsDragging] = useState(false);
  const timer = useRef<number | null>(null);
  const latestRequest = useRef(0);

  useEffect(() => {
    setValue(Number(entry.output.agency_slider ?? 0.5));
  }, [entry.timestamp]);

  function schedule(next: number) {
    setValue(next);
    setIsDragging(true);
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      latestRequest.current += 1;
      const requestId = latestRequest.current;
      onUpdate(`Regenerating image at agency ${next.toFixed(2)}…`, async () => {
        const response = await api.regenerate(sessionId, 'creative_generation', { agency_slider: next });
        if (requestId === latestRequest.current) setIsDragging(false);
        return response;
      });
    }, 650);
  }

  return (
    <div className="agency-slider">
      <div className="slider-copy">
        <span>Human-shaped</span>
        <strong>Agency Slider · {value.toFixed(2)}</strong>
        <span>AI-shaped</span>
      </div>
      <input type="range" min="0" max="1" step="0.01" value={value} onChange={(event) => schedule(Number(event.target.value))} />
      <p>{isDragging ? 'Waiting for your pause before calling MemoryGAN…' : 'Pause after dragging to regenerate; Regenerate also commits immediately.'}</p>
    </div>
  );
}

function ReplayMode() {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(true);
  const maxStep = replaySession.stage_log.length - 1;
  const activeEntry = replaySession.stage_log[step];
  const visibleSession = useMemo<GiftSessionResponse>(() => {
    const stageLog = replaySession.stage_log.slice(0, step + 1);
    return {
      ...replaySession,
      stage_log: stageLog,
      ledger: {
        ...replaySession.ledger,
        timeline: replaySession.ledger.timeline.slice(0, Math.min(step + 1, replaySession.ledger.timeline.length)),
        completed: step === maxStep,
      },
    };
  }, [maxStep, step]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setStep((current) => (current >= maxStep ? 0 : current + 1));
    }, 4200);
    return () => window.clearInterval(timer);
  }, [maxStep, playing]);

  return (
    <main className="replay-screen">
      <section className="replay-hero warm-card">
        <div>
          <p className="eyebrow">Poster replay · offline loop</p>
          <h2>{labelFor(activeEntry.stage)}</h2>
          <p>{activeEntry.rationale ?? 'A recorded agency event from the fixture session.'}</p>
        </div>
        <div className="replay-counter">
          <strong>{step + 1}</strong>
          <span>of {replaySession.stage_log.length}</span>
        </div>
      </section>

      <section className="replay-grid">
        <article className="stage-card replay-focus">
          <div className="stage-head">
            <div>
              <p className="eyebrow">{labelFor(activeEntry.stage)} · recorded {activeEntry.proposed_by}</p>
              <h3>{ACTION_LABELS[activeEntry.human_action ?? activeEntry.status] ?? activeEntry.status}</h3>
            </div>
            <span className={`stage-status ${activeEntry.human_action ?? activeEntry.status}`}>cached</span>
          </div>
          <Proposal entry={activeEntry} />
          <div className="replay-controls">
            <button onClick={() => setStep((current) => Math.max(0, current - 1))}>Previous</button>
            <button className="primary" onClick={() => setPlaying((value) => !value)}>{playing ? 'Pause loop' : 'Resume loop'}</button>
            <button onClick={() => setStep((current) => (current >= maxStep ? 0 : current + 1))}>Next</button>
          </div>
        </article>

        <aside className="warm-card replay-ledger">
          <p className="eyebrow">Agency Ledger</p>
          <LedgerBar session={visibleSession} />
          <div className="timeline compact">
            {visibleSession.ledger.timeline.map((item, index) => (
              <div className={`timeline-item ${item.action}`} key={`${item.stage}-${index}`}>
                <span>{index + 1}</span>
                <div>
                  <strong>{labelFor(item.stage)}</strong>
                  <p>{ACTION_LABELS[item.action] ?? item.action}</p>
                </div>
              </div>
            ))}
          </div>
          <p className="muted">No backend or GPU is used here; the generated artifact is bundled as a cached replay asset for unattended large-screen looping.</p>
        </aside>
      </section>
    </main>
  );
}
function Ledger({ session, goFeedback }: { session: GiftSessionResponse; goFeedback: () => void }) {
  return (
    <main className="ledger-screen warm-card">
      <p className="eyebrow">3 · Agency Ledger</p>
      <h2>A compact authorship trace for this gift.</h2>
      <LedgerBar session={session} />
      <div className="timeline">
        {session.ledger.timeline.map((item, index) => (
          <div className={`timeline-item ${item.action}`} key={`${item.stage}-${index}`}>
            <span>{index + 1}</span>
            <div>
              <strong>{labelFor(item.stage)}</strong>
              <p>{ACTION_LABELS[item.action] ?? item.action} · {item.actor}</p>
              {item.rationale && <small>{item.rationale}</small>}
            </div>
          </div>
        ))}
      </div>
      <button className="primary" onClick={goFeedback}>Continue to feedback</button>
    </main>
  );
}

function Feedback({ session, result, onSubmit }: { session: GiftSessionResponse; result: string | null; onSubmit: (payload: { rating: number; authorship?: string; open_text?: string; measures?: Record<string, number> }) => void }) {
  const [rating, setRating] = useState(5);
  const [agency, setAgency] = useState(5);
  const [satisfaction, setSatisfaction] = useState(5);
  const [authorship, setAuthorship] = useState('hybrid');
  const [openText, setOpenText] = useState('');

  return (
    <main className="feedback-screen warm-card">
      <p className="eyebrow">4 · Feedback</p>
      <h2>How did the co-creation feel?</h2>
      <Likert label="Overall satisfaction" value={rating} onChange={setRating} />
      <Likert label="I felt agency over the final gift" value={agency} onChange={setAgency} />
      <Likert label="The result felt personal" value={satisfaction} onChange={setSatisfaction} />
      <label>
        Authorship, in your words
        <select value={authorship} onChange={(event) => setAuthorship(event.target.value)}>
          <option value="human">Mostly human-authored</option>
          <option value="hybrid">Hybrid / co-authored</option>
          <option value="ai">Mostly AI-authored</option>
        </select>
      </label>
      <label>
        Open-ended authorship note
        <textarea value={openText} onChange={(event) => setOpenText(event.target.value)} placeholder="What parts felt like yours? What parts felt like the system's?" />
      </label>
      <button className="primary" onClick={() => onSubmit({ rating, authorship, open_text: openText, measures: { agency, satisfaction } })}>
        Submit feedback
      </button>
      {result && <div className="notice busy">{result}</div>}
      <p className="muted">Session: {session.session_id}</p>
    </main>
  );
}

function Likert({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return (
    <label className="likert">
      {label}
      <div>
        {[1, 2, 3, 4, 5].map((score) => (
          <button key={score} type="button" className={score === value ? 'selected' : ''} onClick={() => onChange(score)}>{score}</button>
        ))}
      </div>
    </label>
  );
}

function LedgerBar({ session }: { session: GiftSessionResponse }) {
  const items = session.ledger.timeline.length ? session.ledger.timeline : session.stage_log.map((entry) => ({ action: entry.human_action ?? entry.status }));
  return (
    <div className="ledger-bar" aria-label="Agency Ledger colored bar">
      {items.map((item, index) => <span key={index} className={`segment ${item.action}`} />)}
    </div>
  );
}

function latestForStage(entries: StageLogEntry[], stage: StageName): StageLogEntry | null {
  return [...entries].reverse().find((entry) => entry.stage === stage) ?? null;
}

function labelFor(stage: StageName): string {
  return STAGES.find((item) => item.id === stage)?.title ?? stage;
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);