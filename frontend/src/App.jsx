import React, { useEffect, useMemo, useState, useRef } from "react";

import DecisionLog from "./components/DecisionLog";
import EmployeeMoodChart from "./components/EmployeeMoodChart";
import EventFeed from "./components/EventFeed";
import LeaderboardPanel from "./components/LeaderboardPanel";
import RevenueCashChart from "./components/RevenueCashChart";
import RewardChart from "./components/RewardChart";
import StatsGrid from "./components/StatsGrid";
import TrainingEvidence from "./components/TrainingEvidence";
import { api } from "./services/api";
import { connectWS } from "./services/ws";
import { useStore } from "./store";

const MAX_POINTS = 120;
const NAV_ITEMS = ["Dashboard", "Overview", "Analytics", "Decisions", "Market", "Team", "Reports", "Training", "Settings"];

function getReason(action, state, phase) {
  const reasons = {
    "hire_employee": `Need more capacity. Revenue: $${state?.revenue}`,
    "fire_employee": `Cutting costs. Burn rate was high.`,
    "increase_salaries": `Boosting morale which was at ${state?.employee_morale}.`,
    "assign_engineering_task": `Accelerating product progress to meet targets.`,
    "launch_product": `Capitalizing on completed features to boost revenue.`,
    "run_ads": `Scaling customer acquisition.`,
    "negotiate_client": `Securing B2B revenue and investor trust.`,
    "reduce_costs": `Preserving cash runway.`,
    "raise_funding": `Cash reserves running low ($${state?.cash_balance}).`,
    "fix_bug_crisis": `Resolving active crises to prevent churn.`,
    "improve_culture": `Investing in long-term employee satisfaction.`,
    "give_bonuses": `Short-term morale boost to retain team.`,
    "change_roadmap": `Pivoting to align with Board mandate.`
  };
  return reasons[action] || `Strategic decision for ${phase} phase.`;
}

export default function App() {
  const {
    state, mode, done, episodeId, events, decisions, rewards, history, moraleHistory, leaderboard, currentDay, currentPhase,
    setState, setMode, setDone, setEpisodeId, setEvents, setDecisions, setRewards, setHistory, setMoraleHistory, setLeaderboard,
    setCurrentDay, setCurrentPhase, resetSimulation, resetForReplay
  } = useStore();

  const [activeNav, setActiveNav] = useState("Dashboard");
  const [wsStatus, setWsStatus] = useState("Connecting...");
  const [mandate, setMandate] = useState("");
  const [selectedMandate, setSelectedMandate] = useState("");
  const isReplayModeRef = useRef(false);

  const MANDATE_OPTIONS = [
    { label: "Random (env picks)",         value: "" },
    { label: "Maximize Growth",             value: "Maximize Growth: Prioritize product progress and revenue even if burn rate increases." },
    { label: "Cost Efficiency",             value: "Cost Efficiency: Minimize burn rate and preserve cash balance at all costs." },
    { label: "Balanced Stability",          value: "Balanced Stability: Maintain a healthy balance between employee morale and revenue." },
  ];

  useEffect(() => {
    boot();
    const ws = connectWS((data) => {
      if (isReplayModeRef.current) return;
      if (data.type === "state_update") {
        const payload = data.payload;
        setCurrentDay(payload.day);
        setCurrentPhase(payload.phase);
        setState(payload.state);
        setEpisodeId(payload.episode_id ?? null);
        // Update live mandate from backend state
        if (payload.state?.mandate) setMandate(payload.state.mandate);
        setDecisions((d) =>
          [{ day: payload.day, phase: payload.phase, action: payload.action, reason: getReason(payload.action, payload.state, payload.phase) }, ...d].slice(0, 30),
        );
        setHistory((h) =>
          [
            ...h,
            { step: h.length + 1, revenue: payload.state.revenue, cash: payload.state.cash_balance, burn: payload.state.burn_rate },
          ].slice(-MAX_POINTS),
        );
        setMoraleHistory((m) =>
          [...m, { name: `S${payload.day}`, mood: payload.state.employee_morale }].slice(-10),
        );
      }
      if (data.type === "market_event" && data.payload?.event) {
        setEvents((e) => [data.payload.event, ...e].slice(0, 30));
      }
      if (data.type === "reward_update") {
        setRewards((r) => {
          const prevCumulative = r.length > 0 ? r[r.length - 1].cumulative_reward : 0;
          return [...r, { step: r.length + 1, reward: data.payload.reward, cumulative_reward: prevCumulative + data.payload.reward }].slice(-MAX_POINTS);
        });
      }
      if (data.type === "episode_done") {
        setDone(true);
        loadLeaderboard();
      }
    }, setWsStatus);
    return () => ws.close();
  }, []);

  async function boot() {
    try {
      if (useStore.getState().state === null) {
        const current = await api.getState();
        setState(current.data.state);
        // episode_id is now nested under info (AtlasObservation-compatible response shape)
        const episodeId = current.data.info?.episode_id ?? current.data.episode_id ?? null;
        setEpisodeId(episodeId);
        setMoraleHistory([{ name: "S1", mood: current.data.state.employee_morale }]);
      }
      loadLeaderboard();
    } catch (_error) {
      // Keep app responsive even if one boot call fails temporarily.
    }
  }

  async function loadLeaderboard() {
    const lb = await api.leaderboard();
    setLeaderboard(lb.data || []);
  }

  async function onReset() {
    isReplayModeRef.current = false;
    const resetRes = await api.reset(mode, selectedMandate || null);
    resetSimulation(resetRes.data.state, resetRes.data.episode_id);
    // Update mandate badge from fresh reset
    if (resetRes.data.state?.mandate) setMandate(resetRes.data.state.mandate);
    await loadLeaderboard();
  }

  async function onReplay(id) {
    isReplayModeRef.current = true;
    const replayRes = await api.replay(id);
    const steps = replayRes.data || [];
    resetForReplay();

    for (let i = 0; i < steps.length; i += 1) {
      const step = steps[i];
      // Keep replay smooth and readable for demo mode.
      await new Promise((resolve) => setTimeout(resolve, 200));
      if (!isReplayModeRef.current) break; // abort if user reset
      setState(step.state);
      setCurrentDay(step.day);
      setCurrentPhase(step.phase);
      setDecisions((d) => [{ day: step.day, phase: step.phase, action: step.action, reason: getReason(step.action, step.state, step.phase) }, ...d].slice(0, 30));
      setRewards((r) => {
        const prevCumulative = r.length > 0 ? r[r.length - 1].cumulative_reward : 0;
        return [...r, { step: r.length + 1, reward: step.reward, cumulative_reward: prevCumulative + step.reward }].slice(-MAX_POINTS);
      });
      setHistory((h) =>
        [...h, { step: h.length + 1, revenue: step.state.revenue, cash: step.state.cash_balance, burn: step.state.burn_rate }].slice(
          -MAX_POINTS,
        ),
      );
      setMoraleHistory((m) =>
        [...m, { name: `S${step.day}`, mood: step.state.employee_morale }].slice(-10),
      );
      if (step.event) {
        const ev = typeof step.event === 'string' ? { name: step.event, severity: 'info' } : step.event;
        if (ev.name) {
          setEvents((e) => [ev, ...e].slice(0, 30));
        }
      }
    }
    setDone(true);
  }

  function exportCsv() {
    const rows = ["step,revenue,cash"];
    history.forEach((row) => rows.push(`${row.step},${row.revenue},${row.cash}`));
    const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `atlas_metrics_episode_${episodeId || "latest"}.csv`;
    anchor.click();
  }

  const mood = useMemo(() => moraleHistory, [moraleHistory]);

  useEffect(() => {
    const sectionMap = [
      { id: "section-dashboard", nav: "Dashboard" },
      { id: "section-overview", nav: "Overview" },
      { id: "section-analytics", nav: "Analytics" },
      { id: "section-decisions", nav: "Decisions" },
      { id: "section-market", nav: "Market" },
      { id: "section-team", nav: "Team" },
      { id: "section-reports", nav: "Reports" },
      { id: "section-training", nav: "Training" },
      { id: "section-settings", nav: "Settings" },
    ];

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        const match = sectionMap.find((section) => section.id === visible.target.id);
        if (match) setActiveNav(match.nav);
      },
      { threshold: [0.25, 0.5, 0.75], rootMargin: "-10% 0px -50% 0px" },
    );

    sectionMap.forEach((section) => {
      const node = document.getElementById(section.id);
      if (node) observer.observe(node);
    });

    return () => observer.disconnect();
  }, []);

  function goToSection(item) {
    const targetId = `section-${item.toLowerCase()}`;
    const node = document.getElementById(targetId);
    if (node) {
      node.scrollIntoView({ behavior: "smooth", block: "start" });
      setActiveNav(item);
    }
  }

  return (
    <div className="dashboard-shell">
      <div className="dashboard-grid">
        <aside className="sidebar-panel">
          <div className="mb-8">
            <div className="text-2xl font-bold tracking-wide text-cyan-300">ATLAS</div>
            <div className="mt-1 text-xs text-slate-400">AI CEO Dashboard</div>
          </div>

          <nav className="space-y-2 text-sm">
            {NAV_ITEMS.map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => goToSection(item)}
                className={`w-full rounded-xl border px-3 py-2 text-left transition ${activeNav === item
                    ? "border-cyan-400/60 bg-cyan-500/10 text-cyan-200"
                    : "border-transparent text-slate-300 hover:border-slate-600 hover:bg-slate-800/30"
                  }`}
              >
                {item}
              </button>
            ))}
          </nav>

          <div className={`mt-auto space-y-2 rounded-xl border p-3 text-xs ${wsStatus === 'Online' ? 'border-emerald-400/20 bg-emerald-500/5' : 'border-amber-400/20 bg-amber-500/5'}`}>
            <div className={wsStatus === 'Online' ? 'text-emerald-300' : 'text-amber-300'}>System Status</div>
            <div className={`font-medium ${wsStatus === 'Online' ? 'text-emerald-200' : 'text-amber-200'}`}>{wsStatus}</div>
          </div>
        </aside>

        <main className="main-panel">
          <header id="section-dashboard" className="glass-card">
            <div className="flex flex-wrap items-center justify-between gap-3">
              {/* Left side: Status indicators */}
              <div className="flex flex-wrap items-center gap-2">
                <div className="rounded-xl border border-violet-400/30 bg-violet-500/10 px-4 py-2 text-sm text-violet-100">
                  {isReplayModeRef.current ? "[REPLAY] " : ""}Day {currentDay} &middot; {currentPhase} Phase
                </div>
                {mandate && (
                  <div className="rounded-xl border border-amber-400/40 bg-amber-500/10 px-4 py-2 text-sm font-medium text-amber-200"
                       title={mandate}>
                    📋 Mandate: {mandate.split(":")[0]}
                  </div>
                )}
              </div>

              {/* Right side: Controls */}
              <div className="flex flex-wrap items-center gap-2">
                <div className="flex items-center gap-1 rounded bg-slate-800/50 p-1">
                  <button className="rounded px-2 py-1 text-xs text-slate-300 hover:bg-slate-700 hover:text-white" onClick={() => api.pause()}>⏸ Pause</button>
                  <button className="rounded px-2 py-1 text-xs text-slate-300 hover:bg-slate-700 hover:text-white" onClick={() => api.resume()}>▶ Resume</button>
                  <button className="rounded px-2 py-1 text-xs text-slate-300 hover:bg-slate-700 hover:text-white" onClick={() => api.speed(1.0)}>1x</button>
                  <button className="rounded px-2 py-1 text-xs text-slate-300 hover:bg-slate-700 hover:text-white" onClick={() => api.speed(2.0)}>2x</button>
                  <button className="rounded px-2 py-1 text-xs text-slate-300 hover:bg-slate-700 hover:text-white" onClick={() => api.speed(5.0)}>5x</button>
                </div>
                <select
                  className="top-button"
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                >
                  <option value="startup">Startup</option>
                  <option value="crisis">Crisis</option>
                  <option value="growth">Growth</option>
                </select>
                <button className="top-button" onClick={onReset}>
                  Reset Simulation
                </button>
                <button className="top-button" onClick={exportCsv}>
                  Download CSV
                </button>
                {episodeId && (
                  <a className="top-button-primary" href={api.investorReport(episodeId)} target="_blank" rel="noopener noreferrer">
                    Generate Report
                  </a>
                )}
              </div>
            </div>
          </header>

          <section id="section-overview">
            <StatsGrid state={state} />
          </section>

          <section id="section-analytics">
            <div className="grid gap-4 lg:grid-cols-2">
              <RevenueCashChart data={history} />
              <RewardChart data={rewards} />
            </div>
          </section>

          <section id="section-team">
            <EmployeeMoodChart
              data={mood}
              currentMorale={state?.employee_morale ?? 0}
              currentSatisfaction={state?.customer_satisfaction ?? 0}
            />
          </section>

          <section id="section-decisions" className="grid gap-4 lg:grid-cols-2">
            <DecisionLog decisions={decisions} done={done} />
            <section id="section-market">
              <EventFeed events={events} />
            </section>
          </section>

          <section id="section-reports">
            <LeaderboardPanel rows={leaderboard} onReplay={onReplay} onPdf={api.investorReport} />
          </section>

          <section id="section-training">
            <TrainingEvidence />
          </section>

          <section id="section-settings" className="glass-card">
            <div className="panel-title mb-4">Settings</div>

            <div className="space-y-6">
              {/* Board Mandate Selector */}
              <div className="rounded-xl border border-slate-700/50 bg-slate-800/20 p-5">
                <label className="mb-2 flex items-center gap-2 text-sm font-semibold text-amber-300">
                  <span className="text-lg">📋</span> Board Mandate Override
                </label>
                <p className="mb-4 text-xs text-slate-400">
                  Select the strategic directive the AI CEO must follow. This overrides the default random selection and takes effect on the next Reset.
                </p>
                <div className="grid gap-3 sm:grid-cols-2">
                  {MANDATE_OPTIONS.map((opt) => (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setSelectedMandate(opt.value)}
                      className={`flex flex-col justify-start rounded-xl border px-4 py-3 text-left transition ${
                        selectedMandate === opt.value
                          ? "border-amber-400/70 bg-amber-500/20 text-amber-100 shadow-[0_0_15px_rgba(251,191,36,0.15)]"
                          : "border-slate-700/50 bg-slate-800/40 text-slate-300 hover:border-amber-400/30 hover:bg-slate-800/80"
                      }`}
                    >
                      <div className="font-medium text-sm mb-1">{opt.label}</div>
                      <div className="text-xs text-slate-400 leading-relaxed">
                        {opt.value ? opt.value.split(":")[1]?.trim() : "Let the environment randomly pick a strategic mandate for this episode (Default)."}
                      </div>
                    </button>
                  ))}
                </div>
                <div className="mt-5 flex justify-end">
                  <button
                    type="button"
                    onClick={onReset}
                    className="rounded-lg border border-amber-400/60 bg-amber-500/20 px-5 py-2 text-sm font-semibold text-amber-200 transition hover:bg-amber-500/30 hover:text-amber-100"
                  >
                    Apply Mandate &amp; Reset Simulation
                  </button>
                </div>
              </div>

              {/* Current Active Mandate */}
              {mandate && (
                <div className="flex items-start gap-3 rounded-xl border border-amber-400/30 bg-amber-500/10 p-4">
                  <div className="text-2xl">🎯</div>
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-wider text-amber-400/80 mb-1">Active Board Mandate</div>
                    <div className="text-sm font-medium text-amber-100">{mandate.split(":")[0]}</div>
                    {mandate.includes(":") && (
                      <div className="mt-1 text-xs text-amber-200/70">{mandate.split(":")[1]?.trim()}</div>
                    )}
                  </div>
                </div>
              )}

              {/* Other controls */}
              <div>
                <label className="mb-2 block text-sm font-semibold text-slate-300">Other Controls</label>
                <p className="text-sm text-slate-400">Use the controls in the top bar to change mode, reset, export CSV, and generate report.</p>
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
