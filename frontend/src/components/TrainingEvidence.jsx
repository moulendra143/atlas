import React, { useState } from "react";

export default function TrainingEvidence() {
  const [activeTab, setActiveTab] = useState("trained");

  return (
    <div className="glass-card p-6 border border-slate-800 bg-slate-900/50 rounded-xl mb-6">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-white mb-1">
          RL Training Evidence & Learning Outcome
        </h2>
      </div>

      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setActiveTab("baseline")}
          className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
            activeTab === "baseline"
              ? "bg-slate-700 text-white"
              : "bg-slate-800/50 text-slate-400 hover:bg-slate-700/50"
          }`}
        >
          [Baseline]
        </button>
        <button
          onClick={() => setActiveTab("trained")}
          className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
            activeTab === "trained"
              ? "bg-emerald-600 text-white"
              : "bg-slate-800/50 text-slate-400 hover:bg-slate-700/50"
          }`}
        >
          [Trained]
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
        {/* Before Training Box */}
        <div
          className={`p-5 rounded-xl border transition-opacity duration-300 ${
            activeTab === "baseline" ? "border-rose-500/50 bg-rose-950/30" : "border-rose-900/20 bg-rose-950/10 opacity-60"
          }`}
        >
          <h3 className="text-lg font-bold text-rose-300 mb-4">Learning Outcome: Before Training</h3>
          <ul className="space-y-3">
            <li className="flex items-center text-rose-200">
              <span className="mr-3 text-rose-500 font-bold text-lg">✕</span> Random decisions
            </li>
            <li className="flex items-center text-rose-200">
              <span className="mr-3 text-rose-500 font-bold text-lg">✕</span> High burn rate
            </li>
            <li className="flex items-center text-rose-200">
              <span className="mr-3 text-rose-500 font-bold text-lg">✕</span> Low reward
            </li>
          </ul>
        </div>

        {/* After Training Box */}
        <div
          className={`p-5 rounded-xl border transition-opacity duration-300 ${
            activeTab === "trained" ? "border-emerald-500/50 bg-emerald-950/30" : "border-emerald-900/20 bg-emerald-950/10 opacity-60"
          }`}
        >
          <h3 className="text-lg font-bold text-emerald-300 mb-4">Learning Outcome: After Training</h3>
          <ul className="space-y-3">
            <li className="flex items-center text-emerald-100">
              <span className="mr-3 text-emerald-400 font-bold text-lg">✓</span> Strategic planning
            </li>
            <li className="flex items-center text-emerald-100">
              <span className="mr-3 text-emerald-400 font-bold text-lg">✓</span> Stable growth
            </li>
            <li className="flex items-center text-emerald-100">
              <span className="mr-3 text-emerald-400 font-bold text-lg">✓</span> Higher reward
            </li>
          </ul>
        </div>
      </div>

      {/* Plots Section */}
      {activeTab === "baseline" ? (
        <div className="grid grid-cols-1 gap-4">
          <div className="border border-slate-700/50 rounded-lg bg-slate-900/80 p-4 flex flex-col items-center">
            <div className="text-xs font-bold text-slate-300 mb-2 uppercase tracking-wider">Untrained Baseline Performance (Random vs Heuristic)</div>
            <img
              src="/training_plots/reward_curve.png"
              alt="Baseline Reward Curve"
              className="w-full h-auto rounded object-contain bg-white"
              style={{ maxHeight: "300px", maxWidth: "800px" }}
            />
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="border border-slate-700/50 rounded-lg bg-slate-900/80 p-3 flex flex-col items-center">
            <div className="text-xs font-bold text-slate-300 mb-2 uppercase tracking-wider">GRPO Reward Curve</div>
            <img
              src="/training_plots/trl_grpo_reward_curve.png"
              alt="GRPO Reward Curve"
              className="w-full h-auto rounded object-contain bg-white"
              style={{ maxHeight: "200px" }}
            />
          </div>
          <div className="border border-slate-700/50 rounded-lg bg-slate-900/80 p-3 flex flex-col items-center">
            <div className="text-xs font-bold text-slate-300 mb-2 uppercase tracking-wider">SFT Initial Warm-Start</div>
            <img
              src="/training_plots/trl_reward_curve.png"
              alt="SFT Reward Curve"
              className="w-full h-auto rounded object-contain bg-white"
              style={{ maxHeight: "200px" }}
            />
          </div>
          <div className="border border-slate-700/50 rounded-lg bg-slate-900/80 p-3 flex flex-col items-center">
            <div className="text-xs font-bold text-slate-300 mb-2 uppercase tracking-wider">SFT Loss Curve (Convergence)</div>
            <img
              src="/training_plots/trl_loss_curve.png"
              alt="SFT Loss Curve"
              className="w-full h-auto rounded object-contain bg-white"
              style={{ maxHeight: "200px" }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
