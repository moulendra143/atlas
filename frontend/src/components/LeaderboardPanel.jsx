import React from "react";

export default function LeaderboardPanel({ rows, onReplay, onPdf }) {
  return (
    <div className="glass-card overflow-auto">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="panel-title mb-0">AI Agents Leaderboard</h2>
        <span className="text-xs text-slate-400">Top Episodes</span>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="text-xs uppercase tracking-wider text-slate-400">
            <tr>
              <th className="px-2 py-2">Rank</th>
              <th className="px-2 py-2">Episode</th>
              <th className="px-2 py-2">Mode</th>
              <th className="px-2 py-2">Total Reward</th>
              <th className="px-2 py-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            {Array.isArray(rows) &&
              rows.map((row, idx) => (
                <tr key={row.id} className="border-t border-slate-800/80">
                  <td className="px-2 py-2 text-cyan-300">{idx + 1}</td>
                  <td className="px-2 py-2">#{row.id}</td>
                  <td className="px-2 py-2 capitalize text-slate-300">{row.mode}</td>
                  <td className="px-2 py-2 text-emerald-300">{row.total_reward.toFixed(2)}</td>
                  <td className="px-2 py-2">
                    <div className="flex gap-2">
                      <button
                        className="rounded-lg border border-cyan-400/30 bg-cyan-500/10 px-2 py-1 text-xs text-cyan-200"
                        onClick={() => onReplay(row.id)}
                      >
                        Replay
                      </button>
                      <a
                        className="rounded-lg border border-violet-400/30 bg-violet-500/10 px-2 py-1 text-xs text-violet-200"
                        href={onPdf(row.id)}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        PDF
                      </a>
                    </div>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
