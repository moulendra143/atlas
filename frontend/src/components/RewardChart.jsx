import React from "react";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function RewardChart({ data }) {
  return (
    <div className="chart-card">
      <div className="panel-title">AI Performance</div>
      <ResponsiveContainer width="100%" height="90%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id="rewardFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#c084fc" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#c084fc" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="cumulativeFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f472b6" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#f472b6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="4 4" stroke="#1e293b" />
          <XAxis dataKey="step" stroke="#64748b" />
          <YAxis stroke="#64748b" />
          <Tooltip />
          <Area
            type="monotone"
            dataKey="reward"
            name="Step Reward"
            stroke="#c084fc"
            fill="url(#rewardFill)"
            strokeWidth={2.8}
            dot={true}
            activeDot={{ r: 6 }}
            isAnimationActive
            animationDuration={280}
            animationEasing="linear"
          />
          <Area
            type="monotone"
            dataKey="cumulative_reward"
            name="Cumulative Reward"
            stroke="#f472b6"
            fill="url(#cumulativeFill)"
            strokeWidth={2}
            dot={true}
            activeDot={{ r: 6 }}
            isAnimationActive
            animationDuration={280}
            animationEasing="linear"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
