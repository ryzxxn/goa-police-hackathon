import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const ScoreChart = ({ reportData }:any) => {
  if (!reportData.length || !Array.isArray(reportData[0].frame_timeline)) {
    return <div>{reportData[0].frame_timeline}</div>; // Handle the case where there is no data
  }

  // Prepare data for the chart
  const data = reportData[0].frame_timeline.map((item:any, index:any) => ({
    frame: index,
    score: item.score,
  }));

  // Calculate total scores
  const totalScores = data.reduce((acc:any, curr:any) => acc + curr.score, 0);

  return (
    <div className="px-6 shadow-md bg-gray-950 border-white border-2 flex-1">
      <h2 className="text-gray-800">Total number of frames</h2>
      <h3 className="flex items-baseline gap-1 text-m tabular-nums text-gray-100">
        Frames
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <XAxis dataKey="frame" />
          <YAxis domain={[0, 'dataMax']} />
          <Tooltip
            formatter={(value, name) => {
              return [`Score: ${value}`, 'Total Frames'];
            }}
          />
          <Area
            type="monotone"
            dataKey="score"
            stroke="white"
            fill="url(#fillFrames)"
            fillOpacity={0.4}
          />
          <defs>
            <linearGradient id="fillFrames" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="white" stopOpacity={0.8} />
              <stop offset="95%" stopColor="red" stopOpacity={0.1} />
            </linearGradient>
          </defs>
        </AreaChart>
      </ResponsiveContainer>
      <div className="mt-2 text-gray-100">
        Total Scores: {totalScores}
      </div>
    </div>
  );
};

export default ScoreChart;
