import React from "react";
import {
  AbsoluteFill,
  Easing,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

const points = [0, 12, 19, 28, 39, 51, 64, 78, 94, 112, 132];

export const Infographic: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = interpolate(frame, [0, 18], [0, 1], {
    extrapolateRight: "clamp",
  });
  const chartProgress = interpolate(frame, [3 * fps, 10 * fps], [0, 1], {
    easing: Easing.out(Easing.cubic),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const countProgress = interpolate(frame, [fps, 8 * fps], [0, 1], {
    easing: Easing.out(Easing.quad),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const population = Math.round(1_000_000 + 250_000 * countProgress);
  const conclusionOpacity = interpolate(frame, [11 * fps, 12 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const chartWidth = 760;
  const chartHeight = 430;
  const maxValue = 140;
  const path = points
    .map((value, index) => {
      const x = (index / (points.length - 1)) * chartWidth;
      const y = chartHeight - (value / maxValue) * chartHeight;
      return `${index === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  const visiblePoints = Math.max(1, Math.ceil(chartProgress * points.length));

  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#101b2a",
        color: "#f5f7fa",
        fontFamily: "Arial, sans-serif",
        alignItems: "center",
        padding: "130px 90px",
        boxSizing: "border-box",
      }}
    >
      <div style={{ opacity: titleOpacity, textAlign: "center" }}>
        <div style={{ color: "#79c2d0", fontSize: 24, letterSpacing: 7, fontWeight: 700 }}>
          A SIMPLE DATA STORY
        </div>
        <div style={{ fontSize: 66, fontWeight: 800, marginTop: 24 }}>
          How a city grows
        </div>
        <div style={{ color: "#aebdca", fontSize: 25, marginTop: 16 }}>
          An illustrative population-growth animation
        </div>
      </div>

      <div style={{ marginTop: 100, textAlign: "center" }}>
        <div style={{ color: "#aebdca", fontSize: 23, letterSpacing: 3 }}>
          POPULATION
        </div>
        <div
          style={{
            color: "#f4bd65",
            fontSize: 82,
            fontWeight: 800,
            fontVariantNumeric: "tabular-nums",
            marginTop: 14,
          }}
        >
          {population.toLocaleString("en-US")}
        </div>
        <div style={{ color: "#aebdca", fontSize: 22, marginTop: 8 }}>
          +25% over the period
        </div>
      </div>

      <div style={{ width: chartWidth, marginTop: 80 }}>
        <svg width={chartWidth} height={chartHeight + 70} viewBox={`0 0 ${chartWidth} ${chartHeight + 70}`}>
          {[0, 35, 70, 105, 140].map((value) => {
            const y = chartHeight - (value / maxValue) * chartHeight;
            return (
              <g key={value}>
                <line x1="0" y1={y} x2={chartWidth} y2={y} stroke="#344455" strokeWidth="2" />
                <text x="-12" y={y + 7} fill="#aebdca" fontSize="18" textAnchor="end">
                  {value}
                </text>
              </g>
            );
          })}
          <path d={path} fill="none" stroke="#79c2d0" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round" pathLength="1" strokeDasharray="1" strokeDashoffset={1 - chartProgress} />
          {points.slice(0, visiblePoints).map((value, index) => {
            const x = (index / (points.length - 1)) * chartWidth;
            const y = chartHeight - (value / maxValue) * chartHeight;
            return <circle key={index} cx={x} cy={y} r="9" fill="#f4bd65" stroke="#101b2a" strokeWidth="4" />;
          })}
          <text x="0" y={chartHeight + 48} fill="#aebdca" fontSize="19">Year 1</text>
          <text x={chartWidth} y={chartHeight + 48} fill="#aebdca" fontSize="19" textAnchor="end">Year 10</text>
        </svg>
      </div>

      <div
        style={{
          opacity: conclusionOpacity,
          marginTop: 36,
          color: "#f4bd65",
          fontSize: 27,
          fontWeight: 700,
          letterSpacing: 1,
        }}
      >
        Steady growth, visualised with code.
      </div>
    </AbsoluteFill>
  );
};
