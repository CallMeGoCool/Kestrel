import React from "react";
import {
  AbsoluteFill,
  Easing,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

export const CarAnimation: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width, height, durationInFrames } = useVideoConfig();

  const progress = interpolate(frame, [0, durationInFrames - 1], [0, 1], {
    easing: Easing.inOut(Easing.quad),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const carX = interpolate(progress, [0, 1], [-260, width + 260]);
  const elapsed = frame / fps;
  const wheelRotation = (elapsed * 360 * 1.8) % 360;
  const roadOffset = (elapsed * 260) % 180;

  const roadY = height * 0.72;
  const carY = roadY - 112;

  return (
    <AbsoluteFill style={{ backgroundColor: "#eaf1f5", overflow: "hidden" }}>
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: roadY - 5,
          height: 5,
          backgroundColor: "#b8c5cc",
        }}
      />
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: roadY,
          height: height - roadY,
          backgroundColor: "#35434d",
        }}
      />
      {Array.from({ length: 9 }, (_, index) => (
        <div
          key={index}
          style={{
            position: "absolute",
            left: index * 180 - roadOffset,
            top: roadY + 88,
            width: 90,
            height: 8,
            borderRadius: 4,
            backgroundColor: "#f4f0d9",
          }}
        />
      ))}

      <div
        style={{
          position: "absolute",
          left: carX,
          top: carY,
          width: 260,
          height: 112,
        }}
      >
        <svg width="260" height="150" viewBox="0 0 260 150">
          <ellipse cx="130" cy="128" rx="119" ry="9" fill="#18232b" opacity="0.2" />
          <path
            d="M28 72 L48 42 Q55 32 72 32 L164 32 Q179 32 190 47 L218 77 L238 83 Q247 86 247 98 L247 111 L13 111 L13 91 Q13 78 28 72Z"
            fill="#e18a3b"
            stroke="#7b4825"
            strokeWidth="4"
            strokeLinejoin="round"
          />
          <path
            d="M65 39 L157 39 Q169 39 178 53 L193 74 L49 74Z"
            fill="#cce7ef"
            stroke="#7b4825"
            strokeWidth="3"
            strokeLinejoin="round"
          />
          <path d="M126 40 L126 74" stroke="#7b4825" strokeWidth="3" />
          <rect x="16" y="82" width="18" height="12" rx="4" fill="#fff0b3" />
          <rect x="229" y="83" width="13" height="12" rx="4" fill="#d94b45" />
          {[65, 199].map((cx) => (
            <g key={cx} transform={`rotate(${wheelRotation} ${cx} 111)`}>
              <circle cx={cx} cy="111" r="28" fill="#202a30" />
              <circle cx={cx} cy="111" r="15" fill="#cbd3d7" />
              <circle cx={cx} cy="111" r="5" fill="#687780" />
              <path d={`M${cx} 98 V124 M${cx - 13} 111 H${cx + 13}`} stroke="#687780" strokeWidth="3" />
            </g>
          ))}
        </svg>
      </div>

      <div
        style={{
          position: "absolute",
          top: height * 0.16,
          width: "100%",
          textAlign: "center",
          color: "#263944",
          fontFamily: "Arial, sans-serif",
        }}
      >
        <div style={{ fontSize: 26, letterSpacing: 7, fontWeight: 700 }}>
          REMOTION MOTION TEST
        </div>
        <div style={{ fontSize: 18, marginTop: 14, color: "#607782" }}>
          A car built entirely from code
        </div>
      </div>
    </AbsoluteFill>
  );
};
