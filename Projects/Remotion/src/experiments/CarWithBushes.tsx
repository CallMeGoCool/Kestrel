import { useCurrentFrame, useVideoConfig, interpolate } from 'remotion';

export const CarWithBushes: React.FC = () => {
  const { width, height } = useVideoConfig();
  const frame = useCurrentFrame();

  // Car animation
  const carWidth = 100;
  const carHeight = 50;
  const carX = interpolate(frame, [0, 300], [0, width - carWidth]);
  const carY = height - 100; // near the lower-middle

  // Wheel rotation
  const wheelRotation = interpolate(frame, [0, 300], [0, 360]);

  // Road lane markings
  const roadMarkingSpacing = 100;
  const roadMarkingCount = 5;
  const roadMarkingWidth = 2;

  // Bushes configuration
  const bushes = [
    { x: 1080, y: 1800, size: 20, color: '#228B22', shape: 'circle', startFrame: 0 },
    { x: 1080, y: 1750, size: 30, color: '#008000', shape: 'ellipse', startFrame: 50 },
    { x: 1080, y: 1700, size: 15, color: '#006400', shape: 'rectangle', startFrame: 100 },
    { x: 1080, y: 1650, size: 25, color: '#32CD32', shape: 'circle', startFrame: 150 },
    { x: 1080, y: 1600, size: 20, color: '#00FF00', shape: 'ellipse', startFrame: 200 },
    { x: 1080, y: 1550, size: 15, color: '#008000', shape: 'rectangle', startFrame: 250 },
    { x: 1080, y: 1500, size: 20, color: '#228B22', shape: 'circle', startFrame: 0 },
    { x: 1080, y: 1450, size: 25, color: '#00FF00', shape: 'ellipse', startFrame: 50 },
    { x: 1080, y: 1400, size: 30, color: '#008000', shape: 'rectangle', startFrame: 100 },
    { x: 1080, y: 1350, size: 15, color: '#228B22', shape: 'circle', startFrame: 150 },
  ];

  return (
    <svg width={width} height={height} viewBox="0 0 1080 1920">
      {/* Background */}
      <rect x="0" y="0" width="1080" height="1920" fill="#87CEEB" />

      {/* Clouds */}
      <ellipse cx="500" cy="200" rx="50" ry="30" fill="#FFFFFF" />
      <ellipse cx="600" cy="250" rx="40" ry="25" fill="#FFFFFF" />

      {/* Hills */}
      <rect x="0" y="1800" width="1080" height="120" fill="#228B22" />

      {/* Road */}
      <rect x="0" y="1800" width="1080" height="120" fill="#8B4513" />

      {/* Lane markings */}
      {[...Array(roadMarkingCount)].map((_, i) => {
        const x = interpolate(
          frame,
          [0, 300],
          [width - 20 - i * roadMarkingSpacing, width - 20 - i * roadMarkingSpacing - roadMarkingWidth]
        );
        return (
          <line
            key={i}
            x1={x}
            y1="1860"
            x2={x}
            y2="1880"
            stroke="white"
            strokeWidth="2"
            strokeDasharray="6,6"
          />
        );
      })}

      {/* Car */}
      <g>
        {/* Car body */}
        <rect x={carX} y={carY} width={carWidth} height={carHeight} fill="#8B4513" />
        {/* Front part */}
        <polygon
          points={`${carX + 20},${carY} ${carX + 50},${carY + carHeight} ${carX + 40},${carY + carHeight + 20}`}
          fill="#8B4513"
        />
        {/* Wheels */}
        <g transform={`rotate(${wheelRotation}, ${carX + 20}, ${carY + carHeight})`}>
          <circle cx={0} cy={0} r={15} fill="#000000" />
        </g>
        <g transform={`rotate(${wheelRotation}, ${carX + 80}, ${carY + carHeight})`}>
          <circle cx={0} cy={0} r={15} fill="#000000" />
        </g>
      </g>

      {/* Bushes */}
      {bushes.map((bush, i) => {
        const bushDuration = 300 - bush.startFrame;
        const bushX = interpolate(
          frame,
          [bush.startFrame, bush.startFrame + bushDuration],
          [1080, 0 - bush.size]
        );
        const bushOpacity = interpolate(
          frame,
          [bush.startFrame, bush.startFrame + bushDuration],
          [1, 0]
        );
        return (
          <g key={i} opacity={bushOpacity}>
            {bush.shape === 'circle' ? (
              <circle
                cx={bushX}
                cy={bush.y}
                r={bush.size}
                fill={bush.color}
              />
            ) : bush.shape === 'ellipse' ? (
              <ellipse
                cx={bushX}
                cy={bush.y}
                rx={bush.size}
                ry={bush.size * 0.5}
                fill={bush.color}
              />
            ) : bush.shape === 'rectangle' ? (
              <rect
                x={bushX - bush.size / 2}
                y={bush.y - bush.size / 2}
                width={bush.size}
                height={bush.size}
                fill={bush.color}
              />
            ) : null}
          </g>
        );
      })}
    </svg>
  );
};