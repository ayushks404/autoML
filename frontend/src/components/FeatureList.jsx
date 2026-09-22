export default function FeatureList({ features }) {
  if (!Array.isArray(features) || features.length === 0) {
    return <p className="muted">No feature importance available yet.</p>;
  }

  const maxVal = Math.max(...features.map(([, v]) => Math.abs(v)), 1e-9);

  return (
    <ul className="feature-list">
      {features.map(([name, value], i) => (
        <li key={i} className="feature-row">
          <span className="feature-name">{name}</span>
          <div className="feature-bar-track">
            <div
              className="feature-bar-fill"
              style={{ width: `${(Math.abs(value) / maxVal) * 100}%` }}
            />
          </div>
          <span className="feature-value">{Number(value).toFixed(3)}</span>
        </li>
      ))}
    </ul>
  );
}
