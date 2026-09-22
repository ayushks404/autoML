export default function ExperimentalTable({ experiments, bestModel }) {
  if (!Array.isArray(experiments) || experiments.length === 0) {
    return <p className="muted">No experiments logged yet.</p>;
  }

  const sorted = [...experiments].sort(
    (a, b) => (b.metric_value ?? -Infinity) - (a.metric_value ?? -Infinity)
  );

  return (
    <table className="experiment-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Metric</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((exp) => (
          <tr key={exp.id} className={exp.model_name === bestModel ? "best-row" : ""}>
            <td>
              {exp.model_name}
              {exp.model_name === bestModel && <span className="best-tag">best</span>}
            </td>
            <td>{exp.metric_name}</td>
            <td>{exp.metric_value != null ? Number(exp.metric_value).toFixed(4) : "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
