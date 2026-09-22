export default function InsightBox({ insights }) {
  if (!insights || typeof insights !== "string" || !insights.trim()) {
    return <p className="muted">No insights available yet.</p>;
  }

  const lines = insights.split("\n").filter((l) => l.trim().length > 0);

  return (
    <div className="insight-box">
      {lines.map((line, i) => (
        <p key={i}>{line}</p>
      ))}
    </div>
  );
}
