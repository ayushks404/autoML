import ExperimentalTable from "./ExperimentalTable";
import FeatureList from "./FeatureList";
import InsightBox from "./InsightBox";
import StatusBadge from "./StatusBadge";

function formatMetric(name, value) {
  if (value == null) return "N/A";
  if (name === "rmse") return Number(value).toFixed(2);
  return `${(value * 100).toFixed(2)}%`;
}

export default function ProjectCard({ project }) {
  const {
    name,
    status,
    problem_description,
    task_type,
    target_column,
    best_model,
    metric_name,
    metric_value,
    top_features,
    insights,
    error_message,
    experiments,
  } = project;

  return (
    <div className="project-card">
      <div className="project-card-header">
        <h2>{name}</h2>
        <StatusBadge status={status} />
      </div>

      {problem_description && <p className="muted">{problem_description}</p>}

      {status === "training" && (
        <p className="training-note">Training in progress — this updates automatically.</p>
      )}

      {status === "failed" && (
        <div className="error-box">
          <strong>Training failed.</strong>
          <p>{error_message || "Unknown error."}</p>
        </div>
      )}

      {status === "completed" && (
        <>
          <div className="metrics-row">
            <div>
              <span className="metric-label">Task</span>
              <span className="metric-value">{task_type || "—"}</span>
            </div>
            <div>
              <span className="metric-label">Target</span>
              <span className="metric-value">{target_column || "—"}</span>
            </div>
            <div>
              <span className="metric-label">Best model</span>
              <span className="metric-value">{best_model || "—"}</span>
            </div>
            <div>
              <span className="metric-label">{metric_name || "score"}</span>
              <span className="metric-value">{formatMetric(metric_name, metric_value)}</span>
            </div>
          </div>

          <h4>Top Features</h4>
          <FeatureList features={top_features} />

          <h4>Model Leaderboard</h4>
          <ExperimentalTable experiments={experiments} bestModel={best_model} />

          <h4>Insights</h4>
          <InsightBox insights={insights} />
        </>
      )}
    </div>
  );
}
