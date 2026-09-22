const STATUS_META = {
  created: { label: "Created", className: "badge badge-created" },
  training: { label: "Training…", className: "badge badge-training" },
  completed: { label: "Completed", className: "badge badge-completed" },
  failed: { label: "Failed", className: "badge badge-failed" },
};

export default function StatusBadge({ status }) {
  const meta = STATUS_META[status] || { label: status, className: "badge" };
  return <span className={meta.className}>{meta.label}</span>;
}
