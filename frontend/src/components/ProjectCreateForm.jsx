import { useState } from "react";
import { createProject } from "../services/api";

export default function ProjectCreateForm({ onCreated }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (!name.trim()) {
      setError("Project name is required.");
      return;
    }
    if (!file) {
      setError("Please choose a CSV dataset.");
      return;
    }
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Only .csv files are supported.");
      return;
    }

    setSubmitting(true);
    try {
      const res = await createProject({ name, problem_description: description, dataset_file: file });
      setName("");
      setDescription("");
      setFile(null);
      e.target.reset();
      onCreated?.(res.data.project);
    } catch (err) {
      const data = err?.response?.data;
      if (data && typeof data === "object") {
        const firstKey = Object.keys(data)[0];
        setError(Array.isArray(data[firstKey]) ? data[firstKey][0] : String(data[firstKey]));
      } else {
        setError("Could not create project. Is the backend running?");
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="create-form" onSubmit={handleSubmit}>
      <h3>New Project</h3>
      <input
        type="text"
        placeholder="Project name"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      <input
        type="text"
        placeholder="Problem description (optional)"
        value={description}
        onChange={(e) => setDescription(e.target.value)}
      />
      <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0] || null)} />
      <button type="submit" disabled={submitting}>
        {submitting ? "Creating…" : "Create & Train"}
      </button>
      {error && <p className="error-text">{error}</p>}
    </form>
  );
}
