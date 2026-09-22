import { useCallback, useEffect, useState } from "react";
import ProjectCard from "../components/ProjectCard";
import ProjectCreateForm from "../components/ProjectCreateForm";
import { listProjects } from "../services/api";

const POLL_INTERVAL_MS = 4000;

export default function Dashboard() {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState(null);

  const fetchProjects = useCallback(async () => {
    try {
      const res = await listProjects();
      setProjects(res.data);
      setFetchError(null);
    } catch (err) {
      setFetchError("Could not reach the backend at the configured API URL.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchProjects();
    const interval = setInterval(() => {
      // only poll while something could still change, to avoid hammering the API forever
      setProjects((current) => {
        const stillActive = current.some((p) => p.status === "training" || p.status === "created");
        if (stillActive || current.length === 0) fetchProjects();
        return current;
      });
    }, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchProjects]);

  return (
    <div className="dashboard">
      <h1>AI Project Dashboard</h1>
      <p className="muted">Total Projects: {projects.length}</p>

      <ProjectCreateForm onCreated={fetchProjects} />

      {fetchError && <div className="error-box">{fetchError}</div>}

      {loading ? (
        <p className="muted">Loading projects…</p>
      ) : projects.length === 0 ? (
        <p className="muted">No projects yet — create one above to get started.</p>
      ) : (
        <div className="project-list">
          {projects.map((project) => (
            <ProjectCard key={project.id} project={project} />
          ))}
        </div>
      )}
    </div>
  );
}
